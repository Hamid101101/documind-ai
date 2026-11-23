
from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from evaluation.ingestion.load_scifact import SciFactLoader
from evaluation.metrics.retrieval_metrics import (
    aggregate_metrics,
    evaluate_query,
)
from ingestion.chunking.config import ChunkingConfig
from ingestion.chunking.text_chunker import TextChunker
from ingestion.cleaning.text_cleaner import TextCleaner
from ingestion.loaders.base import DocumentPage, LoadedDocument
from ingestion.embeddings.config import EmbeddingConfig
from ingestion.embeddings.embedding_generator import EmbeddingGenerator


EXPECTED_MODEL = "intfloat/multilingual-e5-base"
K_VALUES = (1, 3, 5, 10)


def validate_dataset(documents, queries, qrels):
    document_ids = {doc["id"] for doc in documents}

    if len(document_ids) != len(documents):
        raise ValueError("Duplicate document IDs detected.")

    if set(queries) != set(qrels):
        raise ValueError("Query IDs and qrels IDs do not match.")

    for query_id, relevant_ids in qrels.items():
        if not relevant_ids:
            raise ValueError(f"Query {query_id} has no relevant documents.")

        missing = relevant_ids - document_ids
        if missing:
            raise ValueError(
                f"Query {query_id} references missing documents: {missing}"
            )


def build_chunks(documents, chunker, cleaner):
    all_chunks = []
    chunk_to_document = {}
    chunk_index = 0

    for document in documents:
        loaded = LoadedDocument(
            source_format="txt",
            page_count=None,
            pages=[
                DocumentPage(
                    page_number=None,
                    text=document["content"],
                )
            ],
        )

        cleaned = cleaner.clean_document(loaded)
        document_chunks = chunker.chunk_document(cleaned)

        if not document_chunks:
            raise ValueError(
                f"No chunks generated for document {document['id']}"
            )

        for chunk in document_chunks:
            chunk.chunk_index = chunk_index
            chunk.metadata = {
                "source": "scifact",
                "document_id": document["id"],
                "title": document["title"],
            }

            all_chunks.append(chunk)
            chunk_to_document[chunk_index] = document["id"]
            chunk_index += 1

    return all_chunks, chunk_to_document


def encode_queries(embedder, queries):
    config = embedder.config

    query_ids = list(queries.keys())
    query_texts = [
        f"{config.query_prefix}{queries[qid].strip()}"
        for qid in query_ids
    ]

    embeddings = embedder.model.encode(
        query_texts,
        batch_size=config.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=config.normalize_embeddings,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    expected_shape = (len(query_ids), config.expected_dimension)

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected query embedding shape: {embeddings.shape}"
        )

    return query_ids, embeddings


def retrieve_documents(
    query_embeddings,
    chunk_embeddings,
    chunks,
    chunk_to_document,
):
    """
    Score every chunk, then aggregate each document using
    its highest-scoring chunk.

    This produces one ranked result per document.
    """

    similarity_matrix = query_embeddings @ chunk_embeddings.T

    document_results = []

    for query_scores in similarity_matrix:
        document_scores = defaultdict(lambda: -1.0)

        for chunk_index, score in enumerate(query_scores):
            document_id = chunk_to_document[chunk_index]

            if score > document_scores[document_id]:
                document_scores[document_id] = float(score)

        ranked_documents = sorted(
            document_scores.items(),
            key=lambda item: item[1],
            reverse=True,
        )

        document_results.append([
            document_id
            for document_id, _ in ranked_documents
        ])

    return document_results


def main():
    print("\n========== SCIFACT EXPERIMENT B ==========\n")

    # 1. Load dataset
    print("[1/7] Loading SciFact dataset...")

    loader = SciFactLoader()

    documents = loader.load_corpus()
    queries = loader.load_queries()
    qrels = loader.load_qrels()

    validate_dataset(documents, queries, qrels)

    print(f"Documents: {len(documents)}")
    print(f"Queries: {len(queries)}")
    print(f"Relevance associations: {sum(map(len, qrels.values()))}")

    # 2. Initialize embedding model
    print("\n[2/7] Initializing embedding model...")

    config = EmbeddingConfig()

    if config.model_name != EXPECTED_MODEL:
        raise ValueError(
            f"Expected {EXPECTED_MODEL}, got {config.model_name}"
        )

    if config.expected_dimension != 768:
        raise ValueError("Expected embedding dimension 768.")

    embedder = EmbeddingGenerator(config=config)

    print(f"Model: {config.model_name}")
    print(f"Device: {embedder._resolve_device()}")

    # 3. Initialize the exact QASPER chunker
    print("\n[3/7] Initializing QASPER chunking configuration...")

    chunking_config = ChunkingConfig(
        chunk_size=500,
        chunk_overlap=100,
    )

    chunker = TextChunker(
        config=chunking_config,
        model_name=config.model_name,
    )

    cleaner = TextCleaner()

    print(f"Chunk size: {chunker.config.chunk_size}")
    print(f"Chunk overlap: {chunker.config.chunk_overlap}")
    print(f"Separators: {chunker.config.separators}")

    # 4. Clean and chunk documents
    print("\n[4/7] Cleaning and chunking SciFact documents...")

    chunks, chunk_to_document = build_chunks(
        documents,
        chunker,
        cleaner,
    )

    chunk_counts = defaultdict(int)

    for document_id in chunk_to_document.values():
        chunk_counts[document_id] += 1

    counts = np.array(list(chunk_counts.values()))

    print(f"Original documents: {len(documents)}")
    print(f"Generated chunks: {len(chunks)}")
    print(f"Mean chunks per document: {counts.mean():.2f}")
    print(f"Median chunks per document: {np.median(counts):.2f}")
    print(f"Maximum chunks per document: {counts.max()}")

    # Verify chunk lengths against E5's maximum sequence length.
    tokenizer = embedder.model[0].tokenizer
    max_seq_length = embedder.model.max_seq_length

    chunk_lengths = np.array([
        len(
            tokenizer.encode(
                f"{config.document_prefix}{chunk.content}",
                add_special_tokens=True,
                truncation=False,
            )
        )
        for chunk in chunks
    ])

    oversized = int(np.sum(chunk_lengths > max_seq_length))

    print(f"Maximum model sequence length: {max_seq_length}")
    print(f"Chunks exceeding model limit: {oversized}")

    if oversized:
        raise ValueError(
            "Some chunks exceed the model sequence limit. "
            "Inspect chunking before continuing."
        )

    # 5. Generate embeddings
    print("\n[5/7] Generating chunk embeddings...")

    chunk_embeddings = np.asarray(
        embedder.embed_chunks(chunks),
        dtype=np.float32,
    )

    expected_shape = (
        len(chunks),
        config.expected_dimension,
    )

    if chunk_embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected chunk embedding shape: {chunk_embeddings.shape}"
        )

    if not np.isfinite(chunk_embeddings).all():
        raise ValueError("Chunk embeddings contain NaN or infinity.")

    if not np.allclose(
        np.linalg.norm(chunk_embeddings, axis=1),
        1.0,
        atol=1e-3,
    ):
        raise ValueError("Chunk embeddings are not normalized.")

    print(f"Chunk embedding matrix: {chunk_embeddings.shape}")

    print("\nGenerating query embeddings...")

    query_ids, query_embeddings = encode_queries(
        embedder,
        queries,
    )

    # 6. Exact chunk retrieval + document-level aggregation
    print("\n[6/7] Running exact cosine-similarity retrieval...")

    retrieved_results = retrieve_documents(
        query_embeddings=query_embeddings,
        chunk_embeddings=chunk_embeddings,
        chunks=chunks,
        chunk_to_document=chunk_to_document,
    )

    print("Evaluating document-level relevance...")

    query_results = []

    for query_id, retrieved_ids in zip(query_ids, retrieved_results):
        result = evaluate_query(
            retrieved_ids=retrieved_ids,
            relevant_ids=qrels[query_id],
            k_values=K_VALUES,
        )

        query_results.append(result)

    metrics = aggregate_metrics(query_results)

    print("\n========== RETRIEVAL RESULTS ==========")

    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    # 7. Save report
    report = {
        "experiment": "SciFact Experiment B - Chunk-level Retrieval",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "name": "BEIR SciFact",
            "documents": len(documents),
            "queries": len(queries),
            "relevance_associations": sum(
                len(value) for value in qrels.values()
            ),
        },
        "embedding": {
            "model": config.model_name,
            "dimension": config.expected_dimension,
            "normalized": config.normalize_embeddings,
            "document_prefix": config.document_prefix,
            "query_prefix": config.query_prefix,
            "max_sequence_length": max_seq_length,
        },
        "chunking": {
            "chunk_size": chunker.config.chunk_size,
            "chunk_overlap": chunker.config.chunk_overlap,
            "total_chunks": len(chunks),
            "mean_chunks_per_document": float(counts.mean()),
            "median_chunks_per_document": float(np.median(counts)),
            "maximum_chunks_per_document": int(counts.max()),
            "oversized_chunks": oversized,
        },
        "retrieval": {
            "method": "Exact cosine similarity over chunks; max-score aggregation per document",
            "database": None,
            "k_values": list(K_VALUES),
        },
        "metrics": {
            name: float(value)
            for name, value in metrics.items()
        },
    }

    reports_dir = PROJECT_ROOT / "evaluation/reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = (
        reports_dir / f"scifact_experiment_b_{timestamp}.json"
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n========== EXPERIMENT COMPLETED ==========")
    print(f"Report saved to: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
