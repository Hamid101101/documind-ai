
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Resolve the project root and load its .env before importing project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from evaluation.ingestion.load_scifact import SciFactLoader
from evaluation.metrics.retrieval_metrics import (
    aggregate_metrics,
    evaluate_query,
)
from ingestion.chunking.base import DocumentChunk
from ingestion.embeddings.config import EmbeddingConfig
from ingestion.embeddings.embedding_generator import EmbeddingGenerator


EXPECTED_MODEL = "intfloat/multilingual-e5-base"
K_VALUES = (1, 3, 5, 10)


def validate_dataset(documents, queries, qrels):
    """Validate corpus and relevance mappings before evaluation."""

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

    print("Dataset validation passed.")


def build_document_chunks(documents, tokenizer):
    """Wrap complete SciFact documents in DocumentChunk objects."""

    chunks = []

    for index, document in enumerate(documents):
        content = document["content"]

        # Count tokens including the E5 document prefix.
        token_count = len(
            tokenizer.encode(
                f"passage: {content}",
                add_special_tokens=True,
                truncation=False,
            )
        )

        chunks.append(
            DocumentChunk(
                chunk_index=index,
                content=content,
                page_start=None,
                page_end=None,
                token_count=token_count,
                metadata={
                    "source": "scifact",
                    "document_id": document["id"],
                    "title": document["title"],
                },
            )
        )

    return chunks


def encode_queries(embedder, queries):
    """Batch-encode queries using the same configuration as embed_query."""

    config = embedder.config

    query_ids = list(queries.keys())
    query_texts = [
        f"{config.query_prefix}{queries[query_id].strip()}"
        for query_id in query_ids
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
            f"Unexpected query embedding shape: {embeddings.shape}; "
            f"expected {expected_shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Query embeddings contain NaN or infinity.")

    return query_ids, embeddings


def retrieve_top_k(
    query_embeddings: np.ndarray,
    document_embeddings: np.ndarray,
    document_ids: list[str],
    k: int,
):
    """Exact cosine-similarity retrieval for normalized embeddings."""

    # Since both matrices are normalized, dot product equals cosine similarity.
    similarity_matrix = query_embeddings @ document_embeddings.T

    results = []

    for scores in similarity_matrix:
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results.append([document_ids[index] for index in top_indices])

    return results


def main():
    print("\n========== SCIFACT EXPERIMENT A ==========\n")

    # ---------------------------------------------------------
    # 1. Load dataset
    # ---------------------------------------------------------

    print("[1/6] Loading SciFact dataset...")

    loader = SciFactLoader()

    documents = loader.load_corpus()
    queries = loader.load_queries()
    qrels = loader.load_qrels()

    validate_dataset(documents, queries, qrels)

    document_ids = [document["id"] for document in documents]

    print(f"Documents: {len(documents)}")
    print(f"Queries: {len(queries)}")
    print(f"Relevance judgments: {sum(len(v) for v in qrels.values())}")

    # ---------------------------------------------------------
    # 2. Initialize embedding model
    # ---------------------------------------------------------

    print("\n[2/6] Initializing embedding model...")

    config = EmbeddingConfig()

    if config.model_name != EXPECTED_MODEL:
        raise ValueError(
            f"Expected {EXPECTED_MODEL}, but configured model is "
            f"{config.model_name}. Check your .env file."
        )

    if config.expected_dimension != 768:
        raise ValueError(
            f"Expected embedding dimension 768, got "
            f"{config.expected_dimension}."
        )

    embedder = EmbeddingGenerator(config=config)

    # Loading the model also gives us access to its tokenizer.
    model = embedder.model
    tokenizer = model[0].tokenizer
    max_seq_length = model.max_seq_length

    print(f"Model: {config.model_name}")
    print(f"Device: {embedder._resolve_device()}")
    print(f"Embedding dimension: {config.expected_dimension}")
    print(f"Maximum sequence length: {max_seq_length}")

    # ---------------------------------------------------------
    # 3. Measure document truncation
    # ---------------------------------------------------------

    print("\n[3/6] Preparing documents and measuring token lengths...")

    chunks = build_document_chunks(documents, tokenizer)

    token_counts = np.array(
        [chunk.token_count for chunk in chunks],
        dtype=np.int32,
    )

    truncated_documents = int(np.sum(token_counts > max_seq_length))

    print(f"Documents: {len(chunks)}")
    print(f"Mean token count: {token_counts.mean():.1f}")
    print(f"Median token count: {np.median(token_counts):.1f}")
    print(f"Maximum token count: {token_counts.max()}")
    print(
        f"Documents exceeding model limit: "
        f"{truncated_documents} "
        f"({truncated_documents / len(chunks) * 100:.2f}%)"
    )

    # ---------------------------------------------------------
    # 4. Generate document and query embeddings
    # ---------------------------------------------------------

    print("\n[4/6] Generating document embeddings...")

    document_embeddings = np.asarray(
        embedder.embed_chunks(chunks),
        dtype=np.float32,
    )

    if document_embeddings.shape != (
        len(documents),
        config.expected_dimension,
    ):
        raise ValueError(
            f"Unexpected document embedding shape: "
            f"{document_embeddings.shape}"
        )

    if not np.isfinite(document_embeddings).all():
        raise ValueError("Document embeddings contain NaN or infinity.")

    document_norms = np.linalg.norm(document_embeddings, axis=1)

    if not np.allclose(document_norms, 1.0, atol=1e-3):
        raise ValueError("Document embeddings are not normalized.")

    print(f"Document embedding matrix: {document_embeddings.shape}")

    print("\nGenerating query embeddings...")

    query_ids, query_embeddings = encode_queries(embedder, queries)

    query_norms = np.linalg.norm(query_embeddings, axis=1)

    if not np.allclose(query_norms, 1.0, atol=1e-3):
        raise ValueError("Query embeddings are not normalized.")

    print(f"Query embedding matrix: {query_embeddings.shape}")

    # ---------------------------------------------------------
    # 5. Exact retrieval
    # ---------------------------------------------------------

    print("\n[5/6] Running exact cosine-similarity retrieval...")

    retrieved_results = retrieve_top_k(
        query_embeddings=query_embeddings,
        document_embeddings=document_embeddings,
        document_ids=document_ids,
        k=max(K_VALUES),
    )

    # ---------------------------------------------------------
    # 6. Evaluate retrieval metrics
    # ---------------------------------------------------------

    print("\n[6/6] Evaluating retrieval metrics...")

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

    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value:.6f}")

    # ---------------------------------------------------------
    # 7. Save experiment report
    # ---------------------------------------------------------

    report = {
        "experiment": "SciFact Experiment A - Document-level Retrieval",
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
        "document_lengths": {
            "mean_tokens": float(token_counts.mean()),
            "median_tokens": float(np.median(token_counts)),
            "maximum_tokens": int(token_counts.max()),
            "documents_exceeding_limit": truncated_documents,
            "truncation_percentage": float(
                truncated_documents / len(chunks) * 100
            ),
        },
        "retrieval": {
            "method": "Exact cosine similarity using NumPy",
            "database": None,
            "chunking": False,
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
        reports_dir / f"scifact_experiment_a_{timestamp}.json"
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print("\n========== EXPERIMENT COMPLETED ==========")
    print(f"Report saved to: {report_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
