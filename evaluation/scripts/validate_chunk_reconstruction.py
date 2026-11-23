
"""Validate QASPER chunk reproducibility and overlap reconstruction."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

from database.repositories.chunk_repository import ChunkRepository
from ingestion.chunking.text_chunker import TextChunker
from ingestion.cleaning.text_cleaner import TextCleaner
from ingestion.loaders.txt_loader import TXTLoader


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "evaluation" / "datasets" / "processed"
DOCUMENTS_DIR = PROCESSED_DIR / "documents"
MAPPING_PATH = PROCESSED_DIR / "qasper_document_mapping.json"
REPORT_PATH = PROCESSED_DIR / "chunk_reconstruction_validation.json"

CHUNK_OVERLAP = 100


def load_mapping() -> dict[str, dict[str, Any]]:
    with MAPPING_PATH.open("r", encoding="utf-8") as file:
        mapping = json.load(file)

    if not isinstance(mapping, dict):
        raise ValueError("Document mapping must be a JSON object.")

    return mapping


def find_overlap(
    previous: list[int],
    current: list[int],
    max_overlap: int = CHUNK_OVERLAP,
) -> int:
    """Find the longest exact suffix-prefix token overlap."""
    limit = min(max_overlap, len(previous), len(current))

    for size in range(limit, 0, -1):
        if previous[-size:] == current[:size]:
            return size

    return 0


def reconstruct_tokens(
    chunk_tokens: list[list[int]],
) -> tuple[list[int], list[int]]:
    """Reconstruct a token sequence by removing consecutive overlaps."""
    if not chunk_tokens:
        return [], []

    reconstructed = list(chunk_tokens[0])
    overlaps = []

    for previous, current in zip(
        chunk_tokens,
        chunk_tokens[1:],
    ):
        overlap = find_overlap(previous, current)
        overlaps.append(overlap)
        reconstructed.extend(current[overlap:])

    return reconstructed, overlaps


def first_difference(
    expected: list[int],
    actual: list[int],
) -> int | None:
    """Return the first differing token position."""
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return index

    if len(expected) != len(actual):
        return min(len(expected), len(actual))

    return None


def validate_document(
    paper_id: str,
    document_id: str,
    loader: TXTLoader,
    cleaner: TextCleaner,
    chunker: TextChunker,
    repository: ChunkRepository,
) -> dict[str, Any]:
    """Validate one QASPER document against its stored chunks."""
    source_path = DOCUMENTS_DIR / f"{paper_id}.txt"

    if not source_path.is_file():
        raise FileNotFoundError(
            f"Prepared document not found: {source_path}"
        )

    # Reproduce the same load -> clean -> chunk sequence as ingestion.
    loaded_document = loader.load(str(source_path))
    cleaned_document = cleaner.clean_document(loaded_document)

    regenerated_chunks = chunker.chunk_document(cleaned_document)

    stored_chunks = repository.get_by_document_id(
        document_id
    )

    stored_chunks.sort(key=lambda chunk: chunk["chunk_index"])

    # Verify stored chunk indices are consecutive.
    indices = [
        chunk["chunk_index"]
        for chunk in stored_chunks
    ]

    consecutive_indices = (
        indices == list(range(len(indices)))
    )

    # Compare regenerated and stored chunks one by one.
    regenerated_contents = [
        chunk.content for chunk in regenerated_chunks
    ]

    stored_contents = [
        chunk["content"] for chunk in stored_chunks
    ]

    common_count = min(
        len(regenerated_chunks),
        len(stored_chunks),
    )

    content_mismatches = []

    for index in range(common_count):
        if regenerated_contents[index] != stored_contents[index]:
            content_mismatches.append(index)

    regenerated_token_counts = [
        chunk.token_count for chunk in regenerated_chunks
    ]

    stored_token_counts = [
        chunk["token_count"] for chunk in stored_chunks
    ]

    token_count_mismatches = []

    for index in range(common_count):
        if (
            regenerated_token_counts[index]
            != stored_token_counts[index]
        ):
            token_count_mismatches.append(index)

    # Tokenize the stored chunks using the same E5 tokenizer.
    stored_chunk_tokens = [
        chunker._tokenize(chunk["content"])
        for chunk in stored_chunks
    ]

    # Check stored token counts against actual re-tokenization.
    actual_token_counts = [
        len(tokens)
        for tokens in stored_chunk_tokens
    ]

    database_count_mismatches = [
        index
        for index, (stored, actual) in enumerate(
            zip(stored_token_counts, actual_token_counts)
        )
        if stored != actual
    ]

    # Reconstruct the document by removing exact overlaps.
    reconstructed, overlaps = reconstruct_tokens(
        stored_chunk_tokens
    )

    # Compare against the cleaned source text.
    source_text = cleaned_document.pages[0].text
    source_tokens = chunker._tokenize(source_text)

    difference_position = first_difference(
        source_tokens,
        reconstructed,
    )

    exact_token_reconstruction = (
        source_tokens == reconstructed
    )

    overlap_mismatches = [
        index
        for index, overlap in enumerate(overlaps, start=1)
        if overlap != CHUNK_OVERLAP
    ]

    return {
        "paper_id": paper_id,
        "document_id": document_id,
        "stored_chunk_count": len(stored_chunks),
        "regenerated_chunk_count": len(regenerated_chunks),
        "chunk_count_matches": (
            len(stored_chunks) == len(regenerated_chunks)
        ),
        "chunk_indices_consecutive": consecutive_indices,
        "regenerated_content_mismatches": content_mismatches,
        "regenerated_token_count_mismatches": (
            token_count_mismatches
        ),
        "database_token_count_mismatches": (
            database_count_mismatches
        ),
        "overlap_count": len(overlaps),
        "overlap_mismatches": overlap_mismatches,
        "overlap_values": overlaps,
        "source_token_count": len(source_tokens),
        "reconstructed_token_count": len(reconstructed),
        "exact_token_reconstruction": exact_token_reconstruction,
        "first_token_difference": difference_position,
    }


def evaluate() -> None:
    logging.basicConfig(level=logging.WARNING)

    mapping = load_mapping()

    loader = TXTLoader()
    cleaner = TextCleaner()
    chunker = TextChunker()
    repository = ChunkRepository()

    results = []
    failures = []

    print("Starting QASPER chunk reconstruction validation...")
    print(f"Documents in mapping: {len(mapping)}")

    for paper_id, mapping_record in sorted(mapping.items()):
        document_id = str(mapping_record["document_id"])

        try:
            result = validate_document(
                paper_id=paper_id,
                document_id=document_id,
                loader=loader,
                cleaner=cleaner,
                chunker=chunker,
                repository=repository,
            )

            results.append(result)

            print(
                f"[{len(results):03d}] {paper_id}: "
                f"{result['stored_chunk_count']} chunks, "
                f"reproducible="
                f"{result['chunk_count_matches'] and not result['regenerated_content_mismatches']}, "
                f"reconstructed="
                f"{result['exact_token_reconstruction']}"
            )

        except Exception as exc:
            logging.exception(
                "Validation failed for paper %s",
                paper_id,
            )

            failures.append({
                "paper_id": paper_id,
                "document_id": document_id,
                "error": str(exc),
            })

    total_documents = len(results)

    reproducible_documents = sum(
        result["chunk_count_matches"]
        and not result["regenerated_content_mismatches"]
        for result in results
    )

    exact_reconstructions = sum(
        result["exact_token_reconstruction"]
        for result in results
    )

    total_stored_chunks = sum(
        result["stored_chunk_count"]
        for result in results
    )

    total_content_mismatches = sum(
        len(result["regenerated_content_mismatches"])
        for result in results
    )

    total_database_count_mismatches = sum(
        len(result["database_token_count_mismatches"])
        for result in results
    )

    total_overlap_mismatches = sum(
        len(result["overlap_mismatches"])
        for result in results
    )

    report = {
        "configuration": {
            "embedding_model": chunker.tokenizer.name_or_path,
            "chunk_size": chunker.config.chunk_size,
            "chunk_overlap": chunker.config.chunk_overlap,
        },
        "summary": {
            "documents_validated": total_documents,
            "validation_failures": len(failures),
            "total_stored_chunks": total_stored_chunks,
            "reproducible_documents": reproducible_documents,
            "exact_token_reconstructions": exact_reconstructions,
            "regenerated_content_mismatches": (
                total_content_mismatches
            ),
            "database_token_count_mismatches": (
                total_database_count_mismatches
            ),
            "overlap_mismatches": total_overlap_mismatches,
        },
        "documents": results,
        "failures": failures,
    }

    with REPORT_PATH.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print("\n" + "=" * 65)
    print("VALIDATION SUMMARY")
    print("=" * 65)
    print(f"Documents validated: {total_documents}")
    print(f"Validation failures: {len(failures)}")
    print(f"Stored chunks: {total_stored_chunks}")
    print(f"Reproducible documents: {reproducible_documents}")
    print(f"Exact token reconstructions: {exact_reconstructions}")
    print(f"Content mismatches: {total_content_mismatches}")
    print(
        f"Database token-count mismatches: "
        f"{total_database_count_mismatches}"
    )
    print(f"Overlap mismatches: {total_overlap_mismatches}")
    print(f"\nReport saved to: {REPORT_PATH}")


if __name__ == "__main__":
    try:
        evaluate()
    except Exception as exc:
        print(f"Validation failed: {exc}", file=sys.stderr)
        raise
