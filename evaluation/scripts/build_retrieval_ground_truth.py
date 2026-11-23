
"""Build QASPER retrieval ground truth from source evidence and stored chunks."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path
from uuid import UUID
from typing import Any

from database.repositories.chunk_repository import ChunkRepository


ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "evaluation" / "datasets" / "processed"
EVIDENCE_PATH = DATA_DIR / "ground_truth" / "evidence.jsonl"
MAPPING_PATH = DATA_DIR / "qasper_document_mapping.json"
DOCUMENTS_DIR = DATA_DIR / "documents"
OUTPUT_PATH = DATA_DIR / "ground_truth" / "retrieval_ground_truth.jsonl"
SUMMARY_PATH = DATA_DIR / "ground_truth" / "retrieval_ground_truth_summary.json"


def normalize_text(text: str) -> str:
    """Normalize text while preserving punctuation and character offsets."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """Tokenize normalized text for the evidence token count."""
    return re.findall(r"\w+|[^\w\s]", normalize_text(text))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def get_evidence_text(record: dict[str, Any]) -> str:
    """Read the evidence text from supported dataset field names."""
    for key in ("evidence_text", "evidence", "text", "content"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value

    raise ValueError(
        f"Evidence text missing for question {record.get('question_id')}"
    )


def merge_intervals(
    intervals: list[tuple[int, int]],
) -> list[tuple[int, int]]:
    """Merge overlapping or adjacent character intervals."""
    if not intervals:
        return []

    intervals = sorted(intervals)
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        previous_start, previous_end = merged[-1]

        if start <= previous_end:
            merged[-1] = (
                previous_start,
                max(previous_end, end),
            )
        else:
            merged.append((start, end))

    return merged


def interval_length(intervals: list[tuple[int, int]]) -> int:
    """Return the total length of a union of intervals."""
    return sum(end - start for start, end in merge_intervals(intervals))


def align_chunks(
    source_text: str,
    chunks: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[int]]:
    """
    Align stored chunks with the normalized source document.

    Chunks are processed in chunk_index order. If sequential matching
    fails, a global fallback searches from the beginning of the source.
    """
    source = normalize_text(source_text)
    aligned = []
    unaligned = []
    previous_start = -1

    for chunk in sorted(chunks, key=lambda item: item["chunk_index"]):
        content = normalize_text(chunk["content"])

        if not content:
            unaligned.append(chunk["chunk_index"])
            continue

        start = source.find(content, previous_start + 1)

        if start == -1:
            start = source.find(content)

        if start == -1:
            unaligned.append(chunk["chunk_index"])
            continue

        aligned.append({
            "chunk_index": chunk["chunk_index"],
            "start": start,
            "end": start + len(content),
        })

        previous_start = start

    return aligned, unaligned


def find_evidence_interval(
    source_text: str,
    evidence_text: str,
) -> tuple[int, int] | None:
    """Find the first exact occurrence of normalized evidence in source."""
    source = normalize_text(source_text)
    evidence = normalize_text(evidence_text)

    if not evidence:
        return None

    start = source.find(evidence)

    if start == -1:
        return None

    return start, start + len(evidence)


def count_evidence_tokens(
    source_text: str,
    evidence_intervals: list[tuple[int, int]],
) -> int:
    """Count unique source tokens intersecting the evidence intervals."""
    normalized_source = normalize_text(source_text)
    merged_evidence = merge_intervals(evidence_intervals)

    count = 0

    for match in re.finditer(r"\w+|[^\w\s]", normalized_source):
        token_start, token_end = match.span()

        if any(
            token_start < evidence_end and token_end > evidence_start
            for evidence_start, evidence_end in merged_evidence
        ):
            count += 1

    return count


def calculate_chunk_coverage(
    chunk_interval: tuple[int, int],
    evidence_intervals: list[tuple[int, int]],
    total_evidence_length: int,
) -> float:
    """Calculate the fraction of all evidence characters covered by a chunk."""
    if total_evidence_length <= 0:
        return 0.0

    chunk_start, chunk_end = chunk_interval

    intersections = []

    for evidence_start, evidence_end in evidence_intervals:
        start = max(chunk_start, evidence_start)
        end = min(chunk_end, evidence_end)

        if start < end:
            intersections.append((start, end))

    covered_length = interval_length(intersections)

    return covered_length / total_evidence_length


def main() -> None:
    evidence_records = load_jsonl(EVIDENCE_PATH)
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))

    # Support both direct mappings and mappings nested under "documents".
    if "documents" in mapping:
        mapping = mapping["documents"]

    # Only exact paragraph evidence belongs to this benchmark.
    evidence_records = [
        item
        for item in evidence_records
        if item.get("match_type") == "exact_paragraph"
    ]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)

    for item in evidence_records:
        key = (
            str(item["paper_id"]),
            str(item["question_id"]),
        )
        grouped[key].append(item)

    repository = ChunkRepository()

    output_records = []
    unmatched = []
    alignment_failures = 0

    # Cache source text, aligned chunks, and database IDs per document.
    document_cache: dict[str, dict[str, Any] | None] = {}

    for (paper_id, question_id), annotations in grouped.items():
        mapping_entry = mapping.get(paper_id)

        if isinstance(mapping_entry, str):
            document_id_string = mapping_entry
            mapping_status = "completed"
        elif isinstance(mapping_entry, dict):
            document_id_string = mapping_entry.get("document_id")
            mapping_status = mapping_entry.get("status")
        else:
            document_id_string = None
            mapping_status = None

        if not document_id_string or mapping_status != "completed":
            unmatched.append({
                "paper_id": paper_id,
                "question_id": question_id,
                "reason": "document_mapping_missing_or_incomplete",
            })
            continue

        document_id = UUID(str(document_id_string))

        if paper_id not in document_cache:
            document_path = DOCUMENTS_DIR / f"{paper_id}.txt"

            if not document_path.is_file():
                document_cache[paper_id] = None
            else:
                source_text = document_path.read_text(encoding="utf-8")
                stored_chunks = repository.get_by_document_id(document_id)

                aligned_chunks, unaligned_indices = align_chunks(
                    source_text,
                    stored_chunks,
                )

                # Preserve the mapping from chunk_index to the real DB ID.
                chunk_ids_by_index = {
                    chunk["chunk_index"]: chunk["id"]
                    for chunk in stored_chunks
                }

                document_cache[paper_id] = {
                    "source_text": source_text,
                    "aligned_chunks": aligned_chunks,
                    "chunk_ids_by_index": chunk_ids_by_index,
                    "unaligned_chunk_indices": unaligned_indices,
                }

        document_data = document_cache[paper_id]

        if document_data is None:
            unmatched.append({
                "paper_id": paper_id,
                "question_id": question_id,
                "reason": "source_document_missing",
            })
            continue

        source_text = document_data["source_text"]
        aligned_chunks = document_data["aligned_chunks"]
        chunk_ids_by_index = document_data["chunk_ids_by_index"]

        # Locate every evidence annotation in the normalized source.
        evidence_intervals = []
        evidence_alignment_failed = False

        for annotation in annotations:
            evidence_text = get_evidence_text(annotation)

            interval = find_evidence_interval(
                source_text,
                evidence_text,
            )

            if interval is None:
                evidence_alignment_failed = True
                break

            evidence_intervals.append(interval)

        if evidence_alignment_failed or not evidence_intervals:
            alignment_failures += 1
            unmatched.append({
                "paper_id": paper_id,
                "question_id": question_id,
                "reason": "evidence_alignment_failed",
            })
            continue

        evidence_intervals = merge_intervals(evidence_intervals)
        total_evidence_length = interval_length(evidence_intervals)

        if total_evidence_length == 0:
            alignment_failures += 1
            unmatched.append({
                "paper_id": paper_id,
                "question_id": question_id,
                "reason": "evidence_alignment_failed",
            })
            continue

        relevant_chunk_ids = []
        chunk_coverage = {}

        for chunk in aligned_chunks:
            chunk_index = chunk["chunk_index"]
            chunk_interval = (chunk["start"], chunk["end"])

            coverage = calculate_chunk_coverage(
                chunk_interval,
                evidence_intervals,
                total_evidence_length,
            )

            # Preserve the original policy: any positive overlap is relevant.
            if coverage > 0:
                chunk_id = chunk_ids_by_index[chunk_index]

                relevant_chunk_ids.append(chunk_id)
                chunk_coverage[str(chunk_id)] = round(coverage, 6)

        if not relevant_chunk_ids:
            unmatched.append({
                "paper_id": paper_id,
                "question_id": question_id,
                "reason": "no_relevant_chunks",
            })
            continue

        evidence_token_count = count_evidence_tokens(
            source_text,
            evidence_intervals,
        )

        output_records.append({
            "paper_id": paper_id,
            "document_id": str(document_id),
            "question_id": question_id,
            "evidence_count": len(annotations),
            "evidence_token_count": evidence_token_count,
            "relevant_chunk_ids": relevant_chunk_ids,
            "chunk_coverage": chunk_coverage,
        })

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        for record in output_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "input_questions": len(grouped),
        "ground_truth_questions": len(output_records),
        "unmatched_questions": len(unmatched),
        "alignment_failures": alignment_failures,
        "unmatched": unmatched,
        "output": str(OUTPUT_PATH),
    }

    SUMMARY_PATH.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps({
        key: value
        for key, value in summary.items()
        if key != "unmatched"
    }, indent=2))


if __name__ == "__main__":
    main()
