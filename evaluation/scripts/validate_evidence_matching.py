
import json
import logging
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from uuid import UUID

from database.repositories.chunk_repository import ChunkRepository


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "evaluation" / "datasets" / "processed"
DOCUMENTS_DIR = PROCESSED_DIR / "documents"
EVIDENCE_PATH = PROCESSED_DIR / "ground_truth" / "evidence.jsonl"
MAPPING_PATH = PROCESSED_DIR / "qasper_document_mapping.json"
OUTPUT_PATH = PROCESSED_DIR / "evidence_matching_validation.json"

THRESHOLDS = (0.70, 0.80, 0.90, 1.00)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def normalize_text(text: str) -> str:
    """Normalize text consistently while preserving punctuation."""
    text = unicodedata.normalize("NFKC", text).casefold()
    return re.sub(r"\s+", " ", text).strip()


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def get_evidence_text(record: dict) -> str:
    """Support the evidence field names used by the prepared dataset."""
    for key in ("evidence_text", "evidence", "text", "content"):
        value = record.get(key)

        if isinstance(value, str) and value.strip():
            return value

    raise ValueError(
        f"Evidence text missing from record: {record.get('question_id')}"
    )


def align_chunks(
    source_text: str,
    chunks: list[dict],
) -> tuple[list[dict], list[dict]]:
    """
    Locate stored chunk contents inside the normalized source document.

    Search proceeds in chunk order. A global fallback handles cases where
    a chunk cannot be found after the previous chunk's starting position.
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

        end = start + len(content)

        aligned.append({
            "chunk_index": chunk["chunk_index"],
            "start": start,
            "end": end,
        })

        previous_start = start

    return aligned, unaligned


def calculate_interval_coverage(
    evidence_start: int,
    evidence_end: int,
    intervals: list[tuple[int, int]],
) -> float:
    """Calculate the fraction of evidence characters covered by intervals."""
    if evidence_end <= evidence_start:
        return 0.0

    clipped = []

    for start, end in intervals:
        overlap_start = max(start, evidence_start)
        overlap_end = min(end, evidence_end)

        if overlap_start < overlap_end:
            clipped.append((overlap_start, overlap_end))

    if not clipped:
        return 0.0

    clipped.sort()

    covered = 0
    current_start, current_end = clipped[0]

    for start, end in clipped[1:]:
        if start <= current_end:
            current_end = max(current_end, end)
        else:
            covered += current_end - current_start
            current_start, current_end = start, end

    covered += current_end - current_start

    return covered / (evidence_end - evidence_start)


def evaluate_evidence(
    evidence_text: str,
    source_text: str,
    aligned_chunks: list[dict],
) -> dict:
    """Measure exact source alignment and single/combined chunk coverage."""
    evidence = normalize_text(evidence_text)
    source = normalize_text(source_text)

    if not evidence:
        return {
            "source_match": False,
            "best_single_chunk_coverage": 0.0,
            "combined_chunk_coverage": 0.0,
            "matching_chunk_indices": [],
        }

    evidence_start = source.find(evidence)

    if evidence_start == -1:
        return {
            "source_match": False,
            "best_single_chunk_coverage": 0.0,
            "combined_chunk_coverage": 0.0,
            "matching_chunk_indices": [],
        }

    evidence_end = evidence_start + len(evidence)

    relevant = [
        chunk
        for chunk in aligned_chunks
        if chunk["start"] < evidence_end
        and chunk["end"] > evidence_start
    ]

    intervals = [
        (chunk["start"], chunk["end"])
        for chunk in relevant
    ]

    single_coverages = [
        calculate_interval_coverage(
            evidence_start,
            evidence_end,
            [interval],
        )
        for interval in intervals
    ]

    combined_coverage = calculate_interval_coverage(
        evidence_start,
        evidence_end,
        intervals,
    )

    return {
        "source_match": True,
        "best_single_chunk_coverage": max(single_coverages, default=0.0),
        "combined_chunk_coverage": combined_coverage,
        "matching_chunk_indices": [
            chunk["chunk_index"] for chunk in relevant
        ],
    }


def main() -> None:
    evidence_records = load_jsonl(EVIDENCE_PATH)
    mapping = load_json(MAPPING_PATH)

    # Support either a direct paper_id -> document_id mapping or
    # a mapping containing a "documents" collection.
    if "documents" in mapping:
        mapping = mapping["documents"]

    paper_to_document = {}

    if isinstance(mapping, dict):
        for paper_id, value in mapping.items():
            if isinstance(value, str):
                paper_to_document[paper_id] = value
            elif isinstance(value, dict):
                document_id = value.get("document_id")
                if document_id:
                    paper_to_document[paper_id] = document_id

    repository = ChunkRepository()
    chunks_by_document = {}
    alignments_by_document = {}
    source_texts = {}

    match_types = Counter()
    results = []
    alignment_failures = {}

    for record in evidence_records:
        match_type = record.get("match_type", "unknown")
        match_types[match_type] += 1

    eligible_records = [
        record
        for record in evidence_records
        if record.get("match_type") == "exact_paragraph"
    ]

    logger.info(
        "Validating %d exact-paragraph evidence records.",
        len(eligible_records),
    )

    for record in eligible_records:
        paper_id = str(record["paper_id"])

        document_id = paper_to_document.get(paper_id)

        if document_id is None:
            logger.warning("No document mapping for paper %s", paper_id)
            continue

        document_id = str(UUID(document_id))

        if document_id not in chunks_by_document:
            chunks = repository.get_by_document_id(UUID(document_id))
            chunks_by_document[document_id] = chunks

            document_path = DOCUMENTS_DIR / f"{paper_id}.txt"

            if not document_path.is_file():
                logger.warning("Missing source document: %s", document_path)
                source_texts[document_id] = None
                alignments_by_document[document_id] = []
                continue

            source_text = document_path.read_text(encoding="utf-8")
            source_texts[document_id] = source_text

            aligned, unaligned = align_chunks(source_text, chunks)
            alignments_by_document[document_id] = aligned

            if unaligned:
                alignment_failures[paper_id] = unaligned

        source_text = source_texts[document_id]

        if source_text is None:
            continue

        evaluation = evaluate_evidence(
            evidence_text=get_evidence_text(record),
            source_text=source_text,
            aligned_chunks=alignments_by_document[document_id],
        )

        results.append({
            "question_id": record.get("question_id"),
            "paper_id": paper_id,
            "source_match": evaluation["source_match"],
            "best_single_chunk_coverage": round(
                evaluation["best_single_chunk_coverage"], 4
            ),
            "combined_chunk_coverage": round(
                evaluation["combined_chunk_coverage"], 4
            ),
            "matching_chunk_indices": evaluation["matching_chunk_indices"],
        })

    total = len(results)
    source_matches = sum(item["source_match"] for item in results)

    def threshold_summary(metric: str) -> dict:
        values = [item[metric] for item in results]

        return {
            str(threshold): {
                "matches": sum(value >= threshold for value in values),
                "percentage": round(
                    100 * sum(value >= threshold for value in values) / total,
                    2,
                ) if total else 0.0,
            }
            for threshold in THRESHOLDS
        }

    report = {
        "summary": {
            "total_evidence_records": total,
            "original_document_matches": source_matches,
            "original_document_match_percentage": round(
                100 * source_matches / total, 2
            ) if total else 0.0,
            "best_single_chunk_coverage": threshold_summary(
                "best_single_chunk_coverage"
            ),
            "combined_chunk_coverage": threshold_summary(
                "combined_chunk_coverage"
            ),
            "match_types_in_dataset": dict(match_types),
            "documents_with_unaligned_chunks": len(alignment_failures),
            "unaligned_chunk_count": sum(
                len(indices) for indices in alignment_failures.values()
            ),
        },
        "alignment_failures": alignment_failures,
        "results": results,
    }

    OUTPUT_PATH.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    logger.info("Validation report saved to %s", OUTPUT_PATH)
    logger.info(
        "Source alignment: %d/%d (%.2f%%)",
        source_matches,
        total,
        report["summary"]["original_document_match_percentage"],
    )
    logger.info(
        "Combined coverage: %s",
        report["summary"]["combined_chunk_coverage"],
    )


if __name__ == "__main__":
    main()
