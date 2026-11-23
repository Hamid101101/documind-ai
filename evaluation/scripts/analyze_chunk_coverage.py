"""Analyze how QASPER evidence is distributed across stored chunks."""

import json
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]

GROUND_TRUTH_PATH = (
    ROOT
    / "evaluation/datasets/processed/ground_truth/retrieval_ground_truth.jsonl"
)


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def summarize(values: list[float]) -> dict:
    array = np.asarray(values, dtype=float)

    return {
        "count": int(len(array)),
        "mean": round(float(np.mean(array)), 4),
        "median": round(float(np.median(array)), 4),
        "p25": round(float(np.percentile(array, 25)), 4),
        "p75": round(float(np.percentile(array, 75)), 4),
        "min": round(float(np.min(array)), 4),
        "max": round(float(np.max(array)), 4),
    }


def main() -> None:
    records = load_jsonl(GROUND_TRUTH_PATH)

    best_coverages = []
    relevant_chunk_counts = []
    evidence_token_counts = []

    coverage_distribution = Counter()
    chunk_count_distribution = Counter()

    for record in records:
        coverage_values = [
            float(value)
            for value in record["chunk_coverage"].values()
        ]

        if not coverage_values:
            continue

        best_coverage = max(coverage_values)
        relevant_count = len(record["relevant_chunk_ids"])

        best_coverages.append(best_coverage)
        relevant_chunk_counts.append(relevant_count)
        evidence_token_counts.append(
            int(record["evidence_token_count"])
        )

        if best_coverage < 0.25:
            coverage_distribution["<25%"] += 1
        elif best_coverage < 0.50:
            coverage_distribution["25-50%"] += 1
        elif best_coverage < 0.75:
            coverage_distribution["50-75%"] += 1
        elif best_coverage < 0.90:
            coverage_distribution["75-90%"] += 1
        else:
            coverage_distribution["90-100%"] += 1

        chunk_count_distribution[relevant_count] += 1

    total = len(best_coverages)

    print("=" * 65)
    print("QASPER CHUNK COVERAGE ANALYSIS")
    print("=" * 65)

    print(f"\nQuestions analyzed: {total}")

    print("\nBEST SINGLE-CHUNK EVIDENCE COVERAGE")
    for key, value in summarize(best_coverages).items():
        print(f"  {key}: {value}")

    print("\nCOVERAGE DISTRIBUTION")
    for category in [
        "<25%",
        "25-50%",
        "50-75%",
        "75-90%",
        "90-100%",
    ]:
        count = coverage_distribution[category]
        percentage = 100 * count / total
        print(
            f"  {category:10} {count:4} "
            f"({percentage:.2f}%)"
        )

    print("\nQUESTIONS BY NUMBER OF RELEVANT CHUNKS")
    for count, frequency in sorted(chunk_count_distribution.items()):
        print(f"  {count:2} chunks: {frequency:4} questions")

    print("\nRELEVANT CHUNK COUNT STATISTICS")
    for key, value in summarize(relevant_chunk_counts).items():
        print(f"  {key}: {value}")

    print("\nEVIDENCE TOKEN COUNT STATISTICS")
    for key, value in summarize(evidence_token_counts).items():
        print(f"  {key}: {value}")

    print("\nCOVERAGE THRESHOLDS")
    for threshold in [0.50, 0.80, 0.90]:
        count = sum(value >= threshold for value in best_coverages)
        print(
            f"  At least {threshold:.0%}: "
            f"{count}/{total} ({100 * count / total:.2f}%)"
        )


if __name__ == "__main__":
    main()
