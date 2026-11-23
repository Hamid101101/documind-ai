"""Diagnose dense-retrieval similarity scores against QASPER ground truth."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from retrieval.vector_search import VectorSearch


ROOT = Path(__file__).resolve().parents[2]

QUESTIONS_PATH = (
    ROOT / "evaluation/datasets/processed/ground_truth/questions.jsonl"
)
GROUND_TRUTH_PATH = (
    ROOT
    / "evaluation/datasets/processed/ground_truth/retrieval_ground_truth.jsonl"
)
REPORT_DIR = ROOT / "evaluation/reports"

TOP_K = 10


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def summarize_scores(scores: list[float]) -> dict[str, Any]:
    if not scores:
        return {"count": 0}

    values = np.asarray(scores, dtype=float)

    return {
        "count": int(len(values)),
        "min": round(float(np.min(values)), 6),
        "p25": round(float(np.percentile(values, 25)), 6),
        "median": round(float(np.median(values)), 6),
        "mean": round(float(np.mean(values)), 6),
        "p75": round(float(np.percentile(values, 75)), 6),
        "max": round(float(np.max(values)), 6),
    }


def main() -> None:
    questions = load_jsonl(QUESTIONS_PATH)
    ground_truth = load_jsonl(GROUND_TRUTH_PATH)

    questions_by_id = {
        str(record["question_id"]): record
        for record in questions
    }

    eligible_records = []

    for record in ground_truth:
        question_id = str(record["question_id"])

        if question_id not in questions_by_id:
            raise ValueError(
                f"Question missing from questions dataset: {question_id}"
            )

        question_record = questions_by_id[question_id]
        question_text = (
            question_record.get("question")
            or question_record.get("question_text")
            or question_record.get("query")
        )

        if not question_text:
            raise ValueError(
                f"Question text missing for question: {question_id}"
            )

        eligible_records.append({
            **record,
            "question": question_text,
        })

    print(f"Loaded {len(questions)} questions.")
    print(f"Eligible ground-truth questions: {len(eligible_records)}")

    search = VectorSearch()
    repository = search.chunk_repository

    if search.config.similarity_threshold is not None:
        raise ValueError(
            "Expected similarity_threshold=None for baseline diagnostics."
        )

    retrieved_relevant_scores = []
    retrieved_irrelevant_scores = []
    missed_relevant_scores = []

    per_question_results = []

    for index, record in enumerate(eligible_records, start=1):
        question_id = str(record["question_id"])
        question = record["question"]

        relevant_ids = {
            int(chunk_id)
            for chunk_id in record["relevant_chunk_ids"]
        }

        # Generate the query embedding once.
        query_embedding = search.embedding_generator.embed_query(question)

        # Reproduce the global dense top-10 baseline retrieval.
        retrieved = repository.search_similar(
            query_embedding=query_embedding,
            top_k=TOP_K,
            filters=None,
        )

        retrieved_ids = {
            int(result["id"])
            for result in retrieved
        }

        retrieved_relevant = [
            result
            for result in retrieved
            if int(result["id"]) in relevant_ids
        ]

        retrieved_irrelevant = [
            result
            for result in retrieved
            if int(result["id"]) not in relevant_ids
        ]

        missed_ids = relevant_ids - retrieved_ids

        retrieved_relevant_scores.extend(
            float(result["similarity"])
            for result in retrieved_relevant
        )

        retrieved_irrelevant_scores.extend(
            float(result["similarity"])
            for result in retrieved_irrelevant
        )

        # Calculate scores for relevant chunks that did not reach top 10.
        missed_scores = {}

        if missed_ids:
            vector = repository._serialize_embedding(query_embedding)

            sql = """
                SELECT
                    id,
                    1 - (embedding <=> %s::vector) AS similarity
                FROM document_chunks
                WHERE id = ANY(%s)
                  AND embedding IS NOT NULL;
            """

            with repository.db.connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        sql,
                        (vector, list(missed_ids)),
                    )
                    rows = cursor.fetchall()

            missed_scores = {
                int(row["id"]): float(row["similarity"])
                for row in rows
            }

            if set(missed_scores) != missed_ids:
                missing = missed_ids - set(missed_scores)
                raise ValueError(
                    f"Could not score relevant chunks: {sorted(missing)}"
                )

            missed_relevant_scores.extend(missed_scores.values())

        top10_scores = [
            float(result["similarity"])
            for result in retrieved
        ]

        retrieved_irrelevant_values = [
            float(result["similarity"])
            for result in retrieved_irrelevant
        ]

        per_question_results.append({
            "question_id": question_id,
            "paper_id": record["paper_id"],
            "question": question,
            "relevant_chunk_count": len(relevant_ids),
            "retrieved_relevant_count": len(retrieved_relevant),
            "retrieved_irrelevant_count": len(retrieved_irrelevant),
            "missed_relevant_count": len(missed_ids),
            "best_missed_relevant_score": (
                max(missed_scores.values())
                if missed_scores else None
            ),
            "lowest_top10_score": (
                min(top10_scores) if top10_scores else None
            ),
            "highest_retrieved_irrelevant_score": (
                max(retrieved_irrelevant_values)
                if retrieved_irrelevant_values else None
            ),
        })

        if index % 25 == 0 or index == len(eligible_records):
            print(
                f"Processed {index}/{len(eligible_records)} questions."
            )

    report = {
        "experiment": "dense_retrieval_similarity_diagnostics",
        "dataset": "QASPER",
        "retrieval_method": "dense",
        "corpus_scope": "global",
        "top_k": TOP_K,
        "total_questions": len(eligible_records),
        "score_groups": {
            "retrieved_relevant": summarize_scores(
                retrieved_relevant_scores
            ),
            "retrieved_irrelevant": summarize_scores(
                retrieved_irrelevant_scores
            ),
            "missed_relevant": summarize_scores(
                missed_relevant_scores
            ),
        },
        "per_question_results": per_question_results,
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "embedding_model": search.embedding_generator.config.model_name,
            "embedding_dimension": (
                search.embedding_generator.config.expected_dimension
            ),
            "similarity_metric": "1 - cosine_distance",
            "similarity_threshold": None,
        },
    }

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = (
        REPORT_DIR / f"similarity_diagnostics_{timestamp}.json"
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, ensure_ascii=False)
        file.write("\n")

    print("\n" + "=" * 65)
    print("SIMILARITY SCORE DIAGNOSTICS")
    print("=" * 65)

    for group_name, scores in report["score_groups"].items():
        print(f"\n{group_name.upper().replace('_', ' ')}")
        for metric, value in scores.items():
            print(f"  {metric}: {value}")

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()
