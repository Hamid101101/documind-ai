
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from evaluation.metrics.retrieval_metrics import (
    aggregate_metrics,
    evaluate_query,
)
from ingestion.embeddings.config import EmbeddingConfig
from ingestion.embeddings.embedding_generator import EmbeddingGenerator


EXPECTED_MODEL = "intfloat/multilingual-e5-base"
K_VALUES = (1, 3, 5, 10)

DATA_DIR = PROJECT_ROOT / "evaluation/datasets/processed"
QUESTIONS_PATH = DATA_DIR / "ground_truth/questions.jsonl"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth/retrieval_ground_truth.jsonl"

EMBEDDINGS_PATH = (
    PROJECT_ROOT / "evaluation/reports/qasper_document_embeddings.npz"
)


def load_jsonl(path: Path) -> list[dict]:
    records = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                records.append(json.loads(line))

    return records


def encode_queries(embedder, questions):
    """Encode QASPER questions using the E5 query prefix."""

    config = embedder.config

    query_ids = [item["question_id"] for item in questions]
    query_texts = [
        f"{config.query_prefix}{item['question'].strip()}"
        for item in questions
    ]

    embeddings = embedder.model.encode(
        query_texts,
        batch_size=config.batch_size,
        convert_to_numpy=True,
        normalize_embeddings=config.normalize_embeddings,
        show_progress_bar=True,
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    expected_shape = (
        len(questions),
        config.expected_dimension,
    )

    if embeddings.shape != expected_shape:
        raise ValueError(
            f"Unexpected query embedding shape: {embeddings.shape}; "
            f"expected {expected_shape}"
        )

    if not np.isfinite(embeddings).all():
        raise ValueError("Query embeddings contain NaN or infinity.")

    return query_ids, embeddings


def main():
    print("\n========== QASPER DOCUMENT-LEVEL EVALUATION ==========\n")

    # 1. Load question and ground-truth datasets.
    print("[1/5] Loading evaluation datasets...")

    questions = load_jsonl(QUESTIONS_PATH)
    ground_truth = load_jsonl(GROUND_TRUTH_PATH)

    questions_by_id = {
        item["question_id"]: item
        for item in questions
    }

    if len(questions_by_id) != len(questions):
        raise ValueError("Duplicate question IDs detected.")

    ground_truth_by_id = {
        item["question_id"]: item
        for item in ground_truth
    }

    if len(ground_truth_by_id) != len(ground_truth):
        raise ValueError("Duplicate ground-truth question IDs detected.")

    missing_questions = (
        set(ground_truth_by_id) - set(questions_by_id)
    )

    if missing_questions:
        raise ValueError(
            f"Missing questions for {len(missing_questions)} "
            "ground-truth records."
        )

    eligible_questions = [
        questions_by_id[question_id]
        for question_id in ground_truth_by_id
    ]

    print(f"Available questions: {len(questions)}")
    print(f"Eligible questions: {len(eligible_questions)}")

    # 2. Load previously generated paper embeddings.
    print("\n[2/5] Loading paper embeddings...")

    data = np.load(EMBEDDINGS_PATH, allow_pickle=False)

    paper_ids = [
        str(paper_id)
        for paper_id in data["paper_ids"].tolist()
    ]

    paper_embeddings = np.asarray(
        data["embeddings"],
        dtype=np.float32,
    )

    if len(paper_ids) != len(set(paper_ids)):
        raise ValueError("Duplicate paper IDs in embedding file.")

    if paper_embeddings.shape != (len(paper_ids), 768):
        raise ValueError(
            f"Unexpected paper embedding shape: "
            f"{paper_embeddings.shape}"
        )

    if not np.isfinite(paper_embeddings).all():
        raise ValueError("Paper embeddings contain NaN or infinity.")

    if not np.allclose(
        np.linalg.norm(paper_embeddings, axis=1),
        1.0,
        atol=1e-3,
    ):
        raise ValueError("Paper embeddings are not normalized.")

    paper_id_set = set(paper_ids)

    for record in ground_truth:
        if str(record["paper_id"]) not in paper_id_set:
            raise ValueError(
                f"Paper {record['paper_id']} has no embedding."
            )

    print(f"Paper embeddings: {paper_embeddings.shape}")

    # 3. Initialize the query embedding model.
    print("\n[3/5] Initializing embedding model...")

    config = EmbeddingConfig()

    if config.model_name != EXPECTED_MODEL:
        raise ValueError(
            f"Expected {EXPECTED_MODEL}, got {config.model_name}."
        )

    embedder = EmbeddingGenerator(config=config)

    query_ids, query_embeddings = encode_queries(
        embedder,
        eligible_questions,
    )

    if not np.allclose(
        np.linalg.norm(query_embeddings, axis=1),
        1.0,
        atol=1e-3,
    ):
        raise ValueError("Query embeddings are not normalized.")

    print(f"Query embeddings: {query_embeddings.shape}")

    # 4. Exact cosine-similarity retrieval.
    print("\n[4/5] Running exact paper-level retrieval...")

    # Both matrices are normalized, so dot product = cosine similarity.
    similarity_matrix = query_embeddings @ paper_embeddings.T

    query_results = []
    per_query_results = []

    for index, question_id in enumerate(query_ids):

        scores = similarity_matrix[index]

        ranked_indices = np.argsort(-scores, kind="stable")

        retrieved_ids = [
            paper_ids[i]
            for i in ranked_indices
        ]

        relevant_paper_id = str(
            ground_truth_by_id[question_id]["paper_id"]
        )

        metrics = evaluate_query(
            retrieved_ids=retrieved_ids,
            relevant_ids={relevant_paper_id},
            k_values=K_VALUES,
        )

        query_results.append(metrics)

        target_rank = retrieved_ids.index(relevant_paper_id) + 1

        per_query_results.append({
            "question_id": question_id,
            "paper_id": relevant_paper_id,
            "target_rank": target_rank,
            "target_similarity": float(
                scores[paper_ids.index(relevant_paper_id)]
            ),
            "top_5": [
                {
                    "paper_id": paper_id,
                    "similarity": float(scores[paper_ids.index(paper_id)]),
                }
                for paper_id in retrieved_ids[:5]
            ],
        })

    metrics = aggregate_metrics(query_results)

    # 5. Display and save results.
    print("\n[5/5] Evaluation results...")

    print("\n========== PAPER-LEVEL RETRIEVAL METRICS ==========")

    for name, value in metrics.items():
        print(f"{name}: {value:.6f}")

    target_ranks = np.array(
        [item["target_rank"] for item in per_query_results]
    )

    print("\n========== TARGET PAPER RANKING ==========")
    print(f"Rank 1:  {np.sum(target_ranks == 1)}")
    print(f"Top 3:   {np.sum(target_ranks <= 3)}")
    print(f"Top 5:   {np.sum(target_ranks <= 5)}")
    print(f"Top 10:  {np.sum(target_ranks <= 10)}")
    print(f"Mean rank: {target_ranks.mean():.2f}")
    print(f"Median rank: {np.median(target_ranks):.2f}")

    report = {
        "experiment": "QASPER Paper-Level Retrieval",
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "dataset": {
            "name": "QASPER",
            "total_questions": len(questions),
            "evaluated_questions": len(eligible_questions),
            "papers": len(paper_ids),
        },
        "embedding": {
            "model": config.model_name,
            "dimension": config.expected_dimension,
            "normalized": config.normalize_embeddings,
            "query_prefix": config.query_prefix,
            "paper_representation": (
                "450-token windows, 100-token overlap, "
                "token-count-weighted mean, L2 normalization"
            ),
        },
        "retrieval": {
            "method": "Exact cosine similarity using NumPy",
            "candidate_level": "paper",
            "candidate_count": len(paper_ids),
            "k_values": list(K_VALUES),
        },
        "metrics": {
            name: float(value)
            for name, value in metrics.items()
        },
        "target_ranking": {
            "rank_1": int(np.sum(target_ranks == 1)),
            "top_3": int(np.sum(target_ranks <= 3)),
            "top_5": int(np.sum(target_ranks <= 5)),
            "top_10": int(np.sum(target_ranks <= 10)),
            "mean_rank": float(target_ranks.mean()),
            "median_rank": float(np.median(target_ranks)),
        },
        "per_query_results": per_query_results,
    }

    reports_dir = PROJECT_ROOT / "evaluation/reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    report_path = (
        reports_dir / f"qasper_document_retrieval_{timestamp}.json"
    )

    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=4, ensure_ascii=False)

    print(f"\nReport saved to: {report_path.relative_to(PROJECT_ROOT)}")
    print("\n========== EXPERIMENT COMPLETED ==========\n")


if __name__ == "__main__":
    main()
