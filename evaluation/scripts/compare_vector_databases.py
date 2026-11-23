
import json
import logging
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient, models

# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

load_dotenv(PROJECT_ROOT / ".env")

from infra.postgres.postgres_client import PostgreSQLClient
from ingestion.embeddings.config import EmbeddingConfig
from ingestion.embeddings.embedding_generator import EmbeddingGenerator
from evaluation.metrics.retrieval_metrics import evaluate_query

GROUND_TRUTH_DIR = (
    PROJECT_ROOT / "evaluation" / "datasets"
    / "processed" / "ground_truth"
)

QUESTIONS_FILE = GROUND_TRUTH_DIR / "questions.jsonl"
GROUND_TRUTH_FILE = GROUND_TRUTH_DIR / "retrieval_ground_truth.jsonl"

REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "qasper_chunks"

EXPECTED_DIMENSION = 768
EXPECTED_CHUNKS = 1509
K_VALUES = (1, 3, 5, 10)
TOP_K = max(K_VALUES)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# Dataset loading and validation
# ---------------------------------------------------------

def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON at {path}:{line_number}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected JSON object at {path}:{line_number}"
                )

            records.append(record)

    if not records:
        raise ValueError(f"No records found in {path}")

    return records


def validate_datasets(
    questions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> list[dict[str, Any]]:

    question_ids = [item["question_id"] for item in questions]
    ground_truth_ids = [item["question_id"] for item in ground_truth]

    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Duplicate question IDs.")

    if len(ground_truth_ids) != len(set(ground_truth_ids)):
        raise ValueError("Duplicate ground-truth question IDs.")

    questions_by_id = {
        item["question_id"]: item
        for item in questions
    }

    if not set(ground_truth_ids).issubset(set(question_ids)):
        raise ValueError("Ground truth contains unknown question IDs.")

    for record in ground_truth:
        if not record.get("relevant_chunk_ids"):
            raise ValueError(
                f"Missing relevant chunks: {record['question_id']}"
            )

    eligible = [
        questions_by_id[question_id]
        for question_id in ground_truth_ids
    ]

    logger.info(
        "Eligible questions: %d; excluded questions: %d",
        len(eligible),
        len(questions) - len(eligible),
    )

    return eligible


# ---------------------------------------------------------
# PostgreSQL corpus loading
# ---------------------------------------------------------

def parse_vector(value: str) -> list[float]:
    if not value:
        raise ValueError("Empty embedding.")

    value = value.strip()

    if not value.startswith("[") or not value.endswith("]"):
        raise ValueError("Invalid PostgreSQL vector representation.")

    vector = [float(x) for x in value[1:-1].split(",")]

    if len(vector) != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected {EXPECTED_DIMENSION} dimensions, "
            f"received {len(vector)}."
        )

    if not all(math.isfinite(x) for x in vector):
        raise ValueError("Embedding contains non-finite values.")

    return vector


def load_corpus(
    db: PostgreSQLClient,
) -> tuple[np.ndarray, np.ndarray]:

    rows = db.fetch_all(
        """
        SELECT
            id,
            embedding::text AS embedding
        FROM document_chunks
        WHERE embedding IS NOT NULL
        ORDER BY id;
        """
    )

    if len(rows) != EXPECTED_CHUNKS:
        raise ValueError(
            f"Expected {EXPECTED_CHUNKS} chunks, got {len(rows)}."
        )

    chunk_ids = np.asarray(
        [int(row["id"]) for row in rows],
        dtype=np.int64,
    )

    vectors = np.asarray(
        [parse_vector(row["embedding"]) for row in rows],
        dtype=np.float32,
    ).astype(np.float64)

    if vectors.shape != (EXPECTED_CHUNKS, EXPECTED_DIMENSION):
        raise ValueError(f"Unexpected corpus shape: {vectors.shape}")

    norms = np.linalg.norm(vectors, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Some stored document embeddings are not normalized.")

    logger.info(
        "Loaded corpus: %d vectors, %d dimensions.",
        len(chunk_ids),
        vectors.shape[1],
    )

    return chunk_ids, vectors


# ---------------------------------------------------------
# PostgreSQL exact search
# ---------------------------------------------------------

def vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(str(float(x)) for x in vector) + "]"


POSTGRES_QUERY = """
    SELECT
        id,
        1 - (embedding <=> %s::vector) AS similarity
    FROM document_chunks
    WHERE embedding IS NOT NULL
    ORDER BY embedding <=> %s::vector ASC, id ASC
    LIMIT %s;
"""


def prepare_postgres_exact_search(cursor, query_vector: np.ndarray) -> str:
    """
    Discourage index-based plans so PostgreSQL performs an exact scan.
    Verify the actual execution plan before running the experiment.
    """

    cursor.execute("SET LOCAL enable_indexscan = off;")
    cursor.execute("SET LOCAL enable_bitmapscan = off;")
    cursor.execute("SET LOCAL enable_indexonlyscan = off;")
    cursor.execute("SET LOCAL enable_seqscan = on;")

    literal = vector_literal(query_vector)

    cursor.execute(
        """
        EXPLAIN (COSTS OFF)
        """ + POSTGRES_QUERY,
        (literal, literal, EXPECTED_CHUNKS),
    )

    plan_rows = cursor.fetchall()

    plan = "\n".join(
        str(row["QUERY PLAN"])
        for row in plan_rows
    )

    logger.info("PostgreSQL execution plan:\n%s", plan)

    if "Index Scan" in plan or "Bitmap" in plan:
        raise RuntimeError(
            "PostgreSQL is still using an index. "
            "Cannot claim this is an exact sequential scan."
        )

    if "Seq Scan" not in plan:
        raise RuntimeError(
            "Expected a sequential scan, but the plan does not show one."
        )

    return plan


def postgres_exact_search(
    cursor,
    query_vector: np.ndarray,
) -> tuple[list[int], dict[int, float]]:

    literal = vector_literal(query_vector)

    cursor.execute(
        POSTGRES_QUERY,
        (literal, literal, EXPECTED_CHUNKS),
    )

    rows = cursor.fetchall()

    if len(rows) != EXPECTED_CHUNKS:
        raise RuntimeError(
            f"PostgreSQL returned {len(rows)} results; "
            f"expected {EXPECTED_CHUNKS}."
        )

    ranked_ids = [int(row["id"]) for row in rows]

    scores = {
        int(row["id"]): float(row["similarity"])
        for row in rows
    }

    return ranked_ids, scores


# ---------------------------------------------------------
# Qdrant exact search
# ---------------------------------------------------------

def qdrant_exact_search(
    client: QdrantClient,
    query_vector: np.ndarray,
) -> tuple[list[int], dict[int, float]]:

    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector.astype(np.float32).tolist(),
        limit=EXPECTED_CHUNKS,
        search_params=models.SearchParams(exact=True),
        with_payload=False,
        with_vectors=False,
    )

    points = response.points

    if len(points) != EXPECTED_CHUNKS:
        raise RuntimeError(
            f"Qdrant returned {len(points)} results; "
            f"expected {EXPECTED_CHUNKS}."
        )

    ranked_ids = [int(point.id) for point in points]

    scores = {
        int(point.id): float(point.score)
        for point in points
    }

    return ranked_ids, scores


# ---------------------------------------------------------
# Independent NumPy cosine similarity
# ---------------------------------------------------------

def numpy_exact_search(
    chunk_ids: np.ndarray,
    document_vectors: np.ndarray,
    query_vector: np.ndarray,
) -> tuple[list[int], dict[int, float]]:

    query = query_vector.astype(np.float64)

    document_norms = np.linalg.norm(
        document_vectors,
        axis=1,
    )

    query_norm = np.linalg.norm(query)

    if query_norm == 0:
        raise ValueError("Query embedding has zero norm.")

    scores_array = (
        document_vectors @ query
    ) / (
        document_norms * query_norm
    )

    # Corpus IDs are already sorted ascending.
    # Stable sorting provides deterministic tie handling.
    order = np.argsort(-scores_array, kind="stable")

    ranked_ids = chunk_ids[order].astype(int).tolist()

    scores = {
        int(chunk_id): float(score)
        for chunk_id, score in zip(chunk_ids, scores_array)
    }

    return ranked_ids, scores


# ---------------------------------------------------------
# Metrics and ranking agreement
# ---------------------------------------------------------

def aggregate_metrics(
    query_metrics: list[dict[str, float]],
) -> dict[str, float]:

    if not query_metrics:
        raise ValueError("No metrics to aggregate.")

    keys = query_metrics[0].keys()

    return {
        key: sum(item[key] for item in query_metrics) / len(query_metrics)
        for key in keys
    }


def compare_rankings(
    first: list[int],
    second: list[int],
) -> dict[str, Any]:

    first_top = first[:TOP_K]
    second_top = second[:TOP_K]

    return {
        "top10_overlap": len(set(first_top) & set(second_top)),
        "identical_top10_order": first_top == second_top,
        "top1_match": first[0] == second[0],
    }


def compare_scores(
    first: dict[int, float],
    second: dict[int, float],
) -> dict[str, float]:

    if set(first) != set(second):
        raise ValueError("Score dictionaries contain different chunk IDs.")

    differences = np.asarray(
        [
            abs(first[chunk_id] - second[chunk_id])
            for chunk_id in first
        ],
        dtype=np.float64,
    )

    return {
        "mean_absolute_difference": float(np.mean(differences)),
        "max_absolute_difference": float(np.max(differences)),
        "p99_absolute_difference": float(
            np.percentile(differences, 99)
        ),
    }


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main() -> None:

    logger.info("Starting exact vector database comparison.")

    # 1. Load and validate evaluation data.
    questions = load_jsonl(QUESTIONS_FILE)
    ground_truth = load_jsonl(GROUND_TRUTH_FILE)

    questions = validate_datasets(questions, ground_truth)

    ground_truth_by_id = {
        item["question_id"]: item
        for item in ground_truth
    }

    # 2. Validate embedding configuration.
    embedding_config = EmbeddingConfig()

    if embedding_config.expected_dimension != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected embedding dimension {EXPECTED_DIMENSION}, "
            f"got {embedding_config.expected_dimension}."
        )

    if embedding_config.model_name != "intfloat/multilingual-e5-base":
        raise ValueError(
            f"Unexpected embedding model: {embedding_config.model_name}"
        )

    embedder = EmbeddingGenerator(config=embedding_config)

    # 3. Load the same stored vectors for the NumPy baseline.
    db = PostgreSQLClient()

    if not db.check_connection():
        raise RuntimeError("PostgreSQL connection failed.")

    chunk_ids, document_vectors = load_corpus(db)

    # 4. Initialize Qdrant.
    client = QdrantClient(
        url=QDRANT_URL,
        timeout=60,
    )

    if not client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Qdrant collection does not exist: {COLLECTION_NAME}"
        )

    qdrant_count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    if qdrant_count != len(chunk_ids):
        raise RuntimeError(
            f"Qdrant contains {qdrant_count} points, "
            f"PostgreSQL contains {len(chunk_ids)}."
        )

    # 5. Store all backend results.
    backend_names = (
        "postgresql_exact",
        "qdrant_exact",
        "numpy_exact",
    )

    metric_results = {
        name: [] for name in backend_names
    }

    pair_names = (
        ("postgresql_exact", "qdrant_exact"),
        ("postgresql_exact", "numpy_exact"),
        ("qdrant_exact", "numpy_exact"),
    )

    ranking_comparisons = {
        f"{first}_vs_{second}": []
        for first, second in pair_names
    }

    score_comparisons = {
        f"{first}_vs_{second}": []
        for first, second in pair_names
    }

    per_question_results = []

    # 6. One PostgreSQL transaction for all exact searches.
    with db.connection() as conn, conn.cursor() as cursor:

        first_query_vector = np.asarray(
            embedder.embed_query(questions[0]["question"]),
            dtype=np.float32,
        )

        plan = prepare_postgres_exact_search(
            cursor,
            first_query_vector,
        )

        logger.info(
            "PostgreSQL exact sequential scan verified."
        )

        for index, question_record in enumerate(questions, start=1):

            question_id = question_record["question_id"]
            question_text = question_record["question"].strip()

            relevant_ids = {
                str(chunk_id)
                for chunk_id in ground_truth_by_id[
                    question_id
                ]["relevant_chunk_ids"]
            }

            # Generate exactly one query embedding.
            if index == 1:
                query_vector = first_query_vector
            else:
                query_vector = np.asarray(
                    embedder.embed_query(question_text),
                    dtype=np.float32,
                )

            if query_vector.shape != (EXPECTED_DIMENSION,):
                raise ValueError(
                    f"Invalid query vector shape: {query_vector.shape}"
                )

            if not np.isfinite(query_vector).all():
                raise ValueError("Query vector contains invalid values.")

            # Same query vector goes to all three backends.
            pg_ids, pg_scores = postgres_exact_search(
                cursor,
                query_vector,
            )

            qdrant_ids, qdrant_scores = qdrant_exact_search(
                client,
                query_vector,
            )

            numpy_ids, numpy_scores = numpy_exact_search(
                chunk_ids,
                document_vectors,
                query_vector,
            )

            rankings = {
                "postgresql_exact": pg_ids,
                "qdrant_exact": qdrant_ids,
                "numpy_exact": numpy_ids,
            }

            scores = {
                "postgresql_exact": pg_scores,
                "qdrant_exact": qdrant_scores,
                "numpy_exact": numpy_scores,
            }

            # Verify every backend searched the same corpus.
            expected_ids = set(chunk_ids.tolist())

            for backend in backend_names:
                if set(rankings[backend]) != expected_ids:
                    raise RuntimeError(
                        f"{backend} returned a different corpus."
                    )

            # Evaluate all backends with identical ground truth.
            per_backend_metrics = {}

            for backend in backend_names:

                metrics = evaluate_query(
                    retrieved_ids=[
                        str(chunk_id)
                        for chunk_id in rankings[backend]
                    ],
                    relevant_ids=relevant_ids,
                    k_values=K_VALUES,
                )

                metric_results[backend].append(metrics)
                per_backend_metrics[backend] = metrics

            # Compare rankings and scores.
            per_query_ranking = {}
            per_query_scores = {}

            for first, second in pair_names:

                key = f"{first}_vs_{second}"

                ranking_result = compare_rankings(
                    rankings[first],
                    rankings[second],
                )

                score_result = compare_scores(
                    scores[first],
                    scores[second],
                )

                ranking_comparisons[key].append(ranking_result)
                score_comparisons[key].append(score_result)

                per_query_ranking[key] = ranking_result
                per_query_scores[key] = score_result

            per_question_results.append({
                "question_id": question_id,
                "paper_id": question_record["paper_id"],
                "metrics": per_backend_metrics,
                "top10_chunk_ids": {
                    backend: rankings[backend][:TOP_K]
                    for backend in backend_names
                },
                "ranking_agreement": per_query_ranking,
                "score_agreement": per_query_scores,
            })

            if index % 10 == 0 or index == len(questions):
                logger.info(
                    "Compared %d/%d questions.",
                    index,
                    len(questions),
                )

    # 7. Aggregate metrics.
    aggregate = {
        backend: aggregate_metrics(metric_results[backend])
        for backend in backend_names
    }

    ranking_summary = {}

    for key, results in ranking_comparisons.items():
        ranking_summary[key] = {
            "mean_top10_overlap": float(
                np.mean([item["top10_overlap"] for item in results])
            ),
            "identical_top10_order_count": sum(
                item["identical_top10_order"] for item in results
            ),
            "top1_match_count": sum(
                item["top1_match"] for item in results
            ),
            "question_count": len(results),
        }

    score_summary = {}

    for key, results in score_comparisons.items():
        score_summary[key] = {
            "mean_absolute_difference": float(
                np.mean([
                    item["mean_absolute_difference"]
                    for item in results
                ])
            ),
            "max_absolute_difference": float(
                max(
                    item["max_absolute_difference"]
                    for item in results
                )
            ),
            "p99_absolute_difference": float(
                np.percentile(
                    [
                        item["p99_absolute_difference"]
                        for item in results
                    ],
                    99,
                )
            ),
        }

    # 8. Save report.
    timestamp = datetime.now(timezone.utc)

    report = {
        "experiment": "exact_vector_database_comparison",
        "metadata": {
            "timestamp_utc": timestamp.isoformat(),
            "dataset": "QASPER",
            "embedding_model": embedding_config.model_name,
            "embedding_dimension": EXPECTED_DIMENSION,
            "normalized_embeddings": embedding_config.normalize_embeddings,
            "query_prefix": embedding_config.query_prefix,
            "corpus_chunks": len(chunk_ids),
            "collection": COLLECTION_NAME,
            "similarity_metric": "cosine",
            "search_mode": "exact",
            "postgresql_execution_plan": plan,
            "query_embeddings_generated_once_per_question": True,
            "k_values": list(K_VALUES),
            "aggregation": "macro_average",
            "relevance": "binary_chunk_relevance",
        },
        "total_questions": len(questions),
        "aggregate_metrics": aggregate,
        "ranking_agreement": ranking_summary,
        "score_agreement": score_summary,
        "per_question_results": per_question_results,
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = REPORTS_DIR / (
        f"vector_db_comparison_"
        f"{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    )

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )

    client.close()

    # 9. Display summary.
    logger.info("=" * 65)
    logger.info("EXACT VECTOR DATABASE COMPARISON COMPLETED")
    logger.info("=" * 65)

    for backend, metrics in aggregate.items():
        logger.info("\n%s", backend)

        for metric, value in metrics.items():
            logger.info("%s: %.6f", metric, value)

    logger.info("\nRanking agreement:")
    for key, values in ranking_summary.items():
        logger.info("%s: %s", key, values)

    logger.info("\nScore agreement:")
    for key, values in score_summary.items():
        logger.info("%s: %s", key, values)

    logger.info("Report saved to: %s", output_path)


if __name__ == "__main__":
    main()
