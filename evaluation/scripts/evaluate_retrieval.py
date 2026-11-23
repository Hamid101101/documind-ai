
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from evaluation.retrieval.evaluator import RetrievalEvaluator
from ingestion.embeddings.config import EmbeddingConfig
from retrieval.config import RetrievalConfig
from retrieval.vector_search import VectorSearch


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

GROUND_TRUTH_DIR = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "processed"
    / "ground_truth"
)

QUESTIONS_FILE = GROUND_TRUTH_DIR / "questions.jsonl"

GROUND_TRUTH_FILE = (
    GROUND_TRUTH_DIR / "retrieval_ground_truth.jsonl"
)

REPORTS_DIR = PROJECT_ROOT / "evaluation" / "reports"

K_VALUES = (1, 3, 5, 10)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def load_jsonl(file_path: Path) -> list[dict[str, Any]]:
    """Load a JSONL file and validate its records."""

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    records = []

    with file_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {file_path}, "
                    f"line {line_number}: {exc}"
                ) from exc

            if not isinstance(record, dict):
                raise ValueError(
                    f"Expected a JSON object at line {line_number} "
                    f"in {file_path}."
                )

            records.append(record)

    if not records:
        raise ValueError(f"No records found in {file_path}")

    return records



def validate_datasets(
    questions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Validate alignment and select questions with retrieval ground truth."""

    question_ids = [item.get("question_id") for item in questions]
    ground_truth_ids = [
        item.get("question_id") for item in ground_truth
    ]

    if None in question_ids or None in ground_truth_ids:
        raise ValueError("Every record must have a question_id.")

    if len(question_ids) != len(set(question_ids)):
        raise ValueError("Duplicate question IDs in questions dataset.")

    if len(ground_truth_ids) != len(set(ground_truth_ids)):
        raise ValueError("Duplicate question IDs in ground truth.")

    questions_by_id = {
        item["question_id"]: item for item in questions
    }

    question_set = set(question_ids)
    ground_truth_set = set(ground_truth_ids)

    unknown_ground_truth = ground_truth_set - question_set

    if unknown_ground_truth:
        raise ValueError(
            f"{len(unknown_ground_truth)} ground-truth records "
            "have no corresponding question."
        )

    for record in questions:
        if not isinstance(record.get("question"), str):
            raise ValueError(
                f"Invalid question text: {record['question_id']}"
            )

    for record in ground_truth:
        relevant_ids = record.get("relevant_chunk_ids")

        if not isinstance(relevant_ids, list) or not relevant_ids:
            raise ValueError(
                f"Invalid relevant_chunk_ids: {record['question_id']}"
            )

    # Evaluate only questions with valid chunk-level ground truth.
    eligible_questions = [
        questions_by_id[question_id]
        for question_id in ground_truth_ids
    ]

    excluded_count = len(questions) - len(eligible_questions)

    logger.info(
        "Dataset validation passed: %d eligible questions.",
        len(eligible_questions),
    )
    logger.info(
        "Excluded questions without retrieval ground truth: %d.",
        excluded_count,
    )

    return eligible_questions


def save_report(
    report: dict[str, Any],
    embedding_config: EmbeddingConfig,
    retrieval_config: RetrievalConfig,
) -> Path:
    """Save the evaluation report as a timestamped JSON file."""

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc)

    report["metadata"] = {
        "timestamp_utc": timestamp.isoformat(),
        "dataset": "QASPER",
        "embedding_model": embedding_config.model_name,
        "embedding_dimension": embedding_config.expected_dimension,
        "normalized_embeddings": embedding_config.normalize_embeddings,
        "query_prefix": embedding_config.query_prefix,
        "retrieval_top_k": max(K_VALUES),
        "similarity_threshold": retrieval_config.similarity_threshold,
        "corpus_scope": "global",
        "metric_relevance": "binary_chunk_relevance",
        "aggregation": "macro_average",
    }

    filename = (
        f"dense_retrieval_"
        f"{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    )

    output_path = REPORTS_DIR / filename

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(
            report,
            file,
            indent=2,
            ensure_ascii=False,
        )

    return output_path


def main() -> None:
    logger.info("Starting dense retrieval evaluation.")

    # 1. Load datasets
    questions = load_jsonl(QUESTIONS_FILE)
    ground_truth = load_jsonl(GROUND_TRUTH_FILE)

    logger.info("Loaded %d questions.", len(questions))
    logger.info("Loaded %d ground-truth records.", len(ground_truth))

    # 2. Validate datasets
    questions = validate_datasets(questions, ground_truth)

    # 3. Initialize existing configurations
    embedding_config = EmbeddingConfig()
    retrieval_config = RetrievalConfig()

    vector_search = VectorSearch(
        config=retrieval_config,
    )

    evaluator = RetrievalEvaluator(
        vector_search=vector_search,
        k_values=K_VALUES,
        retrieval_method="hybrid",
    )

    # 4. Run evaluation
    report = evaluator.evaluate(
        questions=questions,
        ground_truth=ground_truth,
    )

    # 5. Attach experiment configuration and save
    output_path = save_report(
        report=report,
        embedding_config=embedding_config,
        retrieval_config=retrieval_config,
    )

    # 6. Display results
    logger.info("=" * 60)
    logger.info("RETRIEVAL EVALUATION COMPLETED")
    logger.info("=" * 60)

    logger.info(
        "Successful queries: %d/%d",
        report["successful_questions"],
        report["total_questions"],
    )

    logger.info(
        "Failed queries: %d",
        report["failed_questions"],
    )

    logger.info("Aggregate metrics:")

    for metric, value in report["aggregate_metrics"].items():
        logger.info("%s: %.4f", metric, value)

    logger.info("Report saved to: %s", output_path)


if __name__ == "__main__":
    main()
