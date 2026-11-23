
from typing import Any, Sequence

from httpx2 import query

from retrieval.vector_search import VectorSearch
from evaluation.metrics.retrieval_metrics import evaluate_query


class RetrievalEvaluator:
    """Evaluate dense retrieval against a chunk-level ground-truth dataset."""

    def __init__(
        self,
        vector_search: VectorSearch | None = None,
        k_values: Sequence[int] = (1, 3, 5, 10),
        retrieval_method: str = "dense",
    ) -> None:
        if not k_values or any(k <= 0 for k in k_values):
            raise ValueError("k_values must contain positive integers.")

        if len(set(k_values)) != len(k_values):
            raise ValueError("k_values must not contain duplicates.")

        self.k_values = tuple(sorted(k_values))
        self.top_k = max(self.k_values)
        self.vector_search = vector_search or VectorSearch()

        if self.top_k > self.vector_search.config.max_top_k:
            raise ValueError(
                f"top_k={self.top_k} exceeds the configured maximum "
                f"of {self.vector_search.config.max_top_k}."
            )

        if retrieval_method not in {"dense", "hybrid"}:
            raise ValueError("retrieval_method must be 'dense' or 'hybrid'.")

        self.retrieval_method = retrieval_method

    def evaluate(
        self,
        questions: list[dict[str, Any]],
        ground_truth: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Evaluate every question and return per-query and aggregate results."""

        ground_truth_by_id = {
            item["question_id"]: item for item in ground_truth
        }

        question_ids = [item["question_id"] for item in questions]

        if len(question_ids) != len(set(question_ids)):
            raise ValueError("Duplicate question IDs found in questions.")

        results = []
        failures = []

        for index, question_record in enumerate(questions, start=1):
            question_id = question_record["question_id"]

            if question_id not in ground_truth_by_id:
                raise ValueError(
                    f"Missing ground truth for question {question_id}."
                )

            query = question_record["question"].strip()

            if not query:
                raise ValueError(
                    f"Question {question_id} has empty question text."
                )

            relevant_ids = {
                str(chunk_id)
                for chunk_id in ground_truth_by_id[question_id]["relevant_chunk_ids"]
            }

            try:
                search_method = (
                    self.vector_search.hybrid_search
                    if self.retrieval_method == "hybrid"
                    else self.vector_search.search
                )

                retrieved_chunks = search_method(
                    query=query,
                    top_k=self.top_k,
                    filters=None,
            )

                retrieved_ids = [
                        str(chunk["id"]) for chunk in retrieved_chunks
                ]

                metrics = evaluate_query(
                    retrieved_ids=retrieved_ids,
                    relevant_ids=relevant_ids,
                    k_values=self.k_values,
                )

                error = None

            except Exception as exc:
                # Keep failed queries in the evaluation.
                # They receive zero retrieval metrics instead of disappearing.
                retrieved_ids = []
                metrics = evaluate_query(
                    retrieved_ids=[],
                    relevant_ids=relevant_ids,
                    k_values=self.k_values,
                )

                error = f"{type(exc).__name__}: {exc}"

                failures.append({
                    "question_id": question_id,
                    "error": error,
                })

            results.append({
                "question_id": question_id,
                "paper_id": question_record["paper_id"],
                "question": query,
                "relevant_chunk_count": len(relevant_ids),
                "retrieved_chunk_ids": retrieved_ids,
                "retrieved_count": len(retrieved_ids),
                "metrics": metrics,
                "error": error,
            })

            if index % 10 == 0 or index == len(questions):
                print(
                    f"Evaluated {index}/{len(questions)} questions"
                )

        if len(ground_truth_by_id) != len(questions):
            raise ValueError(
                "Questions and ground truth have different sizes. "
                "Check for unmatched question IDs."
            )

        aggregate = self._aggregate(results)

        return {
            "experiment": "dense_vector_retrieval",
            "retrieval_method": self.retrieval_method,
            "corpus_scope": "global",
            "k_values": list(self.k_values),
            "total_questions": len(questions),
            "successful_questions": len(questions) - len(failures),
            "failed_questions": len(failures),
            "failures": failures,
            "aggregate_metrics": aggregate,
            "per_question_results": results,
        }

    @staticmethod
    def _aggregate(
        results: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Calculate macro-average metrics across all questions."""

        if not results:
            raise ValueError("Cannot aggregate an empty evaluation.")

        metric_keys = results[0]["metrics"].keys()

        return {
            metric: sum(
                result["metrics"][metric] for result in results
            ) / len(results)
            for metric in metric_keys
        }
