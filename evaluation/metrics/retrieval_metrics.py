
from collections.abc import Sequence
from math import log2
from typing import Any


def _validate_inputs(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> None:
    if k <= 0:
        raise ValueError("k must be greater than zero.")

    if not relevant_ids:
        raise ValueError("relevant_ids cannot be empty.")

    if len(retrieved_ids) != len(set(retrieved_ids)):
        raise ValueError("retrieved_ids must not contain duplicates.")


def recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of relevant chunks retrieved within the top k."""
    _validate_inputs(retrieved_ids, relevant_ids, k)

    retrieved = set(retrieved_ids[:k])
    return len(retrieved & relevant_ids) / len(relevant_ids)


def precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Fraction of the top k results that are relevant."""
    _validate_inputs(retrieved_ids, relevant_ids, k)

    return sum(
        chunk_id in relevant_ids
        for chunk_id in retrieved_ids[:k]
    ) / k


def reciprocal_rank_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Reciprocal rank of the first relevant result within top k."""
    _validate_inputs(retrieved_ids, relevant_ids, k)

    for rank, chunk_id in enumerate(retrieved_ids[:k], start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank

    return 0.0


def ndcg_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """Normalized discounted cumulative gain with binary relevance."""
    _validate_inputs(retrieved_ids, relevant_ids, k)

    dcg = sum(
        1.0 / log2(rank + 1)
        for rank, chunk_id in enumerate(retrieved_ids[:k], start=1)
        if chunk_id in relevant_ids
    )

    ideal_relevant_count = min(len(relevant_ids), k)

    idcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(1, ideal_relevant_count + 1)
    )

    return dcg / idcg if idcg else 0.0


def evaluate_query(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k_values: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """Calculate retrieval metrics for one query."""
    results: dict[str, float] = {}

    for k in k_values:
        results[f"recall@{k}"] = recall_at_k(
            retrieved_ids, relevant_ids, k
        )
        results[f"precision@{k}"] = precision_at_k(
            retrieved_ids, relevant_ids, k
        )
        results[f"mrr@{k}"] = reciprocal_rank_at_k(
            retrieved_ids, relevant_ids, k
        )
        results[f"ndcg@{k}"] = ndcg_at_k(
            retrieved_ids, relevant_ids, k
        )

    return results


def aggregate_metrics(
    query_results: Sequence[dict[str, Any]],
) -> dict[str, float]:
    """Calculate macro-average metrics across evaluated queries."""
    if not query_results:
        raise ValueError("query_results cannot be empty.")

    metric_names = query_results[0].keys()

    return {
        metric: sum(
            result[metric] for result in query_results
        ) / len(query_results)
        for metric in metric_names
    }
