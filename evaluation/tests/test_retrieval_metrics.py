
from evaluation.metrics.retrieval_metrics import (
    evaluate_query,
    aggregate_metrics,
)


def test_retrieval_metrics():
    relevant = {"A", "B", "C"}
    retrieved = ["X", "B", "A", "Y", "C"]

    results = evaluate_query(
        retrieved,
        relevant,
        k_values=(1, 3, 5),
    )

    assert results["recall@1"] == 0
    assert results["recall@3"] == 2 / 3
    assert results["precision@3"] == 2 / 3
    assert results["mrr@3"] == 0.5
    assert results["recall@5"] == 1.0


def test_aggregate_metrics():
    results = [
        {"recall@5": 1.0, "mrr@5": 1.0},
        {"recall@5": 0.5, "mrr@5": 0.5},
    ]

    aggregated = aggregate_metrics(results)

    assert aggregated["recall@5"] == 0.75
    assert aggregated["mrr@5"] == 0.75
