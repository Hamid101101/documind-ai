
import pytest

from retrieval.config import RetrievalConfig


def test_default_configuration():
    config = RetrievalConfig()

    assert config.top_k == 5
    assert config.max_top_k == 20
    assert config.similarity_threshold is None


def test_custom_configuration():
    config = RetrievalConfig(
        top_k=10,
        max_top_k=30,
        similarity_threshold=0.7,
    )

    assert config.top_k == 10
    assert config.max_top_k == 30
    assert config.similarity_threshold == 0.7


@pytest.mark.parametrize("top_k", [0, -1])
def test_rejects_invalid_top_k(top_k):
    with pytest.raises(ValueError, match="top_k"):
        RetrievalConfig(top_k=top_k)


@pytest.mark.parametrize("max_top_k", [0, -1])
def test_rejects_invalid_max_top_k(max_top_k):
    with pytest.raises(ValueError, match="max_top_k"):
        RetrievalConfig(max_top_k=max_top_k)


def test_rejects_top_k_greater_than_max_top_k():
    with pytest.raises(ValueError, match="cannot exceed"):
        RetrievalConfig(top_k=25, max_top_k=20)


@pytest.mark.parametrize("threshold", [-1.1, 1.1])
def test_rejects_invalid_similarity_threshold(threshold):
    with pytest.raises(ValueError, match="between -1 and 1"):
        RetrievalConfig(similarity_threshold=threshold)


@pytest.mark.parametrize("threshold", [-1.0, 0.0, 1.0])
def test_accepts_valid_similarity_threshold(threshold):
    config = RetrievalConfig(similarity_threshold=threshold)

    assert config.similarity_threshold == threshold
