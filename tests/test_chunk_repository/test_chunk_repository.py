
import pytest

from database.repositories.chunk_repository import ChunkRepository


def test_create_many_inserts_all_chunks(
    chunk_repository,
    fake_db,
    document_id,
    make_chunks,
    make_embeddings,
):
    chunks = make_chunks(["First chunk", "Second chunk"])
    embeddings = make_embeddings(2)

    result = chunk_repository.create_many(
        document_id,
        chunks,
        embeddings,
    )

    assert result == 2
    assert fake_db.connection_count == 1
    assert len(fake_db.cursor_instance.batch_params) == 2


def test_create_many_uses_vector_cast(
    chunk_repository,
    fake_db,
    document_id,
    make_chunks,
    make_embeddings,
):
    chunk_repository.create_many(
        document_id,
        make_chunks(["Test chunk"]),
        make_embeddings(1),
    )

    assert "%s::vector" in fake_db.cursor_instance.executed_query


def test_create_many_serializes_vector(
    chunk_repository,
    fake_db,
    document_id,
    make_chunks,
    make_embeddings,
):
    chunk_repository.create_many(
        document_id,
        make_chunks(["Test chunk"]),
        make_embeddings(1),
    )

    params = fake_db.cursor_instance.batch_params[0]

    assert params[0] == document_id
    assert params[7].startswith("[")
    assert params[7].endswith("]")


def test_create_many_returns_zero_for_empty_input(
    chunk_repository,
    fake_db,
    document_id,
):
    result = chunk_repository.create_many(
        document_id,
        [],
        [],
    )

    assert result == 0
    assert fake_db.connection_count == 0


def test_create_many_rejects_mismatched_lengths(
    chunk_repository,
    document_id,
    make_chunks,
    make_embeddings,
):
    with pytest.raises(ValueError, match="must match"):
        chunk_repository.create_many(
            document_id,
            make_chunks(["First", "Second"]),
            make_embeddings(1),
        )


def test_create_many_rejects_wrong_embedding_dimension(
    chunk_repository,
    document_id,
    make_chunks,
):
    with pytest.raises(ValueError, match="384 dimensions"):
        chunk_repository.create_many(
            document_id,
            make_chunks(["Test"]),
            [[0.1] * 128],
        )


def test_create_many_rejects_non_finite_embedding(
    chunk_repository,
    document_id,
    make_chunks,
):
    embedding = [0.1] * 384
    embedding[0] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        chunk_repository.create_many(
            document_id,
            make_chunks(["Test"]),
            [embedding],
        )


def test_create_many_rejects_duplicate_chunk_indexes(
    chunk_repository,
    document_id,
    make_chunks,
    make_embeddings,
):
    chunks = make_chunks(["First", "Second"])
    chunks[1].chunk_index = chunks[0].chunk_index

    with pytest.raises(ValueError, match="unique"):
        chunk_repository.create_many(
            document_id,
            chunks,
            make_embeddings(2),
        )


def test_get_by_document_id(
    chunk_repository,
    fake_db,
    document_id,
):
    fake_db.cursor_instance.fetchall_result = [
        {"chunk_index": 0, "content": "First chunk"},
        {"chunk_index": 1, "content": "Second chunk"},
    ]

    result = chunk_repository.get_by_document_id(document_id)

    assert len(result) == 2
    assert "ORDER BY chunk_index ASC" in fake_db.cursor_instance.executed_query
    assert fake_db.cursor_instance.executed_params == (document_id,)


def test_count_by_document_id(
    chunk_repository,
    fake_db,
    document_id,
):
    result = chunk_repository.count_by_document_id(document_id)

    assert result == 2
    assert "COUNT(*)" in fake_db.cursor_instance.executed_query


def test_delete_by_document_id(
    chunk_repository,
    fake_db,
    document_id,
    make_chunks,
    make_embeddings,
):
    # FakeCursor returns rowcount=0 for execute by default.
    result = chunk_repository.delete_by_document_id(document_id)

    assert result == 0
    assert "DELETE FROM document_chunks" in fake_db.cursor_instance.executed_query
    assert fake_db.cursor_instance.executed_params == (document_id,)

def test_search_similar_returns_ranked_chunks(
    chunk_repository,
    fake_db,
):
    fake_db.cursor_instance.fetchall_result = [
        {"content": "Most relevant chunk", "similarity": 0.95},
        {"content": "Second relevant chunk", "similarity": 0.82},
    ]

    query_embedding = [0.1] * 384

    result = chunk_repository.search_similar(
        query_embedding=query_embedding,
        top_k=2,
    )

    assert len(result) == 2
    assert result[0]["similarity"] > result[1]["similarity"]
    assert fake_db.connection_count == 1


def test_search_similar_uses_cosine_distance(
    chunk_repository,
    fake_db,
):
    chunk_repository.search_similar([0.1] * 384)

    query = fake_db.cursor_instance.executed_query

    assert "<=>" in query
    assert "ORDER BY" in query
    assert "LIMIT" in query


def test_search_similar_uses_vector_cast(
    chunk_repository,
    fake_db,
):
    chunk_repository.search_similar([0.1] * 384)

    query = fake_db.cursor_instance.executed_query

    assert query.count("%s::vector") == 2


def test_search_similar_passes_correct_parameters(
    chunk_repository,
    fake_db,
):
    embedding = [0.1] * 384

    chunk_repository.search_similar(
        query_embedding=embedding,
        top_k=3,
    )

    params = fake_db.cursor_instance.executed_params

    assert len(params) == 3
    assert params[0] == params[1]
    assert params[2] == 3


def test_search_similar_rejects_invalid_top_k(chunk_repository):
    with pytest.raises(ValueError, match="top_k"):
        chunk_repository.search_similar(
            query_embedding=[0.1] * 384,
            top_k=0,
        )


def test_search_similar_rejects_wrong_embedding_dimension(
    chunk_repository,
):
    with pytest.raises(ValueError, match="384 dimensions"):
        chunk_repository.search_similar(
            query_embedding=[0.1] * 128,
        )


def test_search_similar_rejects_non_finite_embedding(
    chunk_repository,
):
    embedding = [0.1] * 384
    embedding[0] = float("nan")

    with pytest.raises(ValueError, match="non-finite"):
        chunk_repository.search_similar(embedding)
