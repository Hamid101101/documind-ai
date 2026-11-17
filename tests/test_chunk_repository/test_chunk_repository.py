
import pytest

from database.repositories.chunk_repository import ChunkRepository
from retrieval.filters import RetrievalFilters

from uuid import UUID

from psycopg.types.json import Jsonb


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


def test_search_similar_filters_by_document_ids(
    chunk_repository,
    fake_db,
):
    document_ids = [UUID("11111111-1111-1111-1111-111111111111")]

    chunk_repository.search_similar(
        query_embedding=[0.1] * 384,
        filters=RetrievalFilters(document_ids=document_ids),
    )

    query = fake_db.cursor_instance.executed_query
    params = fake_db.cursor_instance.executed_params

    assert "c.document_id = ANY(%s)" in query
    assert document_ids == params[1]


def test_search_similar_filters_by_document_metadata(
    chunk_repository,
    fake_db,
):
    metadata = {"category": "research"}

    chunk_repository.search_similar(
        query_embedding=[0.1] * 384,
        filters=RetrievalFilters(document_metadata=metadata),
    )

    query = fake_db.cursor_instance.executed_query
    params = fake_db.cursor_instance.executed_params

    assert "JOIN documents AS d" in query
    assert "d.metadata @> %s::jsonb" in query
    assert params[1].obj == metadata


def test_search_similar_filters_by_chunk_metadata(
    chunk_repository,
    fake_db,
):
    metadata = {"section": "introduction"}

    chunk_repository.search_similar(
        query_embedding=[0.1] * 384,
        filters=RetrievalFilters(chunk_metadata=metadata),
    )

    query = fake_db.cursor_instance.executed_query
    params = fake_db.cursor_instance.executed_params

    assert "c.metadata @> %s::jsonb" in query
    assert params[1].obj == metadata


def test_search_similar_combines_all_filters(
    chunk_repository,
    fake_db,
):
    document_ids = [UUID("11111111-1111-1111-1111-111111111111")]
    document_metadata = {"category": "research"}
    chunk_metadata = {"section": "introduction"}

    chunk_repository.search_similar(
        query_embedding=[0.1] * 384,
        filters=RetrievalFilters(
            document_ids=document_ids,
            document_metadata=document_metadata,
            chunk_metadata=chunk_metadata,
        ),
    )

    query = fake_db.cursor_instance.executed_query
    params = fake_db.cursor_instance.executed_params

    assert "JOIN documents AS d" in query
    assert "c.document_id = ANY(%s)" in query
    assert "d.metadata @> %s::jsonb" in query
    assert "c.metadata @> %s::jsonb" in query

    assert params[1].obj == document_metadata
    assert params[2] == document_ids
    assert params[3].obj == chunk_metadata


def test_search_similar_without_filters_searches_all_documents(
    chunk_repository,
    fake_db,
):
    chunk_repository.search_similar(
        query_embedding=[0.1] * 384,
    )

    query = fake_db.cursor_instance.executed_query

    assert "JOIN documents AS d" in query
    assert "document_id = ANY" not in query
    assert "metadata @>" not in query


def test_search_keyword_returns_matching_chunks(
    chunk_repository,
    fake_db,
):
    fake_db.cursor_instance.fetchall_result = [
        {"content": "PostgreSQL vector search", "keyword_score": 0.8},
        {"content": "PostgreSQL database", "keyword_score": 0.5},
    ]

    result = chunk_repository.search_keyword(
        query_text="PostgreSQL",
        top_k=2,
    )

    assert len(result) == 2
    assert result[0]["keyword_score"] > result[1]["keyword_score"]
    assert fake_db.connection_count == 1


def test_search_keyword_uses_full_text_search(
    chunk_repository,
    fake_db,
):
    chunk_repository.search_keyword("PostgreSQL vector search")

    query = fake_db.cursor_instance.executed_query

    assert "search_vector" in query
    assert "@@" in query
    assert "websearch_to_tsquery" in query
    assert "ts_rank_cd" in query


def test_search_keyword_orders_by_relevance(
    chunk_repository,
    fake_db,
):
    chunk_repository.search_keyword("PostgreSQL")

    query = fake_db.cursor_instance.executed_query

    assert "ORDER BY keyword_score DESC" in query
    assert "LIMIT %s" in query


def test_search_keyword_passes_correct_parameters(
    chunk_repository,
    fake_db,
):
    chunk_repository.search_keyword(
        query_text="PostgreSQL vector search",
        top_k=3,
    )

    params = fake_db.cursor_instance.executed_params

    assert params == (
        "PostgreSQL vector search",
        "PostgreSQL vector search",
        3,
    )


def test_search_keyword_applies_document_id_filter(
    chunk_repository,
    fake_db,
    document_id,
):
    filters = RetrievalFilters(document_ids=[document_id])

    chunk_repository.search_keyword(
        query_text="PostgreSQL",
        filters=filters,
    )

    query = fake_db.cursor_instance.executed_query
    params = fake_db.cursor_instance.executed_params

    assert "c.document_id = ANY(%s)" in query
    assert params[2] == [document_id]


@pytest.mark.parametrize("query_text", ["", "   ", "\n"])
def test_search_keyword_rejects_empty_query(
    chunk_repository,
    query_text,
):
    with pytest.raises(ValueError, match="Query text"):
        chunk_repository.search_keyword(query_text)


@pytest.mark.parametrize("top_k", [0, -1])
def test_search_keyword_rejects_invalid_top_k(
    chunk_repository,
    top_k,
):
    with pytest.raises(ValueError, match="top_k"):
        chunk_repository.search_keyword(
            query_text="PostgreSQL",
            top_k=top_k,
        )
