
import pytest

from ingestion.pipeline import DocumentIngestionPipeline


def test_pipeline_processes_document_successfully(
    pipeline,
    pipeline_dependencies,
    document_id,
    tmp_path,
):
    file_path = tmp_path / "test.txt"
    file_path.write_text("Test document", encoding="utf-8")

    result = pipeline.process(document_id, file_path)

    assert result.status == "completed"
    assert result.chunk_count == 1
    assert result.embedding_model == "intfloat/multilingual-e5-small"

    deps = pipeline_dependencies

    deps["loader"].load.assert_called_once()
    deps["cleaner"].clean_document.assert_called_once()
    deps["chunker"].chunk_document.assert_called_once()
    deps["embedder"].embed_chunks.assert_called_once()

    deps["chunk_repository"].create_many.assert_called_once()
    deps["document_repository"].mark_completed.assert_called_once()


def test_pipeline_rejects_unsupported_extension(
    pipeline,
    document_id,
    tmp_path,
):
    file_path = tmp_path / "test.csv"
    file_path.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported"):
        pipeline.process(document_id, file_path)


def test_pipeline_rejects_unknown_document(
    pipeline,
    pipeline_dependencies,
    document_id,
    tmp_path,
):
    pipeline_dependencies["document_repository"].get_by_id.return_value = None

    file_path = tmp_path / "test.txt"
    file_path.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError, match="Document not found"):
        pipeline.process(document_id, file_path)


def test_pipeline_rejects_empty_chunks(
    pipeline,
    pipeline_dependencies,
    document_id,
    tmp_path,
):
    pipeline_dependencies["chunker"].chunk_document.return_value = []

    file_path = tmp_path / "test.txt"
    file_path.write_text("test", encoding="utf-8")

    with pytest.raises(ValueError, match="no extractable text"):
        pipeline.process(document_id, file_path)

    pipeline_dependencies["document_repository"].update_status.assert_called_once()
    pipeline_dependencies["chunk_repository"].create_many.assert_not_called()


def test_pipeline_marks_document_failed_on_loader_error(
    pipeline,
    pipeline_dependencies,
    document_id,
    tmp_path,
):
    pipeline_dependencies["loader"].load.side_effect = RuntimeError(
        "Loader failure"
    )

    file_path = tmp_path / "test.txt"
    file_path.write_text("test", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Loader failure"):
        pipeline.process(document_id, file_path)

    pipeline_dependencies["document_repository"].update_status.assert_called_once()
