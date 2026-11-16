
import pytest

from ingestion.loaders.docx_loader import DOCXLoader
from ingestion.loaders.base import LoadedDocument


@pytest.fixture
def loader():
    return DOCXLoader()


def test_docx_loader_extracts_paragraphs(loader, sample_docx):
    result = loader.load(str(sample_docx))

    assert isinstance(result, LoadedDocument)
    assert result.source_format == "docx"
    assert result.page_count is None
    assert len(result.pages) == 1

    text = result.pages[0].text

    assert "Professional Experience" in text
    assert "Data Scientist at MLnetworks." in text


def test_docx_loader_extracts_tables(loader, sample_docx):
    result = loader.load(str(sample_docx))

    text = result.pages[0].text

    assert "Skill" in text
    assert "Level" in text
    assert "Python" in text
    assert "Advanced" in text
    assert "Python | Advanced" in text


def test_docx_loader_page_number_is_none(loader, sample_docx):
    result = loader.load(str(sample_docx))

    assert result.pages[0].page_number is None


def test_docx_loader_handles_empty_document(loader, empty_docx):
    result = loader.load(str(empty_docx))

    assert result.source_format == "docx"
    assert result.page_count is None
    assert len(result.pages) == 1
    assert result.pages[0].text == ""


def test_docx_loader_file_not_found(loader, tmp_path):
    file_path = tmp_path / "missing.docx"

    with pytest.raises(FileNotFoundError):
        loader.load(str(file_path))


def test_docx_loader_rejects_invalid_file(loader, invalid_file):
    with pytest.raises(ValueError, match="DOCX"):
        loader.load(str(invalid_file))
