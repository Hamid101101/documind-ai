
import pytest

from ingestion.loaders.pdf_loader import PDFLoader
from ingestion.loaders.base import LoadedDocument


@pytest.fixture
def loader():
    return PDFLoader()


def test_pdf_loader_extracts_text(loader, sample_pdf):
    result = loader.load(str(sample_pdf))

    assert isinstance(result, LoadedDocument)
    assert result.source_format == "pdf"
    assert result.page_count == 2
    assert len(result.pages) == 2

    assert "First page content." in result.pages[0].text
    assert "Second page content." in result.pages[1].text


def test_pdf_loader_preserves_page_numbers(loader, sample_pdf):
    result = loader.load(str(sample_pdf))

    assert result.pages[0].page_number == 1
    assert result.pages[1].page_number == 2


def test_pdf_loader_handles_empty_page(loader, empty_pdf):
    result = loader.load(str(empty_pdf))

    assert result.page_count == 1
    assert len(result.pages) == 1
    assert result.pages[0].text.strip() == ""


def test_pdf_loader_file_not_found(loader, tmp_path):
    file_path = tmp_path / "missing.pdf"

    with pytest.raises(FileNotFoundError):
        loader.load(str(file_path))


def test_pdf_loader_rejects_invalid_pdf(loader, invalid_file):
    with pytest.raises(ValueError, match="invalid"):
        loader.load(str(invalid_file))
