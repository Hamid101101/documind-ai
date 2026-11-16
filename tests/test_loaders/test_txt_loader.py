
import pytest

from ingestion.loaders.txt_loader import TXTLoader
from ingestion.loaders.base import LoadedDocument


@pytest.fixture
def loader():
    return TXTLoader()


def test_txt_loader_extracts_text(loader, sample_txt):
    result = loader.load(str(sample_txt))

    assert isinstance(result, LoadedDocument)
    assert result.source_format == "txt"
    assert result.page_count is None
    assert len(result.pages) == 1

    assert "First paragraph." in result.pages[0].text
    assert "Second paragraph." in result.pages[0].text


def test_txt_loader_handles_utf8_bom(loader, tmp_path):
    file_path = tmp_path / "bom.txt"

    file_path.write_text(
        "\ufeffDocument content.",
        encoding="utf-8",
    )

    result = loader.load(str(file_path))

    assert result.pages[0].text == "Document content."


def test_txt_loader_page_number_is_none(loader, sample_txt):
    result = loader.load(str(sample_txt))

    assert result.pages[0].page_number is None


def test_txt_loader_handles_empty_file(loader, empty_txt):
    result = loader.load(str(empty_txt))

    assert result.source_format == "txt"
    assert result.pages[0].text == ""


def test_txt_loader_rejects_invalid_encoding(loader, tmp_path):
    file_path = tmp_path / "invalid_encoding.txt"

    file_path.write_bytes(b"\xff\xfe\xfa")

    with pytest.raises(ValueError, match="UTF-8"):
        loader.load(str(file_path))


def test_txt_loader_file_not_found(loader, tmp_path):
    file_path = tmp_path / "missing.txt"

    with pytest.raises(FileNotFoundError):
        loader.load(str(file_path))
