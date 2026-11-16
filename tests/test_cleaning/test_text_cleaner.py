
import pytest

from ingestion.cleaning.text_cleaner import TextCleaner
from ingestion.loaders.base import DocumentPage, LoadedDocument


@pytest.fixture
def cleaner():
    return TextCleaner()


# ==========================================
# Basic cleaning
# ==========================================

def test_clean_text_removes_leading_trailing_whitespace(cleaner):
    text = "   Hello World   "

    result = cleaner.clean_text(text)

    assert result == "Hello World"


def test_clean_text_normalizes_multiple_spaces(cleaner):
    text = "Hello     World"

    result = cleaner.clean_text(text)

    assert result == "Hello World"


def test_clean_text_normalizes_tabs(cleaner):
    text = "Hello\t\tWorld"

    result = cleaner.clean_text(text)

    assert result == "Hello World"


def test_clean_text_preserves_paragraph_boundaries(cleaner):
    text = "First paragraph.\n\nSecond paragraph."

    result = cleaner.clean_text(text)

    assert result == text


def test_clean_text_reduces_excessive_newlines(cleaner):
    text = "First paragraph.\n\n\n\nSecond paragraph."

    result = cleaner.clean_text(text)

    assert result == "First paragraph.\n\nSecond paragraph."


# ==========================================
# Line ending normalization
# ==========================================

def test_clean_text_normalizes_windows_line_endings(cleaner):
    text = "First line.\r\nSecond line."

    result = cleaner.clean_text(text)

    assert result == "First line.\nSecond line."


def test_clean_text_normalizes_mac_line_endings(cleaner):
    text = "First line.\rSecond line."

    result = cleaner.clean_text(text)

    assert result == "First line.\nSecond line."


# ==========================================
# Unicode normalization
# ==========================================

def test_clean_text_normalizes_unicode(cleaner):
    text = "ＡＢＣ"

    result = cleaner.clean_text(text)

    assert result == "ABC"


def test_clean_text_preserves_french_accents(cleaner):
    text = "Développeur spécialisé en intelligence artificielle."

    result = cleaner.clean_text(text)

    assert result == text


def test_clean_text_preserves_arabic(cleaner):
    text = "الذكاء الاصطناعي وتحليل البيانات"

    result = cleaner.clean_text(text)

    assert result == text


def test_clean_text_converts_non_breaking_spaces(cleaner):
    text = "Hello\u00A0World"

    result = cleaner.clean_text(text)

    assert result == "Hello World"


# ==========================================
# Control characters
# ==========================================

def test_clean_text_removes_null_bytes(cleaner):
    text = "Hello\x00World"

    result = cleaner.clean_text(text)

    assert result == "HelloWorld"


def test_clean_text_removes_unwanted_control_characters(cleaner):
    text = "Hello\x07World"

    result = cleaner.clean_text(text)

    assert result == "HelloWorld"


def test_clean_text_preserves_newlines_and_tabs(cleaner):
    text = "First line.\nSecond\tline."

    result = cleaner.clean_text(text)

    assert result == "First line.\nSecond line."


# ==========================================
# Empty and invalid inputs
# ==========================================

def test_clean_text_handles_empty_string(cleaner):
    assert cleaner.clean_text("") == ""


def test_clean_text_handles_whitespace_only(cleaner):
    assert cleaner.clean_text("   \n\t  ") == ""


def test_clean_text_rejects_non_string_input(cleaner):
    with pytest.raises(TypeError):
        cleaner.clean_text(None)


# ==========================================
# Document cleaning
# ==========================================

def test_clean_document_preserves_document_metadata(cleaner):
    document = LoadedDocument(
        source_format="pdf",
        page_count=2,
        pages=[
            DocumentPage(
                page_number=1,
                text="  First page.   ",
            ),
            DocumentPage(
                page_number=2,
                text="  Second page.   ",
            ),
        ],
    )

    result = cleaner.clean_document(document)

    assert result.source_format == "pdf"
    assert result.page_count == 2
    assert len(result.pages) == 2

    assert result.pages[0].page_number == 1
    assert result.pages[0].text == "First page."

    assert result.pages[1].page_number == 2
    assert result.pages[1].text == "Second page."


def test_clean_document_does_not_modify_original_document(cleaner):
    document = LoadedDocument(
        source_format="txt",
        page_count=None,
        pages=[
            DocumentPage(
                page_number=None,
                text="   Original   text   ",
            )
        ],
    )

    result = cleaner.clean_document(document)

    assert document.pages[0].text == "   Original   text   "
    assert result.pages[0].text == "Original text"


def test_clean_document_handles_empty_pages(cleaner):
    document = LoadedDocument(
        source_format="pdf",
        page_count=1,
        pages=[
            DocumentPage(
                page_number=1,
                text="   ",
            )
        ],
    )

    result = cleaner.clean_document(document)

    assert result.pages[0].text == ""
    assert result.pages[0].page_number == 1
