
import pytest
import pymupdf

from docx import Document


@pytest.fixture
def sample_pdf(tmp_path):
    """Create a PDF containing two pages."""

    file_path = tmp_path / "sample.pdf"

    with pymupdf.open() as pdf:
        page1 = pdf.new_page()
        page1.insert_text((72, 72), "First page content.")

        page2 = pdf.new_page()
        page2.insert_text((72, 72), "Second page content.")

        pdf.save(file_path)

    return file_path


@pytest.fixture
def empty_pdf(tmp_path):
    """Create a valid PDF with one empty page."""

    file_path = tmp_path / "empty.pdf"

    with pymupdf.open() as pdf:
        pdf.new_page()
        pdf.save(file_path)

    return file_path


@pytest.fixture
def sample_docx(tmp_path):
    """Create a DOCX with paragraphs and a table."""

    file_path = tmp_path / "sample.docx"

    document = Document()

    document.add_paragraph("Professional Experience")
    document.add_paragraph("Data Scientist at MLnetworks.")

    table = document.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Skill"
    table.cell(0, 1).text = "Level"
    table.cell(1, 0).text = "Python"
    table.cell(1, 1).text = "Advanced"

    document.save(file_path)

    return file_path


@pytest.fixture
def empty_docx(tmp_path):
    """Create an empty DOCX."""

    file_path = tmp_path / "empty.docx"

    document = Document()
    document.save(file_path)

    return file_path


@pytest.fixture
def sample_txt(tmp_path):
    """Create a UTF-8 text document."""

    file_path = tmp_path / "sample.txt"

    file_path.write_text(
        "First paragraph.\n\nSecond paragraph.",
        encoding="utf-8",
    )

    return file_path


@pytest.fixture
def empty_txt(tmp_path):
    """Create an empty TXT file."""

    file_path = tmp_path / "empty.txt"
    file_path.write_text("", encoding="utf-8")

    return file_path


@pytest.fixture
def invalid_file(tmp_path):
    """Create a file containing invalid document data."""

    file_path = tmp_path / "invalid.pdf"
    file_path.write_text("This is not a valid PDF.")

    return file_path
