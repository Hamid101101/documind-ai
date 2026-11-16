
import logging
from pathlib import Path

from docx import Document
from docx.document import Document as DocumentObject
from docx.table import Table
from docx.text.paragraph import Paragraph
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P

from ingestion.loaders.base import (
    BaseLoader,
    DocumentPage,
    LoadedDocument,
)

logger = logging.getLogger(__name__)


class DOCXLoader(BaseLoader):

    @staticmethod
    def _extract_body(document: DocumentObject) -> str:
        content = []

        # Preserve the order of paragraphs and tables.
        for element in document.element.body.iterchildren():

            if isinstance(element, CT_P):
                paragraph = Paragraph(element, document)

                if paragraph.text.strip():
                    content.append(paragraph.text)

            elif isinstance(element, CT_Tbl):
                table = Table(element, document)

                for row in table.rows:
                    cells = [
                        cell.text.strip()
                        for cell in row.cells
                    ]

                    content.append(" | ".join(cells))

        return "\n\n".join(content)

    def load(self, file_path: str) -> LoadedDocument:
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"DOCX not found: {path}")

        try:
            document = Document(path)
            text = self._extract_body(document)

            return LoadedDocument(
                source_format="docx",
                page_count=None,
                pages=[
                    DocumentPage(
                        page_number=None,
                        text=text,
                    )
                ],
            )

        except Exception as exc:
            logger.exception("Failed to load DOCX: %s", path)
            raise ValueError("Could not extract the DOCX document.") from exc
