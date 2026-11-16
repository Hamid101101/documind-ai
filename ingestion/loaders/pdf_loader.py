
import logging
from pathlib import Path

import pymupdf

from ingestion.loaders.base import (
    BaseLoader,
    DocumentPage,
    LoadedDocument,
)

logger = logging.getLogger(__name__)


class PDFLoader(BaseLoader):

    def load(self, file_path: str) -> LoadedDocument:
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")

        pages = []

        try:
            with pymupdf.open(path) as document:

                if document.is_encrypted:
                    raise ValueError("Encrypted PDFs are not supported.")

                for index, page in enumerate(document):
                    text = page.get_text("text")

                    pages.append(
                        DocumentPage(
                            page_number=index + 1,
                            text=text,
                        )
                    )

                return LoadedDocument(
                    source_format="pdf",
                    page_count=len(document),
                    pages=pages,
                )

        except pymupdf.FileDataError as exc:
            logger.exception("Invalid PDF file: %s", path)
            raise ValueError("The provided PDF is invalid.") from exc
