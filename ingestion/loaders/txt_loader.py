
import logging
from pathlib import Path

from ingestion.loaders.base import (
    BaseLoader,
    DocumentPage,
    LoadedDocument,
)

logger = logging.getLogger(__name__)


class TXTLoader(BaseLoader):

    def load(self, file_path: str) -> LoadedDocument:
        path = Path(file_path)

        if not path.is_file():
            raise FileNotFoundError(f"TXT file not found: {path}")

        try:
            text = path.read_text(encoding="utf-8-sig")

            return LoadedDocument(
                source_format="txt",
                page_count=None,
                pages=[
                    DocumentPage(
                        page_number=None,
                        text=text,
                    )
                ],
            )

        except UnicodeDecodeError as exc:
            logger.exception("Unsupported text encoding: %s", path)
            raise ValueError(
                "TXT files must currently use UTF-8 encoding."
            ) from exc
