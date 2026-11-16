
import re
import unicodedata

from ingestion.loaders.base import DocumentPage, LoadedDocument


class TextCleaner:
    """
    Clean extracted document text while preserving its meaning
    and paragraph structure.
    """

    # Match horizontal whitespace without consuming newlines.
    HORIZONTAL_WHITESPACE = re.compile(r"[^\S\n]+")

    # Match more than two consecutive newlines.
    EXCESSIVE_NEWLINES = re.compile(r"\n{3,}")

    @staticmethod
    def _normalize_unicode(text: str) -> str:
        """Normalize Unicode compatibility characters."""
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _normalize_line_endings(text: str) -> str:
        """Normalize Windows and legacy Mac line endings."""
        return text.replace("\r\n", "\n").replace("\r", "\n")

    @staticmethod
    def _remove_control_characters(text: str) -> str:
        """
        Remove unwanted Unicode control characters.

        Preserve newline and tab characters for further processing.
        """

        return "".join(
            char
            for char in text
            if char in ("\n", "\t")
            or not unicodedata.category(char).startswith("C")
        )

    def clean_text(self, text: str) -> str:
        """
        Clean a single text string.

        The operation is deterministic and does not perform
        translation, summarization, or semantic modification.
        """

        if not isinstance(text, str):
            raise TypeError("Text must be a string.")

        if not text:
            return ""

        # 1. Unicode normalization.
        text = self._normalize_unicode(text)

        # 2. Normalize line endings.
        text = self._normalize_line_endings(text)

        # 3. Convert non-breaking spaces to regular spaces.
        text = text.replace("\u00A0", " ")

        # 4. Remove unwanted control characters.
        text = self._remove_control_characters(text)

        # 5. Normalize horizontal whitespace.
        text = self.HORIZONTAL_WHITESPACE.sub(" ", text)

        # 6. Reduce excessive blank lines.
        text = self.EXCESSIVE_NEWLINES.sub("\n\n", text)

        # 7. Remove leading and trailing whitespace.
        return text.strip()

    def clean_document(
        self,
        document: LoadedDocument,
    ) -> LoadedDocument:
        """
        Clean all pages while preserving document metadata
        and page references.
        """

        cleaned_pages = [
            DocumentPage(
                page_number=page.page_number,
                text=self.clean_text(page.text),
            )
            for page in document.pages
        ]

        return LoadedDocument(
            source_format=document.source_format,
            page_count=document.page_count,
            pages=cleaned_pages,
        )
