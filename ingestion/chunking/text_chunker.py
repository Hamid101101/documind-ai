
import logging
from functools import lru_cache
from typing import Protocol

from transformers import AutoTokenizer, PreTrainedTokenizerBase

from ingestion.chunking.base import DocumentChunk
from ingestion.chunking.config import ChunkingConfig
from ingestion.loaders.base import LoadedDocument

import os
from dotenv import load_dotenv


logger = logging.getLogger(__name__)

load_dotenv()

DEFAULT_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-small",
)

class TokenizerProtocol(Protocol):
    def encode(
        self,
        text: str,
        add_special_tokens: bool = False,
    ) -> list[int]: ...

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = False,
    ) -> str: ...


@lru_cache(maxsize=2)
def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """Load and cache the tokenizer."""
    logger.info("Loading tokenizer: %s", model_name)
    return AutoTokenizer.from_pretrained(model_name)


class TextChunker:
    def __init__(
        self,
        config: ChunkingConfig | None = None,
        tokenizer: TokenizerProtocol | None = None,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        self.config = config or ChunkingConfig()
        self.tokenizer = tokenizer or load_tokenizer(model_name)

    def _tokenize(self, text: str) -> list[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=False,
        )

    def _count_tokens(self, text: str) -> int:
        return len(self._tokenize(text))

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

    def _split_by_separator(
        self,
        text: str,
        separator: str,
    ) -> list[str]:
        if not separator:
            return list(text)

        parts = text.split(separator)

        # Keep the separator attached to the preceding part.
        result = []

        for index, part in enumerate(parts):
            if index < len(parts) - 1:
                part += separator

            if part:
                result.append(part)

        return result

    def _recursive_split(
        self,
        text: str,
        separators: tuple[str, ...],
    ) -> list[str]:
        if self._count_tokens(text) <= self.config.chunk_size:
            return [text] if text.strip() else []

        if not separators:
            # Last resort: split by token IDs.
            token_ids = self._tokenize(text)
            return [
                self._decode(token_ids[i:i + self.config.chunk_size])
                for i in range(
                    0,
                    len(token_ids),
                    self.config.chunk_size,
                )
            ]

        separator = separators[0]
        remaining_separators = separators[1:]

        parts = self._split_by_separator(text, separator)

        if len(parts) <= 1:
            return self._recursive_split(
                text,
                remaining_separators,
            )

        result = []
        current_parts = []

        for part in parts:
            candidate = "".join(current_parts) + part

            if self._count_tokens(candidate) <= self.config.chunk_size:
                current_parts.append(part)
                continue

            if current_parts:
                result.append("".join(current_parts))
                current_parts = []

            if self._count_tokens(part) <= self.config.chunk_size:
                current_parts.append(part)
            else:
                result.extend(
                    self._recursive_split(
                        part,
                        remaining_separators,
                    )
                )

        if current_parts:
            result.append("".join(current_parts))

        return result



    def _create_overlapping_chunks(self, text: str) -> list[str]:
        segments = self._recursive_split(text, self.config.separators)

        chunks: list[str] = []
        current_tokens: list[int] = []

        for segment in segments:
            remaining_tokens = self._tokenize(segment)

            while remaining_tokens:
                available = self.config.chunk_size - len(current_tokens)
                take = min(available, len(remaining_tokens))

                current_tokens.extend(remaining_tokens[:take])
                remaining_tokens = remaining_tokens[take:]

                if len(current_tokens) == self.config.chunk_size:
                    chunks.append(self._decode(current_tokens))

                    overlap_size = self.config.chunk_overlap
                    current_tokens = (
                        current_tokens[-overlap_size:]
                        if overlap_size > 0
                        else []
                    )

        # Do not emit an overlap-only chunk.
        if current_tokens and (
            not chunks or len(current_tokens) > self.config.chunk_overlap
        ):
            chunks.append(self._decode(current_tokens))

        return [chunk.strip() for chunk in chunks if chunk.strip()]


    def chunk_document(
        self,
        document: LoadedDocument,
    ) -> list[DocumentChunk]:
        """
        Convert a loaded document into token-limited chunks.

        Each source page is initially processed independently so
        that page references remain accurate.
        """

        chunks: list[DocumentChunk] = []

        for page in document.pages:
            if not page.text.strip():
                continue

            page_chunks = self._create_overlapping_chunks(page.text)

            for content in page_chunks:
                chunks.append(
                    DocumentChunk(
                        chunk_index=len(chunks),
                        content=content,
                        page_start=page.page_number,
                        page_end=page.page_number,
                        token_count=self._count_tokens(content),
                    )
                )

        return chunks
