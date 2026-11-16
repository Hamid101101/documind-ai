
import pytest

from ingestion.chunking.config import ChunkingConfig
from ingestion.chunking.text_chunker import TextChunker
from ingestion.loaders.base import DocumentPage, LoadedDocument



class FakeTokenizer:
    """Whitespace tokenizer that preserves token content."""

    def __init__(self):
        self.vocabulary: dict[str, int] = {}
        self.reverse_vocabulary: dict[int, str] = {}

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        token_ids = []

        for token in text.split():
            if token not in self.vocabulary:
                token_id = len(self.vocabulary)
                self.vocabulary[token] = token_id
                self.reverse_vocabulary[token_id] = token

            token_ids.append(self.vocabulary[token])

        return token_ids

    def decode(
        self,
        token_ids: list[int],
        skip_special_tokens: bool = True,
    ) -> str:
        return " ".join(
            self.reverse_vocabulary[token_id]
            for token_id in token_ids
        )
