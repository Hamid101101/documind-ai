
from typing import Any

from ingestion.embeddings.embedding_generator import EmbeddingGenerator
from retrieval.config import RetrievalConfig


class ContextBuilder:
    """Build an LLM-ready context from retrieved document chunks."""

    def __init__(
            self,
        embedding_generator: EmbeddingGenerator | None = None,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.embedding_generator = (
            embedding_generator or EmbeddingGenerator()
        )
        self.config = config or RetrievalConfig()

    def _count_tokens(self, text: str) -> int:
        """Count tokens using the embedding model's tokenizer."""

        tokenizer = self.embedding_generator.model.tokenizer

        return len(
            tokenizer.encode(
                text,
                add_special_tokens=False,
            )
        )

    def build(
        self,
        results: list[dict[str, Any]],
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Build formatted context and preserve its source references."""

        token_budget = (
            max_tokens
            if max_tokens is not None
            else self.config.max_context_tokens
        )

        if token_budget <= 0:
            raise ValueError("max_tokens must be greater than zero.")

        context_parts = []
        sources = []
        total_tokens = 0

        for result in results:
            page_start = result.get("page_start")
            page_end = result.get("page_end")

            if page_start is None:
                page_reference = "Page unknown"
            elif page_end is None or page_end == page_start:
                page_reference = f"Page {page_start}"
            else:
                page_reference = f"Pages {page_start}-{page_end}"

            source_index = len(sources) + 1
            filename = result.get("document_filename", "Unknown document")

            formatted_chunk = (
                f"[Source {source_index}: {filename}, {page_reference}]\n"
                f"{result['content']}"
            )

            chunk_tokens = self._count_tokens(formatted_chunk)

            if total_tokens + chunk_tokens > token_budget:
                break

            context_parts.append(formatted_chunk)
            total_tokens += chunk_tokens

            sources.append(
                {
                    "source_index": source_index,
                    "chunk_id": result["id"],
                    "document_id": result["document_id"],
                    "document_filename": filename,
                    "chunk_index": result["chunk_index"],
                    "page_start": page_start,
                    "page_end": page_end,
                    "similarity": result["similarity"],
                }
            )

        return {
            "context": "\n\n".join(context_parts),
            "sources": sources,
            "total_tokens": total_tokens,
            "included_chunks": len(sources),
        }
