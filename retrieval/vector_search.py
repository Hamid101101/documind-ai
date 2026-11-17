
from typing import Any

from database.repositories.chunk_repository import ChunkRepository
from ingestion.embeddings.embedding_generator import EmbeddingGenerator
from retrieval.config import RetrievalConfig
from retrieval.filters import RetrievalFilters
from retrieval.context_builder import ContextBuilder


class VectorSearch:
    """Orchestrate query embedding and vector similarity retrieval."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator | None = None,
        chunk_repository: ChunkRepository | None = None,
        config: RetrievalConfig | None = None,
        context_builder: ContextBuilder | None = None,
    ) -> None:
        self.embedding_generator = embedding_generator or EmbeddingGenerator()
        self.chunk_repository = chunk_repository or ChunkRepository()
        self.config = config or RetrievalConfig()
        self.context_builder = context_builder or ContextBuilder(
            embedding_generator=self.embedding_generator,
            config=self.config,
        )

    def search(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search for relevant chunks using a natural-language query."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        effective_top_k = top_k if top_k is not None else self.config.top_k

        if effective_top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if effective_top_k > self.config.max_top_k:
            raise ValueError(
                f"top_k cannot exceed {self.config.max_top_k}."
            )

        query_embedding = self.embedding_generator.embed_query(query)

        results = self.chunk_repository.search_similar(
            query_embedding=query_embedding,
            top_k=effective_top_k,
            filters=filters,
        )

        if self.config.similarity_threshold is not None:
            results = [
                result
                for result in results
                if result["similarity"] >= self.config.similarity_threshold
            ]

        return results

    def search_with_context(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Retrieve relevant chunks and assemble an LLM-ready context."""

        results = self.search(
            query=query,
            filters=filters,
            top_k=top_k,
        )

        return self.context_builder.build(
            results=results,
            max_tokens=max_tokens,
        )

    def _fuse_results(
        self,
        semantic_results: list[dict[str, Any]],
        keyword_results: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Fuse semantic and keyword rankings using Reciprocal Rank Fusion."""

        fused_results: dict[Any, dict[str, Any]] = {}
        rrf_k = self.config.rrf_k

        for results in (semantic_results, keyword_results):
            for rank, result in enumerate(results, start=1):
                chunk_id = result["id"]
                rrf_score = 1.0 / (rrf_k + rank)

                if chunk_id not in fused_results:
                    fused_results[chunk_id] = {
                        **result,
                        "rrf_score": rrf_score,
                    }
                else:
                    fused_results[chunk_id]["rrf_score"] += rrf_score

        ranked_results = sorted(
            fused_results.values(),
            key=lambda result: result["rrf_score"],
            reverse=True,
        )

        return ranked_results[:top_k]

    def hybrid_search(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve and fuse semantic and keyword search results."""

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        effective_top_k = top_k if top_k is not None else self.config.top_k

        if effective_top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        if effective_top_k > self.config.max_top_k:
            raise ValueError(
                f"top_k cannot exceed {self.config.max_top_k}."
            )

        query_embedding = self.embedding_generator.embed_query(query)

        semantic_results = self.chunk_repository.search_similar(
            query_embedding=query_embedding,
            top_k=effective_top_k,
            filters=filters,
        )

        if self.config.similarity_threshold is not None:
            semantic_results = [
                result
                for result in semantic_results
                if result["similarity"] >= self.config.similarity_threshold
            ]

        keyword_results = self.chunk_repository.search_keyword(
            query_text=query,
            top_k=effective_top_k,
            filters=filters,
        )

        return self._fuse_results(
            semantic_results=semantic_results,
            keyword_results=keyword_results,
            top_k=effective_top_k,
        )

    def hybrid_search_with_context(
        self,
        query: str,
        filters: RetrievalFilters | None = None,
        top_k: int | None = None,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Perform hybrid retrieval and assemble an LLM-ready context."""

        results = self.hybrid_search(
            query=query,
            filters=filters,
            top_k=top_k,
        )

        return self.context_builder.build(
            results=results,
            max_tokens=max_tokens,
        )