
import logging
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

# Load .env before importing configuration-dependent modules.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

from database.repositories.chunk_repository import ChunkRepository
from infra.postgres.postgres_client import PostgreSQLClient
from ingestion.chunking.base import DocumentChunk
from ingestion.embeddings.config import EmbeddingConfig
from ingestion.embeddings.embedding_generator import EmbeddingGenerator


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)

EXPECTED_DOCUMENTS = 100
EXPECTED_CHUNKS = 1509
EXPECTED_DIMENSION = 768
EXPECTED_MODEL = "intfloat/multilingual-e5-base"


def serialize_embedding(embedding: np.ndarray) -> str:
    """Validate and serialize an embedding for pgvector."""

    if embedding.shape != (EXPECTED_DIMENSION,):
        raise ValueError(
            f"Expected {EXPECTED_DIMENSION} dimensions, "
            f"received {embedding.shape}."
        )

    if not np.isfinite(embedding).all():
        raise ValueError("Embedding contains non-finite values.")

    return "[" + ",".join(str(float(v)) for v in embedding) + "]"


def main() -> None:
    # 1. Validate configuration.
    config = EmbeddingConfig()

    if config.model_name != EXPECTED_MODEL:
        raise ValueError(
            f"Expected model {EXPECTED_MODEL}, "
            f"received {config.model_name}."
        )

    if config.expected_dimension != EXPECTED_DIMENSION:
        raise ValueError(
            f"Expected dimension {EXPECTED_DIMENSION}, "
            f"received {config.expected_dimension}."
        )

    embedder = EmbeddingGenerator(config)
    db = PostgreSQLClient()

    logger.info("Model: %s", config.model_name)
    logger.info("Dimensions: %d", config.expected_dimension)

    # 2. Load existing QASPER chunks.
    rows = db.fetch_all(
        """
        SELECT
            c.id,
            c.document_id,
            c.chunk_index,
            c.content,
            c.page_start,
            c.page_end,
            c.token_count,
            c.metadata
        FROM document_chunks AS c
        JOIN documents AS d
            ON d.id = c.document_id
        WHERE d.metadata->>'source' = 'qasper'
        ORDER BY c.document_id, c.chunk_index;
        """
    )

    document_ids = {row["document_id"] for row in rows}

    if len(document_ids) != EXPECTED_DOCUMENTS:
        raise RuntimeError(
            f"Expected {EXPECTED_DOCUMENTS} documents, "
            f"found {len(document_ids)}."
        )

    if len(rows) != EXPECTED_CHUNKS:
        raise RuntimeError(
            f"Expected {EXPECTED_CHUNKS} chunks, "
            f"found {len(rows)}."
        )

    logger.info(
        "Loaded %d chunks from %d QASPER documents.",
        len(rows),
        len(document_ids),
    )

    # Preserve a snapshot of chunk identities and content.
    original_chunks = [
        (row["id"], row["content"])
        for row in rows
    ]

    # 3. Build DocumentChunk objects without changing chunk content.
    chunks = [
        DocumentChunk(
            chunk_index=row["chunk_index"],
            content=row["content"],
            page_start=row["page_start"],
            page_end=row["page_end"],
            token_count=row["token_count"] or 0,
            metadata=row["metadata"] or {},
        )
        for row in rows
    ]

    # 4. Generate E5-base embeddings.
    logger.info("Generating embeddings...")

    embeddings = np.asarray(
        embedder.embed_chunks(chunks),
        dtype=np.float32,
    )

    expected_shape = (len(chunks), EXPECTED_DIMENSION)

    if embeddings.shape != expected_shape:
        raise RuntimeError(
            f"Expected embedding shape {expected_shape}, "
            f"received {embeddings.shape}."
        )

    if not np.isfinite(embeddings).all():
        raise RuntimeError("Generated embeddings contain NaN or Inf.")

    norms = np.linalg.norm(embeddings, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-5):
        raise RuntimeError("Some embeddings are not normalized.")

    logger.info(
        "Generated %d embeddings. All dimensions and norms validated.",
        len(embeddings),
    )

    # 5. Prepare database updates.
    updates = [
        (
            serialize_embedding(embedding),
            row["id"],
        )
        for row, embedding in zip(rows, embeddings)
    ]

    # 6. Update embeddings and document model metadata
    # in one database transaction.
    with db.connection() as conn:
        with conn.cursor() as cursor:

            cursor.executemany(
                """
                UPDATE document_chunks
                SET embedding = %s::vector
                WHERE id = %s;
                """,
                updates,
            )

            cursor.execute(
                """
                UPDATE documents
                SET
                    embedding_model = %s,
                    updated_at = NOW()
                WHERE metadata->>'source' = 'qasper'
                RETURNING id;
                """,
                (config.model_name,),
            )

            updated_documents = cursor.fetchall()

            if len(updated_documents) != EXPECTED_DOCUMENTS:
                raise RuntimeError(
                    "Unexpected number of updated documents."
                )

            # 7. Verify all embeddings inside the transaction.
            cursor.execute(
                """
                SELECT
                    COUNT(*) AS total_chunks,
                    COUNT(embedding) AS embedded_chunks,
                    MIN(vector_dims(embedding)) AS min_dimensions,
                    MAX(vector_dims(embedding)) AS max_dimensions
                FROM document_chunks AS c
                JOIN documents AS d
                    ON d.id = c.document_id
                WHERE d.metadata->>'source' = 'qasper';
                """
            )

            verification = cursor.fetchone()

            if verification["total_chunks"] != EXPECTED_CHUNKS:
                raise RuntimeError("Chunk count changed unexpectedly.")

            if verification["embedded_chunks"] != EXPECTED_CHUNKS:
                raise RuntimeError("Not all chunks received embeddings.")

            if (
                verification["min_dimensions"] != EXPECTED_DIMENSION
                or verification["max_dimensions"] != EXPECTED_DIMENSION
            ):
                raise RuntimeError("Unexpected stored vector dimensions.")

            # Verify chunk IDs and content were preserved.
            cursor.execute(
                """
                SELECT c.id, c.content
                FROM document_chunks AS c
                JOIN documents AS d
                    ON d.id = c.document_id
                WHERE d.metadata->>'source' = 'qasper'
                ORDER BY c.document_id, c.chunk_index;
                """
            )

            updated_chunks = cursor.fetchall()

            current_chunks = [
                (row["id"], row["content"])
                for row in updated_chunks
            ]

            if current_chunks != original_chunks:
                raise RuntimeError(
                    "Chunk IDs or content changed unexpectedly."
                )

    # Transaction committed successfully.
    logger.info("Database transaction committed successfully.")
    logger.info("Documents updated: %d", EXPECTED_DOCUMENTS)
    logger.info("Chunks embedded: %d", EXPECTED_CHUNKS)
    logger.info("Embedding dimensions: %d", EXPECTED_DIMENSION)
    logger.info("Embedding model: %s", config.model_name)
    logger.info("Chunk IDs and content preserved.")
    logger.info("QASPER re-embedding completed successfully.")


if __name__ == "__main__":
    main()
