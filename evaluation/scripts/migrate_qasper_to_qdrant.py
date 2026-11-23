
import logging
import math
import sys
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

# Add project root to Python path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from infra.postgres.postgres_client import PostgreSQLClient  # noqa: E402


# Experiment configuration
COLLECTION_NAME = "qasper_chunks"
VECTOR_DIMENSION = 768
EXPECTED_CHUNKS = 1509
BATCH_SIZE = 128

QDRANT_URL = "http://localhost:6333"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)


def parse_embedding(value: str) -> list[float]:
    """Convert a PostgreSQL vector string into a Python float list."""

    if not value:
        raise ValueError("Empty embedding received.")

    value = value.strip()

    if not (value.startswith("[") and value.endswith("]")):
        raise ValueError("Invalid PostgreSQL vector representation.")

    vector = [float(x) for x in value[1:-1].split(",")]

    if len(vector) != VECTOR_DIMENSION:
        raise ValueError(
            f"Expected {VECTOR_DIMENSION} dimensions, "
            f"received {len(vector)}."
        )

    if not all(math.isfinite(x) for x in vector):
        raise ValueError("Embedding contains non-finite values.")

    norm = math.sqrt(sum(x * x for x in vector))

    if not math.isclose(norm, 1.0, rel_tol=1e-3, abs_tol=1e-3):
        raise ValueError(
            f"Expected normalized embedding, received norm={norm:.6f}."
        )

    return vector


def load_embeddings(db: PostgreSQLClient) -> list[dict[str, Any]]:
    """Load existing embeddings without modifying PostgreSQL."""

    stats = db.fetch_one(
        """
        SELECT
            COUNT(*) AS total_chunks,
            COUNT(embedding) AS embedded_chunks,
            COUNT(*) FILTER (
                WHERE embedding IS NULL
            ) AS missing_embeddings
        FROM document_chunks;
        """
    )

    logger.info("Database statistics: %s", stats)

    if stats["total_chunks"] != EXPECTED_CHUNKS:
        raise ValueError(
            f"Expected {EXPECTED_CHUNKS} chunks, "
            f"found {stats['total_chunks']}."
        )

    if stats["embedded_chunks"] != EXPECTED_CHUNKS:
        raise ValueError(
            f"Expected {EXPECTED_CHUNKS} embeddings, "
            f"found {stats['embedded_chunks']}."
        )

    rows = db.fetch_all(
        """
        SELECT
            id,
            document_id,
            chunk_index,
            content,
            page_start,
            page_end,
            token_count,
            metadata,
            embedding::text AS embedding,
            vector_dims(embedding) AS embedding_dimension
        FROM document_chunks
        WHERE embedding IS NOT NULL
        ORDER BY id;
        """
    )

    logger.info("Loaded %d chunks from PostgreSQL.", len(rows))

    for row in rows:
        if row["embedding_dimension"] != VECTOR_DIMENSION:
            raise ValueError(
                f"Chunk {row['id']} has dimension "
                f"{row['embedding_dimension']}."
            )

        row["vector"] = parse_embedding(row["embedding"])

    return rows


def create_qdrant_collection(client: QdrantClient) -> None:
    """Create a clean collection for this experiment."""

    if client.collection_exists(COLLECTION_NAME):
        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' already exists. "
            "Inspect it before deciding whether to delete and recreate it."
        )

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=VECTOR_DIMENSION,
            distance=Distance.COSINE,
        ),
    )

    logger.info("Created Qdrant collection: %s", COLLECTION_NAME)


def upload_embeddings(
    client: QdrantClient,
    rows: list[dict[str, Any]],
) -> None:
    """Upload existing PostgreSQL vectors in batches."""

    for start in range(0, len(rows), BATCH_SIZE):
        batch = rows[start:start + BATCH_SIZE]

        points = []

        for row in batch:
            point = PointStruct(
                id=int(row["id"]),
                vector=row["vector"],
                payload={
                    "chunk_id": int(row["id"]),
                    "document_id": str(row["document_id"]),
                    "chunk_index": int(row["chunk_index"]),
                    "content": row["content"],
                    "page_start": row["page_start"],
                    "page_end": row["page_end"],
                    "token_count": row["token_count"],
                    "metadata": row["metadata"] or {},
                },
            )

            points.append(point)

        client.upsert(
            collection_name=COLLECTION_NAME,
            points=points,
            wait=True,
        )

        logger.info(
            "Uploaded %d/%d points.",
            min(start + len(batch), len(rows)),
            len(rows),
        )


def verify_migration(
    client: QdrantClient,
    rows: list[dict[str, Any]],
) -> None:
    """Verify point count, IDs, payloads, and stored vectors."""

    count = client.count(
        collection_name=COLLECTION_NAME,
        exact=True,
    ).count

    if count != len(rows):
        raise RuntimeError(
            f"Point count mismatch: expected {len(rows)}, got {count}."
        )

    logger.info("Point count verified: %d", count)

    sample_ids = [int(row["id"]) for row in rows[:5]]

    points = client.retrieve(
        collection_name=COLLECTION_NAME,
        ids=sample_ids,
        with_payload=True,
        with_vectors=True,
    )

    if len(points) != len(sample_ids):
        raise RuntimeError("Could not retrieve all verification points.")

    original_by_id = {
        int(row["id"]): row
        for row in rows
    }

    for point in points:
        original = original_by_id[point.id]

        if point.payload["chunk_id"] != point.id:
            raise RuntimeError(f"ID mismatch for point {point.id}.")

        if point.payload["document_id"] != str(original["document_id"]):
            raise RuntimeError(
                f"Document ID mismatch for point {point.id}."
            )

        stored_vector = point.vector

        if not isinstance(stored_vector, list):
            raise RuntimeError(
                f"Unexpected vector format for point {point.id}."
            )

        if len(stored_vector) != VECTOR_DIMENSION:
            raise RuntimeError(
                f"Incorrect vector dimension for point {point.id}."
            )

        # Qdrant normalizes vectors for cosine similarity.
        # Compare with a small tolerance.
        for original_value, stored_value in zip(
            original["vector"], stored_vector
        ):
            if not math.isclose(
                original_value,
                stored_value,
                rel_tol=1e-4,
                abs_tol=1e-4,
            ):
                raise RuntimeError(
                    f"Vector mismatch for point {point.id}."
                )

    logger.info("Sample vector and payload verification passed.")


def main() -> None:
    logger.info("Starting QASPER migration to Qdrant.")

    db = PostgreSQLClient()
    client = QdrantClient(url=QDRANT_URL, timeout=30)

    try:
        if not db.check_connection():
            raise RuntimeError("PostgreSQL connection failed.")

        logger.info("PostgreSQL connection successful.")

        rows = load_embeddings(db)

        create_qdrant_collection(client)
        upload_embeddings(client, rows)
        verify_migration(client, rows)

        logger.info("=" * 50)
        logger.info("QASPER MIGRATION COMPLETED SUCCESSFULLY")
        logger.info("Collection: %s", COLLECTION_NAME)
        logger.info("Points: %d", len(rows))
        logger.info("Dimensions: %d", VECTOR_DIMENSION)
        logger.info("Distance: Cosine")
        logger.info("=" * 50)

    finally:
        client.close()


if __name__ == "__main__":
    main()
