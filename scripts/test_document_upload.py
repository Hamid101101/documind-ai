import argparse
import json
import os
import sys
import time
from pathlib import Path
from uuid import UUID

import requests

from database.repositories.document_repository import DocumentRepository


API_URL = os.getenv(
    "DOCUMIND_API_URL",
    "http://localhost:8000/api/v1",
)

POLL_INTERVAL = 2
MAX_WAIT_SECONDS = 600

FINAL_STATUSES = {"completed", "failed"}


def upload_document(file_path: Path) -> dict:
    """Upload a document through the FastAPI endpoint."""

    print("\n[1/3] Uploading document...")
    print(f"File: {file_path.name}")

    with file_path.open("rb") as file:
        response = requests.post(
            f"{API_URL}/documents/",
            files={
                "file": (
                    file_path.name,
                    file,
                    "application/octet-stream",
                )
            },
            data={
                "metadata": json.dumps({
                    "source": "integration_test",
                })
            },
            timeout=120,
        )

    response.raise_for_status()

    document = response.json()

    print("Upload successful!")
    print(f"Document ID: {document['id']}")
    print(f"Initial status: {document['status']}")

    return document


def track_document(document_id: UUID) -> dict:
    """Poll PostgreSQL until document processing finishes."""

    print("\n[2/3] Tracking document processing...")

    repository = DocumentRepository()

    start_time = time.monotonic()
    previous_status = None

    while True:
        document = repository.get_by_id(document_id)

        if document is None:
            raise RuntimeError(
                f"Document {document_id} no longer exists."
            )

        current_status = document["status"]

        if current_status != previous_status:
            elapsed = time.monotonic() - start_time

            print(
                f"[{elapsed:6.1f}s] "
                f"Status: {current_status.upper()}"
            )

            previous_status = current_status

        if current_status in FINAL_STATUSES:
            return document

        elapsed = time.monotonic() - start_time

        if elapsed >= MAX_WAIT_SECONDS:
            raise TimeoutError(
                f"Document processing exceeded "
                f"{MAX_WAIT_SECONDS} seconds. "
                f"Last status: {current_status}"
            )

        time.sleep(POLL_INTERVAL)


def display_result(document: dict) -> None:
    """Display the final document processing result."""

    print("\n[3/3] Processing finished")
    print("=" * 55)

    print(f"Document: {document['filename']}")
    print(f"Document ID: {document['id']}")
    print(f"Status: {document['status'].upper()}")
    print(f"Pages: {document['page_count']}")
    print(f"Chunks: {document['chunk_count']}")
    print(f"Embedding model: {document['embedding_model']}")

    if document["status"] == "failed":
        print(f"Error: {document['error_message']}")

    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="Upload and track a DocuMind AI document."
    )

    parser.add_argument(
        "file",
        type=Path,
        help="Path to PDF, DOCX or TXT document.",
    )

    args = parser.parse_args()

    file_path = args.file.resolve()

    if not file_path.is_file():
        print(f"Error: File not found: {file_path}")
        sys.exit(1)

    try:
        # 1. Upload
        response = upload_document(file_path)

        document_id = UUID(response["id"])

        # 2. Track processing
        final_document = track_document(document_id)

        # 3. Display result
        display_result(final_document)

        if final_document["status"] == "failed":
            sys.exit(1)

    except requests.RequestException as exc:
        print(f"\nAPI request failed: {exc}")
        sys.exit(1)

    except Exception as exc:
        print(f"\nTest failed: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
