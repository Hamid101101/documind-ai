"""Upload prepared QASPER documents through the DocuMind API."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DOCUMENTS_DIR = (
    PROJECT_ROOT / "evaluation" / "datasets" / "processed" / "documents"
)

MAPPING_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "datasets"
    / "processed"
    / "qasper_document_mapping.json"
)

API_BASE_URL = os.getenv(
    "DOCUMIND_API_URL",
    "http://localhost:8000/api/v1",
).rstrip("/")

TERMINAL_STATUSES = {"completed", "failed"}


def load_mapping() -> dict[str, dict[str, Any]]:
    """Load the existing QASPER-to-DocuMind ID mapping."""
    if not MAPPING_PATH.exists():
        return {}

    with MAPPING_PATH.open("r", encoding="utf-8") as file:
        mapping = json.load(file)

    if not isinstance(mapping, dict):
        raise ValueError("The mapping file must contain a JSON object.")

    return mapping


def save_mapping(mapping: dict[str, dict[str, Any]]) -> None:
    """Save the mapping atomically."""
    MAPPING_PATH.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = MAPPING_PATH.with_suffix(".tmp")

    with temporary_path.open("w", encoding="utf-8") as file:
        json.dump(mapping, file, indent=2, ensure_ascii=False)
        file.write("\n")

    temporary_path.replace(MAPPING_PATH)


def fetch_existing_documents(
    session: requests.Session,
) -> dict[str, dict[str, Any]]:
    """Find previously uploaded QASPER documents using their metadata."""
    existing = {}
    page = 1

    while True:
        response = session.get(
            f"{API_BASE_URL}/documents/",
            params={"page": page, "page_size": 100},
            timeout=30,
        )
        response.raise_for_status()

        payload = response.json()

        for document in payload.get("documents", []):
            metadata = document.get("metadata") or {}

            if metadata.get("source") == "qasper":
                paper_id = metadata.get("qasper_paper_id")

                if paper_id:
                    existing[str(paper_id)] = document

        total = payload.get("total", 0)
        page_size = payload.get("page_size", 100)

        if page * page_size >= total:
            break

        page += 1

    return existing


def get_document(
    session: requests.Session,
    document_id: str,
) -> dict[str, Any] | None:
    """Retrieve a document by its DocuMind UUID."""
    response = session.get(
        f"{API_BASE_URL}/documents/{document_id}",
        timeout=30,
    )

    if response.status_code == 404:
        return None

    response.raise_for_status()
    return response.json()


def upload_document(
    session: requests.Session,
    file_path: Path,
    paper_id: str,
) -> dict[str, Any]:
    """Upload one QASPER document through the FastAPI endpoint."""
    metadata = {
        "source": "qasper",
        "qasper_paper_id": paper_id,
        "dataset_split": "validation",
        "evaluation_dataset": True,
    }

    with file_path.open("rb") as document_file:
        response = session.post(
            f"{API_BASE_URL}/documents/",
            files={
                "file": (
                    file_path.name,
                    document_file,
                    "text/plain",
                )
            },
            data={
                "metadata": json.dumps(metadata),
            },
            timeout=120,
        )

    response.raise_for_status()
    return response.json()


def wait_for_processing(
    session: requests.Session,
    document_id: str,
    poll_interval: int,
    timeout_seconds: int,
) -> dict[str, Any]:
    """Wait until document processing completes or fails."""
    deadline = time.monotonic() + timeout_seconds

    while time.monotonic() < deadline:
        document = get_document(session, document_id)

        if document is None:
            raise RuntimeError(
                f"Document {document_id} was not found."
            )

        status = str(document.get("status", "")).lower()

        if status in TERMINAL_STATUSES:
            return document

        time.sleep(poll_interval)

    raise TimeoutError(
        f"Timed out waiting for document {document_id}."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest QASPER documents through DocuMind."
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of files to process.",
    )

    parser.add_argument(
        "--poll-interval",
        type=int,
        default=2,
        help="Seconds between status checks.",
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=600,
        help="Maximum processing time per document in seconds.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.limit is not None and args.limit < 1:
        print("--limit must be greater than zero.")
        return 2

    if args.poll_interval < 1 or args.timeout < 1:
        print("--poll-interval and --timeout must be greater than zero.")
        return 2

    if not DOCUMENTS_DIR.is_dir():
        print(f"Documents directory not found: {DOCUMENTS_DIR}")
        return 2

    files = sorted(DOCUMENTS_DIR.glob("*.txt"))

    if args.limit is not None:
        files = files[:args.limit]

    if not files:
        print("No TXT documents found.")
        return 0

    mapping = load_mapping()
    session = requests.Session()

    try:
        existing_documents = fetch_existing_documents(session)
    except requests.RequestException as exc:
        print(f"Could not retrieve existing documents: {exc}")
        return 1

    completed = 0
    failed = 0
    reused = 0

    print(f"API: {API_BASE_URL}")
    print(f"Documents selected: {len(files)}")

    for index, file_path in enumerate(files, start=1):
        paper_id = file_path.stem
        document_id = mapping.get(paper_id, {}).get("document_id")

        try:
            document = None

            # First try the previously saved DocuMind UUID.
            if document_id:
                document = get_document(session, document_id)

            # Recover existing uploads if the mapping was not saved.
            if document is None:
                document = existing_documents.get(paper_id)

            if document is not None:
                document_id = str(document["id"])
                reused += 1

                print(
                    f"[{index}/{len(files)}] Reusing "
                    f"{file_path.name} ({document.get('status')})"
                )

            else:
                print(
                    f"[{index}/{len(files)}] Uploading "
                    f"{file_path.name}..."
                )

                document = upload_document(
                    session,
                    file_path,
                    paper_id,
                )

                document_id = str(document["id"])

            # Persist the UUID immediately after upload/recovery.
            mapping[paper_id] = {
                "document_id": document_id,
                "filename": file_path.name,
                "status": document.get("status", "unknown"),
            }
            save_mapping(mapping)

            status = str(document.get("status", "")).lower()

            if status not in TERMINAL_STATUSES:
                document = wait_for_processing(
                    session,
                    document_id,
                    args.poll_interval,
                    args.timeout,
                )

                status = str(document.get("status", "")).lower()

            mapping[paper_id]["status"] = status

            if document.get("error_message"):
                mapping[paper_id]["error_message"] = (
                    document["error_message"]
                )
            else:
                mapping[paper_id].pop("error_message", None)

            save_mapping(mapping)

            if status == "completed":
                completed += 1

                print(
                    f"    COMPLETED | "
                    f"Pages: {document.get('page_count')} | "
                    f"Chunks: {document.get('chunk_count')}"
                )

            else:
                failed += 1

                print(
                    f"    FAILED | "
                    f"{document.get('error_message') or 'No error details'}"
                )

        except TimeoutError as exc:
            failed += 1

            mapping[paper_id] = {
                "document_id": document_id,
                "filename": file_path.name,
                "status": "timeout",
            }
            save_mapping(mapping)

            print(f"    TIMEOUT | {exc}")

        except (
            requests.RequestException,
            KeyError,
            ValueError,
            RuntimeError,
        ) as exc:
            failed += 1
            print(f"    ERROR | {exc}")

            if document_id:
                mapping[paper_id] = {
                    "document_id": document_id,
                    "filename": file_path.name,
                    "status": "unknown",
                }
                save_mapping(mapping)

    session.close()

    print("\nIngestion summary")
    print(f"Completed: {completed}")
    print(f"Failed:    {failed}")
    print(f"Reused:    {reused}")
    print(f"Mapping:   {MAPPING_PATH}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
