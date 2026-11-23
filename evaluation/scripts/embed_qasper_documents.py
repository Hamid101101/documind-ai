
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from ingestion.chunking.base import DocumentChunk
from ingestion.embeddings.config import EmbeddingConfig
from ingestion.embeddings.embedding_generator import EmbeddingGenerator


DATA_DIR = ROOT / "evaluation" / "datasets" / "processed"
DOCUMENTS_DIR = DATA_DIR / "documents"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth" / "retrieval_ground_truth.jsonl"

OUTPUT_DIR = ROOT / "evaluation" / "reports"
OUTPUT_PATH = OUTPUT_DIR / "qasper_document_embeddings.npz"
METADATA_PATH = OUTPUT_DIR / "qasper_document_embeddings.json"

WINDOW_SIZE = 450
WINDOW_OVERLAP = 100
WINDOW_STRIDE = WINDOW_SIZE - WINDOW_OVERLAP

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)


def load_ground_truth() -> list[dict]:
    with GROUND_TRUTH_PATH.open(encoding="utf-8") as file:
        return [
            json.loads(line)
            for line in file
            if line.strip()
        ]


def build_windows(
    paper_id: str,
    text: str,
    tokenizer,
    max_seq_length: int,
) -> list[DocumentChunk]:
    """Split a paper into overlapping, model-safe text windows."""

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
        truncation=False,
    )

    offsets = [
        (start, end)
        for start, end in encoded["offset_mapping"]
        if end > start
    ]

    windows = []
    start = 0
    window_index = 0

    while start < len(offsets):
        end = min(start + WINDOW_SIZE, len(offsets))

        char_start = offsets[start][0]
        char_end = offsets[end - 1][1]

        window_text = text[char_start:char_end].strip()

        # Verify the prefixed text fits the model's actual input limit.
        while True:
            check = tokenizer(
                f"passage: {window_text}",
                add_special_tokens=True,
                truncation=False,
            )

            if len(check["input_ids"]) <= max_seq_length:
                break

            end -= 1

            if end <= start:
                raise ValueError(
                    f"Cannot create a valid window for paper {paper_id}."
                )

            char_end = offsets[end - 1][1]
            window_text = text[char_start:char_end].strip()

        windows.append(
            DocumentChunk(
                chunk_index=window_index,
                content=window_text,
                page_start=None,
                page_end=None,
                token_count=end - start,
                metadata={
                    "paper_id": paper_id,
                    "window_index": window_index,
                },
            )
        )

        window_index += 1

        if end == len(offsets):
            break

        start += WINDOW_STRIDE

    return windows


def main() -> None:
    config = EmbeddingConfig()

    if config.model_name != "intfloat/multilingual-e5-base":
        raise ValueError(
            f"Expected E5-base, got {config.model_name}"
        )

    if config.expected_dimension != 768:
        raise ValueError(
            f"Expected 768 dimensions, got {config.expected_dimension}"
        )

    logger.info("Loading embedding model: %s", config.model_name)
    embedder = EmbeddingGenerator(config)

    # Access the model once, then reuse its tokenizer.
    model = embedder.model
    tokenizer = model.tokenizer
    max_seq_length = model.max_seq_length

    if WINDOW_SIZE + len(
        tokenizer.encode("passage: ", add_special_tokens=False)
    ) >= max_seq_length:
        raise ValueError("Window size leaves insufficient model input space.")

    ground_truth = load_ground_truth()
    paper_ids = sorted({record["paper_id"] for record in ground_truth})

    all_windows: list[DocumentChunk] = []
    window_paper_ids: list[str] = []
    window_weights: list[int] = []
    paper_window_counts: dict[str, int] = {}

    logger.info("Preparing %d QASPER papers", len(paper_ids))

    for paper_id in paper_ids:
        path = DOCUMENTS_DIR / f"{paper_id}.txt"

        if not path.is_file():
            raise FileNotFoundError(path)

        text = path.read_text(encoding="utf-8").strip()

        if not text:
            raise ValueError(f"Empty paper: {paper_id}")

        windows = build_windows(
            paper_id=paper_id,
            text=text,
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
        )

        paper_window_counts[paper_id] = len(windows)

        all_windows.extend(windows)
        window_paper_ids.extend([paper_id] * len(windows))
        window_weights.extend([window.token_count for window in windows])

        logger.info(
            "%s: %d tokens -> %d windows",
            paper_id,
            sum(window.token_count for window in windows),
            len(windows),
        )

    logger.info("Total windows to embed: %d", len(all_windows))

    # Existing embedder applies the configured passage prefix,
    # normalization, batch size, and dimension validation.
    window_embeddings = np.asarray(
        embedder.embed_chunks(all_windows),
        dtype=np.float32,
    )

    if window_embeddings.shape != (len(all_windows), 768):
        raise ValueError(
            f"Unexpected embedding shape: {window_embeddings.shape}"
        )

    # Aggregate window embeddings into one vector per paper.
    paper_embeddings = []

    for paper_id in paper_ids:
        indices = [
            i
            for i, candidate in enumerate(window_paper_ids)
            if candidate == paper_id
        ]

        vectors = window_embeddings[indices]
        weights = np.asarray(
            [window_weights[i] for i in indices],
            dtype=np.float32,
        )

        paper_vector = np.average(
            vectors,
            axis=0,
            weights=weights,
        )

        norm = np.linalg.norm(paper_vector)

        if norm == 0 or not np.isfinite(norm):
            raise ValueError(
                f"Invalid aggregated embedding for {paper_id}"
            )

        paper_embeddings.append(paper_vector / norm)

    paper_matrix = np.asarray(paper_embeddings, dtype=np.float32)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    np.savez_compressed(
        OUTPUT_PATH,
        paper_ids=np.asarray(paper_ids),
        embeddings=paper_matrix,
    )

    metadata = {
        "model": config.model_name,
        "dimension": config.expected_dimension,
        "max_seq_length": max_seq_length,
        "window_size": WINDOW_SIZE,
        "window_overlap": WINDOW_OVERLAP,
        "aggregation": "token_count_weighted_mean_then_l2_normalize",
        "paper_count": len(paper_ids),
        "window_count": len(all_windows),
        "paper_window_counts": paper_window_counts,
    }

    METADATA_PATH.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    logger.info("Embedding generation completed.")
    logger.info("Paper embedding shape: %s", paper_matrix.shape)
    logger.info("Saved embeddings: %s", OUTPUT_PATH)
    logger.info("Saved metadata: %s", METADATA_PATH)


if __name__ == "__main__":
    main()
