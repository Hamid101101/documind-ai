
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from ingestion.chunking.base import DocumentChunk


class SciFactLoader:
    """Load the BEIR SciFact dataset for document-level retrieval evaluation."""

    def __init__(self, dataset_dir: str | Path | None = None):
        project_root = Path(__file__).resolve().parents[2]

        self.dataset_dir = Path(
            dataset_dir
            or project_root / "evaluation/datasets/raw/hf_scifact"
        )

        self.corpus_path = (
            self.dataset_dir / "corpus/corpus/corpus-00000-of-00001.parquet"
        )
        self.queries_path = (
            self.dataset_dir / "corpus/queries/queries-00000-of-00001.parquet"
        )
        self.qrels_path = self.dataset_dir / "qrels/test.tsv"

    def load_corpus(self) -> list[dict[str, Any]]:
        """Load document IDs, titles and abstracts."""

        if not self.corpus_path.is_file():
            raise FileNotFoundError(self.corpus_path)

        dataframe = pq.read_table(self.corpus_path).to_pandas()

        documents = []

        for _, row in dataframe.iterrows():
            document_id = str(row["_id"])
            title = str(row["title"]).strip()
            abstract = str(row["text"]).strip()

            if not title or not abstract:
                raise ValueError(f"Empty document content: {document_id}")

            documents.append(
                {
                    "id": document_id,
                    "title": title,
                    "text": abstract,
                    "content": f"{title}\n\n{abstract}",
                }
            )

        return documents

    def load_queries(self) -> dict[str, str]:
        """Load only queries referenced by the official test qrels."""

        if not self.queries_path.is_file():
            raise FileNotFoundError(self.queries_path)

        if not self.qrels_path.is_file():
            raise FileNotFoundError(self.qrels_path)

        queries_df = pq.read_table(self.queries_path).to_pandas()
        qrels_df = pd.read_csv(self.qrels_path, sep="\t")

        evaluation_ids = set(qrels_df["query-id"].astype(str))

        queries = {
            str(row["_id"]): str(row["text"]).strip()
            for _, row in queries_df.iterrows()
            if str(row["_id"]) in evaluation_ids
        }

        if set(queries) != evaluation_ids:
            missing = evaluation_ids - set(queries)
            raise ValueError(f"Missing evaluation queries: {sorted(missing)}")

        return queries

    def load_qrels(self) -> dict[str, set[str]]:
        """Load binary relevance judgments from the official test qrels."""

        if not self.qrels_path.is_file():
            raise FileNotFoundError(self.qrels_path)

        dataframe = pd.read_csv(self.qrels_path, sep="\t")

        qrels: dict[str, set[str]] = {}

        for _, row in dataframe.iterrows():
            if row["score"] > 0:
                query_id = str(row["query-id"])
                document_id = str(row["corpus-id"])

                qrels.setdefault(query_id, set()).add(document_id)

        return qrels

    @staticmethod
    def to_document_chunks(
        documents: list[dict[str, Any]],
    ) -> list[DocumentChunk]:
        """Convert complete abstracts into DocumentChunk objects."""

        return [
            DocumentChunk(
                chunk_index=index,
                content=document["content"],
                page_start=None,
                page_end=None,
                token_count=0,
                metadata={
                    "source": "scifact",
                    "document_id": document["id"],
                    "title": document["title"],
                },
            )
            for index, document in enumerate(documents)
        ]
