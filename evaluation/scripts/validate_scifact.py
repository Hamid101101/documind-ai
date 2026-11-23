
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "evaluation/datasets/raw/hf_scifact"

corpus_path = DATASET_DIR / "corpus/corpus/corpus-00000-of-00001.parquet"
queries_path = DATASET_DIR / "corpus/queries/queries-00000-of-00001.parquet"
qrels_path = DATASET_DIR / "qrels/test.tsv"

corpus = pq.read_table(corpus_path).to_pandas()
queries = pq.read_table(queries_path).to_pandas()
qrels = pd.read_csv(qrels_path, sep="\t")

# Normalize identifiers
corpus_ids = set(corpus["_id"].astype(str))
query_ids = set(queries["_id"].astype(str))

qrels["query-id"] = qrels["query-id"].astype(str)
qrels["corpus-id"] = qrels["corpus-id"].astype(str)

qrels_query_ids = set(qrels["query-id"])
qrels_corpus_ids = set(qrels["corpus-id"])

missing_queries = qrels_query_ids - query_ids
missing_documents = qrels_corpus_ids - corpus_ids

print("========== SCIFACT VALIDATION ==========")
print(f"Corpus documents: {len(corpus_ids)}")
print(f"Available queries: {len(query_ids)}")
print(f"Evaluation queries: {len(qrels_query_ids)}")
print(f"Relevance judgments: {len(qrels)}")
print(f"Unique relevant documents: {len(qrels_corpus_ids)}")

print(f"\nMissing queries: {len(missing_queries)}")
print(f"Missing documents: {len(missing_documents)}")

print(f"\nEmpty document titles: {corpus['title'].isna().sum()}")
print(f"Empty document abstracts: {corpus['text'].isna().sum()}")

print(f"\nEmpty query texts: {queries['text'].isna().sum()}")
print(f"Duplicate corpus IDs: {corpus['_id'].duplicated().sum()}")
print(f"Duplicate query IDs: {queries['_id'].duplicated().sum()}")

print(f"\nRelevance scores: {sorted(qrels['score'].unique())}")

if missing_queries or missing_documents:
    raise ValueError("Qrels reference missing queries or documents.")

if corpus["_id"].duplicated().any():
    raise ValueError("Duplicate corpus IDs found.")

if queries["_id"].duplicated().any():
    raise ValueError("Duplicate query IDs found.")

print("\nSCIFACT VALIDATION PASSED")
