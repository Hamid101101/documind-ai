
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_DIR = PROJECT_ROOT / "evaluation/datasets/raw/hf_scifact"

corpus_path = DATASET_DIR / "corpus/corpus/corpus-00000-of-00001.parquet"
queries_path = DATASET_DIR / "corpus/queries/queries-00000-of-00001.parquet"
qrels_path = DATASET_DIR / "qrels/test.tsv"

# Inspect Parquet schemas
print("\n========== CORPUS SCHEMA ==========")
corpus_table = pq.read_table(corpus_path)
print(corpus_table.schema)
print(f"Rows: {corpus_table.num_rows}")

print("\n========== QUERY SCHEMA ==========")
queries_table = pq.read_table(queries_path)
print(queries_table.schema)
print(f"Rows: {queries_table.num_rows}")

# Inspect sample records
corpus_df = corpus_table.to_pandas()
queries_df = queries_table.to_pandas()
qrels_df = pd.read_csv(qrels_path, sep="\t")

print("\n========== CORPUS SAMPLE ==========")
print(corpus_df.head(2).to_string(index=False))

print("\n========== QUERY SAMPLE ==========")
print(queries_df.head(3).to_string(index=False))

print("\n========== QRELS SCHEMA ==========")
print(qrels_df.dtypes)
print(f"Rows: {len(qrels_df)}")

print("\n========== QRELS SAMPLE ==========")
print(qrels_df.head(5).to_string(index=False))

# Basic validation
print("\n========== VALIDATION ==========")
print(f"Corpus documents: {len(corpus_df)}")
print(f"Queries: {len(queries_df)}")
print(f"Qrels: {len(qrels_df)}")
print(f"Unique query IDs in qrels: {qrels_df['query-id'].nunique()}")
print(f"Unique document IDs in qrels: {qrels_df['corpus-id'].nunique()}")
print(f"Relevance values: {sorted(qrels_df['score'].unique())}")

print("\nInspection completed.")
