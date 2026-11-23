
from evaluation.ingestion.load_scifact import SciFactLoader

loader = SciFactLoader()

corpus = loader.load_corpus()
queries = loader.load_queries()
qrels = loader.load_qrels()

chunks = loader.to_document_chunks(corpus)

print("========== SCIFACT LOADER TEST ==========")
print(f"Documents: {len(corpus)}")
print(f"Document chunks: {len(chunks)}")
print(f"Evaluation queries: {len(queries)}")
print(f"Queries with relevance judgments: {len(qrels)}")
print(f"Relevant document associations: {sum(map(len, qrels.values()))}")

print("\nSample document:")
print(corpus[0]["id"])
print(corpus[0]["title"])
print(corpus[0]["content"][:300])

print("\nSample query:")
query_id = next(iter(queries))
print(query_id, queries[query_id])

print("\nSample qrels:")
print(query_id, qrels[query_id])

assert len(corpus) == 5183
assert len(queries) == 300
assert len(qrels) == 300
assert len(chunks) == 5183

print("\nSCIFACT LOADER TEST PASSED")
