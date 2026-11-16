
from ingestion.embeddings.embedding_generator import EmbeddingGenerator
from ingestion.chunking.base import DocumentChunk

generator = EmbeddingGenerator()

chunks = [
    DocumentChunk(
        chunk_index=0,
        content="Machine learning is a branch of artificial intelligence.",
        page_start=1,
        page_end=1,
        token_count=10,
    ),
    DocumentChunk(
        chunk_index=1,
        content="Deep learning uses neural networks to learn representations.",
        page_start=1,
        page_end=1,
        token_count=10,
    ),
]

embeddings = generator.embed_chunks(chunks)

print("Number of embeddings:", len(embeddings))
print("Embedding dimension:", len(embeddings[0]))
print("First 5 values:", embeddings[0][:5])

assert len(embeddings) == 2
assert len(embeddings[0]) == 384
assert len(embeddings[1]) == 384

print("Real embedding model test passed.")
