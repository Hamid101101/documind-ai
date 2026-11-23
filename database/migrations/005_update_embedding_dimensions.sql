-- Update embedding dimensions from E5-small to E5-base.
-- Existing embeddings must be regenerated.

DROP INDEX IF EXISTS idx_document_chunks_embedding;

ALTER TABLE document_chunks
    ALTER COLUMN embedding TYPE vector(768)
    USING NULL::vector(768);

-- Recreate the cosine similarity index.
CREATE INDEX idx_document_chunks_embedding
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;