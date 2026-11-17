-- ============================================================
-- Migration: 003_add_full_text_search
-- Description: Add PostgreSQL full-text search to document chunks
-- ============================================================

ALTER TABLE document_chunks
ADD COLUMN search_vector TSVECTOR
GENERATED ALWAYS AS (
    to_tsvector('simple', content)
) STORED;

CREATE INDEX idx_document_chunks_search_vector
ON document_chunks
USING GIN (search_vector);
