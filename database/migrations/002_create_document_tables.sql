
-- ============================================================
-- Migration: 002_create_document_tables
-- Description: Create document metadata and vector chunk tables
-- ============================================================


-- ============================================================
-- 1. DOCUMENTS
-- Stores uploaded file metadata and MinIO object references.
-- ============================================================

CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Original file information
    filename VARCHAR(255) NOT NULL,
    mime_type VARCHAR(150) NOT NULL,
    file_extension VARCHAR(20),
    file_size_bytes BIGINT NOT NULL CHECK (file_size_bytes > 0),
    file_checksum VARCHAR(64),

    -- MinIO storage information
    storage_bucket VARCHAR(255) NOT NULL,
    storage_object_key TEXT NOT NULL,

    -- Processing information
    status VARCHAR(20) NOT NULL DEFAULT 'uploaded'
        CHECK (
            status IN (
                'uploaded',
                'queued',
                'processing',
                'completed',
                'failed'
            )
        ),

    page_count INTEGER CHECK (page_count >= 0),
    chunk_count INTEGER NOT NULL DEFAULT 0 CHECK (chunk_count >= 0),

    -- Extracted document metadata (title, author, etc.)
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Embedding configuration
    embedding_model VARCHAR(255),

    -- Error information
    error_message TEXT,

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processing_started_at TIMESTAMPTZ,
    processing_completed_at TIMESTAMPTZ,

    -- Prevent duplicate storage references
    CONSTRAINT unique_storage_object
        UNIQUE (storage_bucket, storage_object_key)
);


-- ============================================================
-- 2. DOCUMENT CHUNKS
-- Stores extracted text and corresponding embeddings.
-- ============================================================

CREATE TABLE document_chunks (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,

    document_id UUID NOT NULL
        REFERENCES documents(id)
        ON DELETE CASCADE,

    -- Chunk information
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 0),
    content TEXT NOT NULL CHECK (LENGTH(TRIM(content)) > 0),

    -- Original document location
    page_start INTEGER CHECK (page_start >= 1),
    page_end INTEGER CHECK (page_end >= 1),

    -- Optional chunk metadata
    token_count INTEGER CHECK (token_count >= 0),
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- 384 dimensions: intfloat/multilingual-e5-small
    embedding vector(384),

    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    CONSTRAINT unique_document_chunk
        UNIQUE (document_id, chunk_index),

    CONSTRAINT valid_page_range
        CHECK (
            page_start IS NULL
            OR page_end IS NULL
            OR page_end >= page_start
        )
);


-- ============================================================
-- 3. INDEXES
-- ============================================================

-- Find documents by processing status
CREATE INDEX idx_documents_status
    ON documents(status);

-- List documents by creation date
CREATE INDEX idx_documents_created_at
    ON documents(created_at DESC);

-- Retrieve chunks belonging to a document
CREATE INDEX idx_document_chunks_document_id
    ON document_chunks(document_id);

-- Vector similarity search using cosine distance
CREATE INDEX idx_document_chunks_embedding
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WHERE embedding IS NOT NULL;
