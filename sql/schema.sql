CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    title TEXT NOT NULL,
    page_number INTEGER,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS page_number INTEGER;

-- Migrate data created by the earlier single-user prototype into an explicit local tenant.
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS tenant_id TEXT;
UPDATE document_chunks SET tenant_id = 'local' WHERE tenant_id IS NULL;
ALTER TABLE document_chunks ALTER COLUMN tenant_id SET NOT NULL;
ALTER TABLE document_chunks
    DROP CONSTRAINT IF EXISTS document_chunks_document_id_chunk_index_key;
CREATE UNIQUE INDEX IF NOT EXISTS document_chunks_tenant_document_chunk_idx
    ON document_chunks (tenant_id, document_id, chunk_index);

DROP INDEX IF EXISTS document_chunks_document_id_idx;
CREATE INDEX IF NOT EXISTS document_chunks_tenant_document_id_idx
    ON document_chunks (tenant_id, document_id);

-- Add an ANN index only after measuring recall/latency on representative data.
