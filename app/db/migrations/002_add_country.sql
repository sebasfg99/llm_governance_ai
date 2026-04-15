-- ============================================================
-- 002_add_country.sql — Add country field for document filtering
--
-- WHY THIS MIGRATION EXISTS:
-- Documents are organized by country (Costa Rica, Panama, etc.).
-- The country field allows filtering RAG retrieval by country:
--   - country IS NULL  → global document (applies to all countries)
--   - country = 'costa-rica' → specific to Costa Rica
--
-- Query behavior:
--   With country='costa-rica': returns chunks where
--     (country = 'costa-rica' OR country IS NULL)
--   Without country: returns chunks where country IS NULL (global only)
--
-- HOW TO RUN:
--   Supabase Dashboard → SQL Editor → paste and Run
-- ============================================================


-- Add country column to documents table
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS country TEXT;

-- Add country column to document_chunks table
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS country TEXT;

-- Index for fast country filtering on document_chunks
CREATE INDEX IF NOT EXISTS idx_document_chunks_country
    ON document_chunks (country);

-- Index for fast country filtering on documents
CREATE INDEX IF NOT EXISTS idx_documents_country
    ON documents (country);


-- ============================================================
-- Update match_global_chunks to support country filtering
--
-- match_country = NULL  → return only global chunks (country IS NULL)
-- match_country = 'x'  → return chunks where country='x' OR country IS NULL
-- ============================================================
CREATE OR REPLACE FUNCTION match_global_chunks(
    query_embedding  vector(1536),
    match_count      INT DEFAULT 3,
    match_country    TEXT DEFAULT NULL
)
RETURNS TABLE (
    id              UUID,
    document_id     UUID,
    tenant_id       TEXT,
    content         TEXT,
    chunk_index     INTEGER,
    similarity      FLOAT
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        dc.id,
        dc.document_id,
        dc.tenant_id,
        dc.content,
        dc.chunk_index,
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    WHERE
        dc.tenant_id = 'global'
        AND dc.embedding IS NOT NULL
        AND (
            CASE
                WHEN match_country IS NULL THEN dc.country IS NULL
                ELSE (dc.country = match_country OR dc.country IS NULL)
            END
        )
    ORDER BY
        dc.embedding <=> query_embedding
    LIMIT match_count;
$$;

COMMENT ON FUNCTION match_global_chunks IS 'Vector similarity search over the global shared knowledge library, with optional country filter.';
