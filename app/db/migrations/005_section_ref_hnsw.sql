-- ============================================================
-- 005_section_ref_hnsw.sql
--
-- Three coupled changes that together raise retrieval quality
-- and the precision of citations:
--
--   A) section_ref column on document_chunks
--      The chunker (after this migration) detects legal markers
--      (Articulo X, Seccion Y, Capitulo Z, Anexo W, Apendice V,
--      Clausula N) and tags each chunk with the most recent
--      marker. This lets the LLM cite at clause granularity
--      ("ISO 42001, Articulo 9 establece...") instead of just
--      at document granularity ("[Fuente: ISO42001.pdf]").
--
--   B) HNSW index replacing IVFFlat
--      HNSW gives better recall at comparable latency than
--      IVFFlat, especially for queries that fall between cluster
--      centroids. We drop the old idx_document_chunks_embedding
--      and create idx_document_chunks_embedding_hnsw with
--      m = 16 and ef_construction = 64 (pgvector's recommended
--      defaults for general use).
--
--   C) Updated match_chunks / match_global_chunks
--      Both RPCs now return section_ref in addition to the
--      existing columns. The query-time tuning parameter
--      changes from `ivfflat.probes` to `hnsw.ef_search`.
--
-- IMPORTANT: PostgreSQL does not allow CREATE OR REPLACE on a
-- function whose argument list changes (it would create an
-- overload instead). The new return shape adds a column, which
-- counts as a change. We DROP both functions explicitly first
-- so the recreate is unambiguous.
--
-- HOW TO RUN:
--   Supabase Dashboard -> SQL Editor -> paste and Run.
-- ============================================================


-- ============================================================
-- A) Add section_ref column
-- ============================================================
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS section_ref TEXT;

COMMENT ON COLUMN document_chunks.section_ref IS
  'Most recent legal section marker visible in this chunk (e.g. "Articulo 9", "Seccion 6.1.2"). NULL when the parent document does not use formal section markers.';


-- ============================================================
-- B) Replace IVFFlat with HNSW
--
-- DROP INDEX IF EXISTS handles two cases:
--   1. The original index from 001_initial.sql exists (drop it).
--   2. The migration is being re-run (already dropped, no-op).
--
-- HNSW build can take a few minutes on large corpora. The CREATE
-- INDEX is non-concurrent because we want a clean swap during a
-- maintenance window. If you need zero downtime, switch to
-- CREATE INDEX CONCURRENTLY (cannot be inside a transaction).
-- ============================================================
DROP INDEX IF EXISTS idx_document_chunks_embedding;

CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding_hnsw
    ON document_chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

COMMENT ON INDEX idx_document_chunks_embedding_hnsw IS
  'HNSW approximate nearest-neighbor index for cosine similarity. Tune query recall via SET LOCAL hnsw.ef_search = N inside the RPC.';


-- ============================================================
-- C1) Drop legacy match_chunks signatures
--
-- We drop every signature that may exist from prior migrations
-- (001 / 003 / 004) so the recreate below is unambiguous.
-- IF EXISTS makes each line a no-op when that signature is gone.
-- ============================================================
DROP FUNCTION IF EXISTS match_chunks(vector, TEXT, INT);
DROP FUNCTION IF EXISTS match_chunks(vector, TEXT, INT, TEXT[]);

DROP FUNCTION IF EXISTS match_global_chunks(vector, INT, TEXT);
DROP FUNCTION IF EXISTS match_global_chunks(vector, INT, TEXT, TEXT[]);


-- ============================================================
-- C2) Recreate match_chunks with section_ref + HNSW tuning
-- ============================================================
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding   vector(1536),
    match_tenant_id   TEXT,
    match_count       INT DEFAULT 5,
    match_frameworks  TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    id              UUID,
    document_id     UUID,
    tenant_id       TEXT,
    content         TEXT,
    chunk_index     INTEGER,
    section_ref     TEXT,
    similarity      FLOAT
)
LANGUAGE plpgsql
VOLATILE
AS $$
BEGIN
    -- HNSW recall tuning. 100 mirrors the recall behaviour we had
    -- with ivfflat.probes = 100. Bump to 200 if QA shows recall drop.
    SET LOCAL hnsw.ef_search = 100;

    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.tenant_id,
        dc.content,
        dc.chunk_index,
        dc.section_ref,
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    WHERE
        dc.tenant_id = match_tenant_id
        AND dc.embedding IS NOT NULL
        AND (
            -- Framework filter: NULL / empty array -> no restriction.
            -- Otherwise include chunks whose framework is in the list
            -- OR chunks with no framework (framework-agnostic docs).
            match_frameworks IS NULL
            OR cardinality(match_frameworks) = 0
            OR dc.framework IS NULL
            OR dc.framework = ANY(match_frameworks)
        )
    ORDER BY
        dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_chunks IS
  'Vector similarity search for RAG retrieval. Returns section_ref alongside content for clause-level citations. Sets hnsw.ef_search=100.';


-- ============================================================
-- C3) Recreate match_global_chunks with section_ref + HNSW tuning
-- ============================================================
CREATE OR REPLACE FUNCTION match_global_chunks(
    query_embedding   vector(1536),
    match_count       INT DEFAULT 3,
    match_country     TEXT DEFAULT NULL,
    match_frameworks  TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    id              UUID,
    document_id     UUID,
    tenant_id       TEXT,
    content         TEXT,
    chunk_index     INTEGER,
    section_ref     TEXT,
    similarity      FLOAT
)
LANGUAGE plpgsql
VOLATILE
AS $$
BEGIN
    SET LOCAL hnsw.ef_search = 100;

    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.tenant_id,
        dc.content,
        dc.chunk_index,
        dc.section_ref,
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
        AND (
            match_frameworks IS NULL
            OR cardinality(match_frameworks) = 0
            OR dc.framework IS NULL
            OR dc.framework = ANY(match_frameworks)
        )
    ORDER BY
        dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_global_chunks IS
  'Vector similarity search over the global library. Returns section_ref alongside content. Sets hnsw.ef_search=100.';
