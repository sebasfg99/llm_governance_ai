-- ============================================================
-- 007_fix_hybrid_rpc_types.sql
--
-- HOTFIX for migration 006_rag_governance.sql
--
-- WHAT WAS BROKEN:
--   The hybrid retrieval RPCs declared `keyword_score FLOAT`
--   in their RETURNS TABLE clause, but `ts_rank_cd()` returns
--   `real` (float4) — not `double precision` (float8 = FLOAT).
--   At runtime Postgres rejected every call with:
--
--     structure of query does not match function result type
--     Returned type real does not match expected type
--     double precision in column 8
--
--   Because the application catches RPC errors and silently
--   falls back to vector-only search, the bug was invisible
--   from the API surface — every "hybrid" call was actually
--   a plain vector call.
--
-- THE FIX:
--   Cast `keyword_score` and `similarity` to DOUBLE PRECISION
--   in the final SELECT so the runtime types match the
--   declared column types.
--
-- HOW TO RUN:
--   Supabase Dashboard → SQL Editor → paste this file → Run.
--   Idempotent: every CREATE uses OR REPLACE.
-- ============================================================


-- ------------------------------------------------------------
-- match_chunks_hybrid — tenant-scoped hybrid search (FIXED)
-- ------------------------------------------------------------
CREATE OR REPLACE FUNCTION match_chunks_hybrid(
    query_embedding   vector(1536),
    query_text        TEXT,
    match_tenant_id   TEXT,
    match_count       INT DEFAULT 30,
    match_frameworks  TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    id              UUID,
    document_id     UUID,
    tenant_id       TEXT,
    content         TEXT,
    chunk_index     INTEGER,
    section_ref     TEXT,
    similarity      FLOAT,
    keyword_score   FLOAT,
    vector_rank     INTEGER,
    keyword_rank    INTEGER
)
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    ts_query tsquery;
BEGIN
    SET LOCAL hnsw.ef_search = 100;

    ts_query := websearch_to_tsquery('simple', coalesce(query_text, ''));

    RETURN QUERY
    WITH vector_hits AS (
        SELECT
            dc.id,
            dc.document_id,
            dc.tenant_id,
            dc.content,
            dc.chunk_index,
            dc.section_ref,
            1 - (dc.embedding <=> query_embedding) AS similarity,
            ROW_NUMBER() OVER (ORDER BY dc.embedding <=> query_embedding) AS vector_rank
        FROM document_chunks dc
        WHERE
            dc.tenant_id = match_tenant_id
            AND dc.embedding IS NOT NULL
            AND (
                match_frameworks IS NULL
                OR cardinality(match_frameworks) = 0
                OR dc.framework IS NULL
                OR dc.framework = ANY(match_frameworks)
            )
        ORDER BY dc.embedding <=> query_embedding
        LIMIT match_count
    ),
    keyword_hits AS (
        SELECT
            dc.id,
            ts_rank_cd(dc.content_tsv, ts_query) AS keyword_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(dc.content_tsv, ts_query) DESC) AS keyword_rank
        FROM document_chunks dc
        WHERE
            dc.tenant_id = match_tenant_id
            AND dc.content_tsv @@ ts_query
            AND (
                match_frameworks IS NULL
                OR cardinality(match_frameworks) = 0
                OR dc.framework IS NULL
                OR dc.framework = ANY(match_frameworks)
            )
        ORDER BY keyword_score DESC
        LIMIT match_count
    ),
    combined AS (
        SELECT v.id FROM vector_hits v
        UNION
        SELECT k.id FROM keyword_hits k
    )
    SELECT
        c.id,
        COALESCE(v.document_id, dc2.document_id)              AS document_id,
        COALESCE(v.tenant_id, dc2.tenant_id)                  AS tenant_id,
        COALESCE(v.content, dc2.content)                      AS content,
        COALESCE(v.chunk_index, dc2.chunk_index)              AS chunk_index,
        COALESCE(v.section_ref, dc2.section_ref)              AS section_ref,
        COALESCE(v.similarity, 0.0)::DOUBLE PRECISION         AS similarity,
        COALESCE(k.keyword_score, 0.0)::DOUBLE PRECISION      AS keyword_score,
        v.vector_rank::INTEGER                                AS vector_rank,
        k.keyword_rank::INTEGER                               AS keyword_rank
    FROM combined c
    LEFT JOIN vector_hits  v  ON v.id = c.id
    LEFT JOIN keyword_hits k  ON k.id = c.id
    LEFT JOIN document_chunks dc2 ON dc2.id = c.id
    LIMIT match_count;
END;
$$;


-- ------------------------------------------------------------
-- match_global_chunks_hybrid — global library hybrid search (FIXED)
-- ------------------------------------------------------------
CREATE OR REPLACE FUNCTION match_global_chunks_hybrid(
    query_embedding   vector(1536),
    query_text        TEXT,
    match_count       INT DEFAULT 15,
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
    similarity      FLOAT,
    keyword_score   FLOAT,
    vector_rank     INTEGER,
    keyword_rank    INTEGER
)
LANGUAGE plpgsql
VOLATILE
AS $$
DECLARE
    ts_query tsquery;
BEGIN
    SET LOCAL hnsw.ef_search = 100;

    ts_query := websearch_to_tsquery('simple', coalesce(query_text, ''));

    RETURN QUERY
    WITH vector_hits AS (
        SELECT
            dc.id,
            dc.document_id,
            dc.tenant_id,
            dc.content,
            dc.chunk_index,
            dc.section_ref,
            1 - (dc.embedding <=> query_embedding) AS similarity,
            ROW_NUMBER() OVER (ORDER BY dc.embedding <=> query_embedding) AS vector_rank
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
        ORDER BY dc.embedding <=> query_embedding
        LIMIT match_count
    ),
    keyword_hits AS (
        SELECT
            dc.id,
            ts_rank_cd(dc.content_tsv, ts_query) AS keyword_score,
            ROW_NUMBER() OVER (ORDER BY ts_rank_cd(dc.content_tsv, ts_query) DESC) AS keyword_rank
        FROM document_chunks dc
        WHERE
            dc.tenant_id = 'global'
            AND dc.content_tsv @@ ts_query
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
        ORDER BY keyword_score DESC
        LIMIT match_count
    ),
    combined AS (
        SELECT v.id FROM vector_hits v
        UNION
        SELECT k.id FROM keyword_hits k
    )
    SELECT
        c.id,
        COALESCE(v.document_id, dc2.document_id)              AS document_id,
        COALESCE(v.tenant_id, dc2.tenant_id)                  AS tenant_id,
        COALESCE(v.content, dc2.content)                      AS content,
        COALESCE(v.chunk_index, dc2.chunk_index)              AS chunk_index,
        COALESCE(v.section_ref, dc2.section_ref)              AS section_ref,
        COALESCE(v.similarity, 0.0)::DOUBLE PRECISION         AS similarity,
        COALESCE(k.keyword_score, 0.0)::DOUBLE PRECISION      AS keyword_score,
        v.vector_rank::INTEGER                                AS vector_rank,
        k.keyword_rank::INTEGER                               AS keyword_rank
    FROM combined c
    LEFT JOIN vector_hits  v  ON v.id = c.id
    LEFT JOIN keyword_hits k  ON k.id = c.id
    LEFT JOIN document_chunks dc2 ON dc2.id = c.id
    LIMIT match_count;
END;
$$;
