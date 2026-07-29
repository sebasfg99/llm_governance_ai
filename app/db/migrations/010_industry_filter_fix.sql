-- ============================================================
-- 010_industry_filter_fix.sql
--
-- Fixes a SILENT CORRECTNESS BUG: the industry_codes filter has
-- never actually filtered anything.
--
-- THE BUG (state before this migration):
--   Migration 008 added the industry_codes column to documents
--   and document_chunks and backfilled the SUGEF/CONASSIF rows,
--   but it never touched the retrieval RPCs. As a result:
--
--     a) None of the four retrieval functions had an industry
--        parameter, so no industry predicate ever reached SQL.
--     b) None of them listed industry_codes in their RETURNS
--        TABLE, so the column never travelled to the client.
--     c) The Python post-filter (_filter_by_industry) is
--        permissive: a chunk whose industry_codes is NULL or
--        empty is treated as "applies to everyone". Since (b)
--        guaranteed the field arrived as NULL for EVERY chunk,
--        the post-filter accepted everything.
--
--   Net effect: a cooperative-only institution could receive
--   bank-only regulatory content, with no error and no log line.
--   This is an accuracy bug, not a cleanup task.
--
-- THE FIX: filter in SQL, the same way migration 009 does for
-- agent_tags. Metadata filtering must happen in the query, not
-- as post-processing over data that never travels.
--
-- WHAT THIS MIGRATION DOES:
--
--   A) GIN index on document_chunks.industry_codes
--      Serves the array-overlap operator used by the filter,
--      exactly as idx_chunks_agent_tags does for agent_tags.
--      Migration 008 never created one.
--
--   B) The four retrieval RPCs gain an optional
--      p_industry_codes parameter:
--        - match_chunks              (tenant, vector-only)
--        - match_global_chunks       (global, vector-only)
--        - match_chunks_hybrid       (tenant, vector + keyword)
--        - match_global_chunks_hybrid(global, vector + keyword)
--
--   C) The four RPCs now also RETURN industry_codes, so the
--      application can log it and assert on it. The filtering
--      itself happens in SQL; the returned column exists so the
--      client can detect a filter that silently failed to apply
--      (see RetrievalService._warn_on_industry_mismatch) instead
--      of trusting it blindly — which is what caused this bug.
--
-- FILTER SEMANTICS (deliberately permissive, and identical to
-- the Python rules it replaces):
--   p_industry_codes NULL or empty     -> no restriction at all,
--                                         so every existing
--                                         caller is unaffected.
--   chunk industry_codes NULL or empty -> always included
--                                         (transition safety net
--                                         for rows predating 008).
--   chunk industry_codes contains ALL  -> always included
--                                         (Layer 1-2 material:
--                                         ISO, NIST, OWASP…).
--   otherwise                          -> included only when the
--                                         arrays overlap, i.e.
--                                         the chunk is tagged for
--                                         the caller's industry.
--
-- PREREQUISITE: migration 009_agent_tags.sql must already be
-- applied. These function bodies reference dc.agent_tags, and
-- the DROP statements below target the signatures 009 created.
--
-- IMPORTANT: PostgreSQL identifies a function by its name plus
-- input argument types, so adding a parameter with CREATE OR
-- REPLACE would create an OVERLOAD and PostgREST calls by name
-- would fail with "function is not unique". We DROP every known
-- historical signature explicitly before each recreate (see B1),
-- so a database in ANY intermediate state converges to exactly
-- one version of each function. The COMMENT ON FUNCTION
-- statements below also spell out their argument types — the
-- unqualified form used in 004/005 is what makes those two
-- migrations error out mid-script while an overload exists.
--
-- HOW TO RUN:
--   Supabase Dashboard -> SQL Editor -> paste this file -> Run.
--   Idempotent: the index uses IF NOT EXISTS, drops use IF EXISTS.
--   No data migration is needed — 008 already populated the
--   column and scripts/seed_documents.py keeps it current.
-- ============================================================


-- ============================================================
-- A) GIN index for the array-overlap filter
--
-- Serves queries of the form:
--   WHERE industry_codes && ARRAY['bank']
-- Mirrors idx_chunks_agent_tags from migration 009. On a small
-- corpus the planner may still prefer a sequential scan; that is
-- expected and fine.
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_chunks_industry_codes
    ON document_chunks
    USING GIN (industry_codes);

COMMENT ON INDEX idx_chunks_industry_codes IS
  'GIN index so industry_codes && ARRAY[...] (overlap) filters scale with corpus size.';


-- ============================================================
-- B1) Drop every historical signature
--
-- On a database that ran the full documented chain, only the 009
-- signatures exist (verified: 001/003's 3-argument forms are
-- dropped by 005, and 006 drops the pre-007 hybrid forms). The
-- first DROP block below is therefore the one that does the work.
--
-- The older shapes are dropped too, purely defensively, because a
-- database can legitimately be stuck mid-chain: migration 004
-- creates a 4-argument overload next to 001/003's 3-argument one
-- and then runs `COMMENT ON FUNCTION match_chunks` WITHOUT an
-- argument list, which fails with "function name is not unique"
-- while both exist. In the Supabase SQL editor that error does not
-- stop the rest of the script, and 005 cleans the overload up — so
-- the ambiguity is transient and self-healing. But a database that
-- stopped at 004 would still carry both, and blindly recreating on
-- top of that would leave PostgREST with an ambiguous name.
--
-- Every DROP is IF EXISTS, so applying this to a database in any
-- intermediate state is safe, and the end state is exactly one
-- version of each function.
-- ============================================================

-- 009 signatures (… match_frameworks TEXT[], p_agent_tags TEXT[])
DROP FUNCTION IF EXISTS match_chunks(vector, TEXT, INT, TEXT[], TEXT[]);
DROP FUNCTION IF EXISTS match_global_chunks(vector, INT, TEXT, TEXT[], TEXT[]);
DROP FUNCTION IF EXISTS match_chunks_hybrid(vector, TEXT, TEXT, INT, TEXT[], TEXT[]);
DROP FUNCTION IF EXISTS match_global_chunks_hybrid(vector, TEXT, INT, TEXT, TEXT[], TEXT[]);

-- 004 / 005 / 006 / 007 signatures (… match_frameworks TEXT[])
DROP FUNCTION IF EXISTS match_chunks(vector, TEXT, INT, TEXT[]);
DROP FUNCTION IF EXISTS match_global_chunks(vector, INT, TEXT, TEXT[]);
DROP FUNCTION IF EXISTS match_chunks_hybrid(vector, TEXT, TEXT, INT, TEXT[]);
DROP FUNCTION IF EXISTS match_global_chunks_hybrid(vector, TEXT, INT, TEXT, TEXT[]);

-- 001 / 002 / 003 signatures (no framework filter at all).
-- These are the ones 009 missed.
DROP FUNCTION IF EXISTS match_chunks(vector, TEXT, INT);
DROP FUNCTION IF EXISTS match_global_chunks(vector, INT, TEXT);


-- ============================================================
-- B2) match_chunks — tenant-scoped vector search
--
-- Use case: retrieve the most similar chunks of one tenant's
--           own documents for a RAG answer.
-- Input:    query_embedding   vector(1536) of the user question
--           match_tenant_id   tenant owning the documents
--           match_count       max rows to return
--           match_frameworks  optional framework slug filter
--           p_agent_tags      optional agent slug filter (overlap)
--           p_industry_codes  optional industry filter (overlap,
--                             permissive — see header)
-- Output:   chunk rows (now including industry_codes) ordered by
--           cosine similarity, descending.
-- Body is identical to 009_agent_tags.sql plus the industry
-- condition and the extra returned column.
-- ============================================================
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding   vector(1536),
    match_tenant_id   TEXT,
    match_count       INT DEFAULT 5,
    match_frameworks  TEXT[] DEFAULT NULL,
    p_agent_tags      TEXT[] DEFAULT NULL,
    p_industry_codes  TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    id              UUID,
    document_id     UUID,
    tenant_id       TEXT,
    content         TEXT,
    chunk_index     INTEGER,
    section_ref     TEXT,
    industry_codes  TEXT[],
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
        dc.industry_codes,
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
        AND (
            -- Agent filter: NULL / empty array -> no restriction,
            -- so existing callers keep their exact behaviour.
            p_agent_tags IS NULL
            OR cardinality(p_agent_tags) = 0
            OR dc.agent_tags && p_agent_tags
        )
        AND (
            -- Industry filter (permissive; see migration header).
            -- NULL / empty request  -> no restriction.
            -- NULL / empty on chunk -> always included.
            -- 'ALL' on chunk        -> always included (Layer 1-2).
            -- otherwise             -> arrays must overlap.
            p_industry_codes IS NULL
            OR cardinality(p_industry_codes) = 0
            OR dc.industry_codes IS NULL
            OR cardinality(dc.industry_codes) = 0
            OR dc.industry_codes && ARRAY['ALL']
            OR dc.industry_codes && p_industry_codes
        )
    ORDER BY
        dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_chunks(vector, TEXT, INT, TEXT[], TEXT[], TEXT[]) IS
  'Vector similarity search for RAG retrieval. Optional p_agent_tags and p_industry_codes filter chunks by business agent and by institution industry via array overlap. Returns industry_codes so callers can verify the filter applied. Sets hnsw.ef_search=100.';


-- ============================================================
-- B3) match_global_chunks — global library vector search
--
-- Use case: retrieve chunks from the shared "global" corpus,
--           optionally scoped to one country.
-- Input:    query_embedding   vector(1536) of the user question
--           match_count       max rows to return
--           match_country     NULL -> country-agnostic docs only;
--                             value -> that country's docs plus
--                             country-agnostic docs
--           match_frameworks  optional framework slug filter
--           p_agent_tags      optional agent slug filter (overlap)
--           p_industry_codes  optional industry filter (overlap)
-- Output:   chunk rows (including industry_codes) ordered by
--           cosine similarity, descending.
--
-- NOTE: this is the function that matters most for the bug being
-- fixed. Layer-3 sector documents (SUGEF, CONASSIF) live in the
-- global library, so this is where bank-only content was leaking
-- into cooperative-only institutions' answers.
-- ============================================================
CREATE OR REPLACE FUNCTION match_global_chunks(
    query_embedding   vector(1536),
    match_count       INT DEFAULT 3,
    match_country     TEXT DEFAULT NULL,
    match_frameworks  TEXT[] DEFAULT NULL,
    p_agent_tags      TEXT[] DEFAULT NULL,
    p_industry_codes  TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    id              UUID,
    document_id     UUID,
    tenant_id       TEXT,
    content         TEXT,
    chunk_index     INTEGER,
    section_ref     TEXT,
    industry_codes  TEXT[],
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
        dc.industry_codes,
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
        AND (
            -- Agent filter (see match_chunks for semantics).
            p_agent_tags IS NULL
            OR cardinality(p_agent_tags) = 0
            OR dc.agent_tags && p_agent_tags
        )
        AND (
            -- Industry filter (see match_chunks for semantics).
            p_industry_codes IS NULL
            OR cardinality(p_industry_codes) = 0
            OR dc.industry_codes IS NULL
            OR cardinality(dc.industry_codes) = 0
            OR dc.industry_codes && ARRAY['ALL']
            OR dc.industry_codes && p_industry_codes
        )
    ORDER BY
        dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_global_chunks(vector, INT, TEXT, TEXT[], TEXT[], TEXT[]) IS
  'Vector similarity search over the global library. Optional p_agent_tags and p_industry_codes filter by business agent and institution industry via array overlap. Returns industry_codes. Sets hnsw.ef_search=100.';


-- ============================================================
-- B4) match_chunks_hybrid — tenant hybrid search
--
-- Use case: candidate generation for hybrid retrieval (vector
--           similarity UNION full-text keyword hits) over one
--           tenant's documents; the application fuses the two
--           rankings afterwards (RRF).
-- Input:    query_embedding   vector(1536) of the user question
--           query_text        raw question for websearch_to_tsquery
--           match_tenant_id   tenant owning the documents
--           match_count       max rows per branch and overall
--           match_frameworks  optional framework slug filter
--           p_agent_tags      optional agent slug filter (overlap)
--           p_industry_codes  optional industry filter (overlap)
-- Output:   chunk rows with industry_codes, similarity,
--           keyword_score and the per-branch ranks (vector_rank /
--           keyword_rank; NULL when the chunk was found by only
--           one branch).
--
-- Both new filters are applied to BOTH branches, so a chunk
-- excluded for one industry can never re-enter through a keyword
-- match. This is the same rule migration 009 established for
-- agent tags, and the reason the filter belongs in SQL: a Python
-- post-filter runs after fusion, when the excluded chunk has
-- already displaced a legitimate candidate from the top-k.
-- ============================================================
CREATE OR REPLACE FUNCTION match_chunks_hybrid(
    query_embedding   vector(1536),
    query_text        TEXT,
    match_tenant_id   TEXT,
    match_count       INT DEFAULT 30,
    match_frameworks  TEXT[] DEFAULT NULL,
    p_agent_tags      TEXT[] DEFAULT NULL,
    p_industry_codes  TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    id              UUID,
    document_id     UUID,
    tenant_id       TEXT,
    content         TEXT,
    chunk_index     INTEGER,
    section_ref     TEXT,
    industry_codes  TEXT[],
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
            dc.industry_codes,
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
            AND (
                -- Agent filter, vector branch.
                p_agent_tags IS NULL
                OR cardinality(p_agent_tags) = 0
                OR dc.agent_tags && p_agent_tags
            )
            AND (
                -- Industry filter, vector branch.
                p_industry_codes IS NULL
                OR cardinality(p_industry_codes) = 0
                OR dc.industry_codes IS NULL
                OR cardinality(dc.industry_codes) = 0
                OR dc.industry_codes && ARRAY['ALL']
                OR dc.industry_codes && p_industry_codes
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
            AND (
                -- Agent filter, keyword branch.
                p_agent_tags IS NULL
                OR cardinality(p_agent_tags) = 0
                OR dc.agent_tags && p_agent_tags
            )
            AND (
                -- Industry filter, keyword branch.
                p_industry_codes IS NULL
                OR cardinality(p_industry_codes) = 0
                OR dc.industry_codes IS NULL
                OR cardinality(dc.industry_codes) = 0
                OR dc.industry_codes && ARRAY['ALL']
                OR dc.industry_codes && p_industry_codes
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
        COALESCE(v.industry_codes, dc2.industry_codes)        AS industry_codes,
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

COMMENT ON FUNCTION match_chunks_hybrid(vector, TEXT, TEXT, INT, TEXT[], TEXT[], TEXT[]) IS
  'Tenant-scoped hybrid (vector + keyword) candidate search. Optional p_agent_tags and p_industry_codes filter BOTH branches via array overlap. Returns industry_codes.';


-- ============================================================
-- B5) match_global_chunks_hybrid — global hybrid search
--
-- Use case: candidate generation for hybrid retrieval over the
--           shared "global" corpus, optionally country-scoped.
-- Input:    query_embedding   vector(1536) of the user question
--           query_text        raw question for websearch_to_tsquery
--           match_count       max rows per branch and overall
--           match_country     NULL -> country-agnostic docs only;
--                             value -> that country's docs plus
--                             country-agnostic docs
--           match_frameworks  optional framework slug filter
--           p_agent_tags      optional agent slug filter (overlap)
--           p_industry_codes  optional industry filter (overlap)
-- Output:   same shape as match_chunks_hybrid.
-- ============================================================
CREATE OR REPLACE FUNCTION match_global_chunks_hybrid(
    query_embedding   vector(1536),
    query_text        TEXT,
    match_count       INT DEFAULT 15,
    match_country     TEXT DEFAULT NULL,
    match_frameworks  TEXT[] DEFAULT NULL,
    p_agent_tags      TEXT[] DEFAULT NULL,
    p_industry_codes  TEXT[] DEFAULT NULL
)
RETURNS TABLE (
    id              UUID,
    document_id     UUID,
    tenant_id       TEXT,
    content         TEXT,
    chunk_index     INTEGER,
    section_ref     TEXT,
    industry_codes  TEXT[],
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
            dc.industry_codes,
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
            AND (
                -- Agent filter, vector branch.
                p_agent_tags IS NULL
                OR cardinality(p_agent_tags) = 0
                OR dc.agent_tags && p_agent_tags
            )
            AND (
                -- Industry filter, vector branch.
                p_industry_codes IS NULL
                OR cardinality(p_industry_codes) = 0
                OR dc.industry_codes IS NULL
                OR cardinality(dc.industry_codes) = 0
                OR dc.industry_codes && ARRAY['ALL']
                OR dc.industry_codes && p_industry_codes
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
            AND (
                -- Agent filter, keyword branch.
                p_agent_tags IS NULL
                OR cardinality(p_agent_tags) = 0
                OR dc.agent_tags && p_agent_tags
            )
            AND (
                -- Industry filter, keyword branch.
                p_industry_codes IS NULL
                OR cardinality(p_industry_codes) = 0
                OR dc.industry_codes IS NULL
                OR cardinality(dc.industry_codes) = 0
                OR dc.industry_codes && ARRAY['ALL']
                OR dc.industry_codes && p_industry_codes
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
        COALESCE(v.industry_codes, dc2.industry_codes)        AS industry_codes,
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

COMMENT ON FUNCTION match_global_chunks_hybrid(vector, TEXT, INT, TEXT, TEXT[], TEXT[], TEXT[]) IS
  'Global-library hybrid (vector + keyword) candidate search. Optional p_agent_tags and p_industry_codes filter BOTH branches via array overlap. Returns industry_codes.';
