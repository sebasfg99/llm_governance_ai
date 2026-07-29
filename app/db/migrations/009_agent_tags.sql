-- ============================================================
-- 009_agent_tags.sql
--
-- Adds business-agent tagging to the RAG corpus so each of the
-- specialized agents (normativa, seguridad, proveedores,
-- privacidad) can retrieve only the content that applies to it.
--
-- WHAT THIS MIGRATION DOES:
--
--   A) agent_tags column on documents + document_chunks
--      A TEXT[] holding one or more agent slugs. A document can
--      belong to several agents at once (e.g. GDPR content is
--      relevant to privacidad, normativa AND proveedores).
--      Documents whose framework is not recognized are tagged
--      ['general'] by the seed script so they stay retrievable
--      as a fallback instead of silently disappearing from
--      every agent. Default is the empty array: rows keep '{}'
--      until scripts/seed_documents.py --force re-tags them
--      (the framework -> agent mapping lives only in Python,
--      FRAMEWORK_AGENT_MAP, so it is not duplicated here).
--
--   B) GIN index on document_chunks.agent_tags
--      Retrieval filters at chunk level with the array-overlap
--      operator (agent_tags && ARRAY[...]); GIN is the index
--      type that serves && efficiently.
--
--   C) The four retrieval RPCs gain an optional p_agent_tags
--      parameter:
--        - match_chunks              (tenant, vector-only)
--        - match_global_chunks       (global, vector-only)
--        - match_chunks_hybrid       (tenant, vector + keyword)
--        - match_global_chunks_hybrid(global, vector + keyword)
--      Semantics: NULL or empty array -> no agent filtering
--      (existing callers that do not pass the parameter behave
--      exactly as before). Non-empty -> only chunks whose
--      agent_tags overlaps the requested set are returned.
--
-- IMPORTANT: PostgreSQL identifies a function by its name plus
-- input argument types. CREATE OR REPLACE with a new parameter
-- would therefore create an OVERLOAD next to the old function,
-- and PostgREST calls by name would fail with "function is not
-- unique". We DROP the previous signatures explicitly before
-- each recreate so exactly one version of each function exists.
--
-- HOW TO RUN:
--   Supabase Dashboard -> SQL Editor -> paste this file -> Run.
--   Idempotent: columns/index use IF NOT EXISTS, drops use
--   IF EXISTS.
-- ============================================================


-- ============================================================
-- A) Add agent_tags columns
--
-- documents:        source of truth, written by the seed script.
-- document_chunks:  denormalized copy written on every
--                   process_document() run; retrieval filters
--                   here (chunk level), so the column must exist
--                   on both tables.
-- Default '{}' (empty array), NOT NULL: an untagged row means
-- "not yet classified", never NULL.
-- ============================================================
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS agent_tags TEXT[] NOT NULL DEFAULT '{}';

ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS agent_tags TEXT[] NOT NULL DEFAULT '{}';

COMMENT ON COLUMN documents.agent_tags IS
  'Business agents this document applies to (normativa, seguridad, proveedores, privacidad, general). Derived statically from framework by scripts/seed_documents.py (FRAMEWORK_AGENT_MAP). Empty array = not yet tagged.';

COMMENT ON COLUMN document_chunks.agent_tags IS
  'Copy of the parent document''s agent_tags, denormalized at chunk-insert time so retrieval can filter without a JOIN.';


-- ============================================================
-- B) GIN index for the array-overlap filter
--
-- Serves queries of the form:
--   WHERE agent_tags && ARRAY['proveedores', 'general']
-- Note: on a small corpus the planner may still prefer a
-- sequential scan; that is expected and fine.
-- ============================================================
CREATE INDEX IF NOT EXISTS idx_chunks_agent_tags
    ON document_chunks
    USING GIN (agent_tags);

COMMENT ON INDEX idx_chunks_agent_tags IS
  'GIN index so agent_tags && ARRAY[...] (overlap) filters scale with corpus size.';


-- ============================================================
-- C1) Drop previous signatures (see header for why DROP is
--     required instead of plain CREATE OR REPLACE)
-- ============================================================
DROP FUNCTION IF EXISTS match_chunks(vector, TEXT, INT, TEXT[]);
DROP FUNCTION IF EXISTS match_global_chunks(vector, INT, TEXT, TEXT[]);
DROP FUNCTION IF EXISTS match_chunks_hybrid(vector, TEXT, TEXT, INT, TEXT[]);
DROP FUNCTION IF EXISTS match_global_chunks_hybrid(vector, TEXT, INT, TEXT, TEXT[]);


-- ============================================================
-- C2) match_chunks — tenant-scoped vector search + agent filter
--
-- Use case: retrieve the most similar chunks of one tenant's
--           own documents for a RAG answer.
-- Input:    query_embedding  vector(1536) of the user question
--           match_tenant_id  tenant owning the documents
--           match_count      max rows to return
--           match_frameworks optional framework slug filter
--           p_agent_tags     optional agent slug filter (overlap)
-- Output:   chunk rows ordered by cosine similarity (descending).
-- Body is identical to 005_section_ref_hnsw.sql plus the
-- p_agent_tags condition.
-- ============================================================
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding   vector(1536),
    match_tenant_id   TEXT,
    match_count       INT DEFAULT 5,
    match_frameworks  TEXT[] DEFAULT NULL,
    p_agent_tags      TEXT[] DEFAULT NULL
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
        AND (
            -- Agent filter: NULL / empty array -> no restriction,
            -- so existing callers keep their exact behaviour.
            -- Otherwise the chunk must share at least one tag with
            -- the requested set (&& = array overlap, served by the
            -- GIN index idx_chunks_agent_tags).
            p_agent_tags IS NULL
            OR cardinality(p_agent_tags) = 0
            OR dc.agent_tags && p_agent_tags
        )
    ORDER BY
        dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_chunks IS
  'Vector similarity search for RAG retrieval. Optional p_agent_tags filters chunks by business agent via array overlap. Sets hnsw.ef_search=100.';


-- ============================================================
-- C3) match_global_chunks — global library vector search + agent
--     filter
--
-- Use case: retrieve chunks from the shared "global" corpus,
--           optionally scoped to one country.
-- Input:    query_embedding  vector(1536) of the user question
--           match_count      max rows to return
--           match_country    NULL -> country-agnostic docs only;
--                            value -> that country's docs plus
--                            country-agnostic docs
--           match_frameworks optional framework slug filter
--           p_agent_tags     optional agent slug filter (overlap)
-- Output:   chunk rows ordered by cosine similarity (descending).
-- ============================================================
CREATE OR REPLACE FUNCTION match_global_chunks(
    query_embedding   vector(1536),
    match_count       INT DEFAULT 3,
    match_country     TEXT DEFAULT NULL,
    match_frameworks  TEXT[] DEFAULT NULL,
    p_agent_tags      TEXT[] DEFAULT NULL
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
        AND (
            -- Agent filter (see match_chunks for semantics).
            p_agent_tags IS NULL
            OR cardinality(p_agent_tags) = 0
            OR dc.agent_tags && p_agent_tags
        )
    ORDER BY
        dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_global_chunks IS
  'Vector similarity search over the global library. Optional p_agent_tags filters chunks by business agent via array overlap. Sets hnsw.ef_search=100.';


-- ============================================================
-- C4) match_chunks_hybrid — tenant hybrid search + agent filter
--
-- Use case: candidate generation for hybrid retrieval (vector
--           similarity UNION full-text keyword hits) over one
--           tenant's documents; the application fuses the two
--           rankings afterwards (RRF).
-- Input:    query_embedding  vector(1536) of the user question
--           query_text       raw question for websearch_to_tsquery
--           match_tenant_id  tenant owning the documents
--           match_count      max rows per branch and overall
--           match_frameworks optional framework slug filter
--           p_agent_tags     optional agent slug filter (overlap)
-- Output:   chunk rows with similarity, keyword_score and the
--           per-branch ranks (vector_rank / keyword_rank; NULL
--           when the chunk was found by only one branch).
-- Body is identical to 007_fix_hybrid_rpc_types.sql plus the
-- p_agent_tags condition applied to BOTH branches, so a chunk
-- excluded for one agent can never re-enter via keywords.
-- ============================================================
CREATE OR REPLACE FUNCTION match_chunks_hybrid(
    query_embedding   vector(1536),
    query_text        TEXT,
    match_tenant_id   TEXT,
    match_count       INT DEFAULT 30,
    match_frameworks  TEXT[] DEFAULT NULL,
    p_agent_tags      TEXT[] DEFAULT NULL
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
            AND (
                -- Agent filter, vector branch.
                p_agent_tags IS NULL
                OR cardinality(p_agent_tags) = 0
                OR dc.agent_tags && p_agent_tags
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

COMMENT ON FUNCTION match_chunks_hybrid IS
  'Tenant-scoped hybrid (vector + keyword) candidate search. Optional p_agent_tags filters both branches by business agent via array overlap.';


-- ============================================================
-- C5) match_global_chunks_hybrid — global hybrid search + agent
--     filter
--
-- Use case: candidate generation for hybrid retrieval over the
--           shared "global" corpus, optionally country-scoped.
-- Input:    query_embedding  vector(1536) of the user question
--           query_text       raw question for websearch_to_tsquery
--           match_count      max rows per branch and overall
--           match_country    NULL -> country-agnostic docs only;
--                            value -> that country's docs plus
--                            country-agnostic docs
--           match_frameworks optional framework slug filter
--           p_agent_tags     optional agent slug filter (overlap)
-- Output:   same shape as match_chunks_hybrid.
-- ============================================================
CREATE OR REPLACE FUNCTION match_global_chunks_hybrid(
    query_embedding   vector(1536),
    query_text        TEXT,
    match_count       INT DEFAULT 15,
    match_country     TEXT DEFAULT NULL,
    match_frameworks  TEXT[] DEFAULT NULL,
    p_agent_tags      TEXT[] DEFAULT NULL
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
            AND (
                -- Agent filter, vector branch.
                p_agent_tags IS NULL
                OR cardinality(p_agent_tags) = 0
                OR dc.agent_tags && p_agent_tags
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

COMMENT ON FUNCTION match_global_chunks_hybrid IS
  'Global-library hybrid (vector + keyword) candidate search. Optional p_agent_tags filters both branches by business agent via array overlap.';
