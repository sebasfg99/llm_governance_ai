-- ============================================================
-- 011_reference_list_flag.sql
--
-- Adds is_reference_list to document_chunks so bibliography /
-- reference-list chunks can be DEMOTED at rerank time.
--
-- THE PROBLEM THIS SOLVES:
-- A bibliography is a list of standard TITLES. Embedded, it lands
-- semantically closer to a question about those standards than the
-- substantive clause that merely mentions them in passing. Measured
-- on the ISO 42001 corpus: the chunk carrying the transparency and
-- explicability clause (Anexo C, chunk_index 64) ranked 227th by
-- cosine similarity, while two Anexo D bibliography chunks
-- (chunk_index 69 and 70 — a list of ISO/IEC standard names) ranked
-- 3rd and 5th. The answer built from that context supported 0 of 4
-- claims.
--
-- WHY A FLAG AND NOT A DELETE:
-- The content stays in the index and stays retrievable. A user who
-- asks "which ISO standards does Anexo D of ISO 42001 cite?" must
-- still get that list. Only its PRIORITY drops, and only when the
-- question is not about references — the penalty is applied in
-- Python by rerank_chunks(), which can see the query. That is why
-- this migration adds no filter parameter: the four RPCs merely
-- RETURN the new column so the reranker can read it.
--
-- WHY CHUNK LEVEL AND NOT DOCUMENT LEVEL:
-- The classification is per section, not per document. ISO 42001 is
-- 71 chunks of which exactly 2 are bibliography; a document-level
-- flag could not express that.
--
-- WHAT THIS MIGRATION DOES:
--
--   A) is_reference_list column on document_chunks.
--      BOOLEAN NOT NULL DEFAULT FALSE: an unclassified row means
--      "ordinary content", which is the safe default — it keeps the
--      row at its current priority instead of silently demoting it.
--      Rows stay FALSE until the document is reprocessed, exactly
--      as agent_tags stayed '{}' until the seed script re-tagged.
--
--      No index. Unlike agent_tags/industry_codes this column is
--      never a WHERE predicate — nothing filters on it in SQL — so
--      an index would only cost write throughput.
--
--   B) The four retrieval RPCs now also RETURN is_reference_list:
--        - match_chunks               (tenant, vector-only)
--        - match_global_chunks        (global, vector-only)
--        - match_chunks_hybrid        (tenant, vector + keyword)
--        - match_global_chunks_hybrid (global, vector + keyword)
--      Argument lists are UNCHANGED from migration 010.
--
-- IMPORTANT: CREATE OR REPLACE cannot change a function's return
-- type ("cannot change return type of existing function"), and the
-- RETURNS TABLE gains a column here. Each function is therefore
-- DROPped and recreated. Since the argument lists are unchanged,
-- there is exactly one signature to drop per function — the one
-- migration 010 left behind.
--
-- ORDERING: apply AFTER 010_industry_filter_fix.sql.
--
-- HOW TO RUN:
--   Supabase Dashboard -> SQL Editor -> paste this file -> Run.
--   Idempotent: the column uses IF NOT EXISTS, the drops IF EXISTS.
-- ============================================================


-- ============================================================
-- A) Add the column
-- ============================================================
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS is_reference_list BOOLEAN NOT NULL DEFAULT FALSE;

COMMENT ON COLUMN document_chunks.is_reference_list IS
  'TRUE when the chunk is predominantly a bibliography / reference list rather than substantive content. Set at chunking time by ReferenceRegionClassifier (app/domain/documents/reference_sections.py). Read by rerank_chunks() to demote the chunk unless the query is itself asking about references. FALSE = ordinary content, and also the not-yet-classified default.';


-- ============================================================
-- B0) Drop the migration-010 signatures
--
-- Required because the RETURNS TABLE changes; see header. The
-- argument lists below are exactly those created by 010.
-- ============================================================
DROP FUNCTION IF EXISTS match_chunks(vector, TEXT, INT, TEXT[], TEXT[], TEXT[]);
DROP FUNCTION IF EXISTS match_global_chunks(vector, INT, TEXT, TEXT[], TEXT[], TEXT[]);
DROP FUNCTION IF EXISTS match_chunks_hybrid(vector, TEXT, TEXT, INT, TEXT[], TEXT[], TEXT[]);
DROP FUNCTION IF EXISTS match_global_chunks_hybrid(vector, TEXT, INT, TEXT, TEXT[], TEXT[], TEXT[]);



-- ============================================================
-- B1) match_chunks — tenant-scoped vector search
--
-- Unchanged from migration 010 except that the result set now
-- carries is_reference_list. Arguments and filtering are identical.
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
    is_reference_list BOOLEAN,
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
            dc.is_reference_list,
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
  'As migration 010, plus is_reference_list in the result set so rerank_chunks() can demote bibliography chunks.';


-- ============================================================
-- B2) match_global_chunks — global library vector search
--
-- Unchanged from migration 010 except for the extra returned
-- column.
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
    is_reference_list BOOLEAN,
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
            dc.is_reference_list,
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
  'As migration 010, plus is_reference_list in the result set so rerank_chunks() can demote bibliography chunks.';


-- ============================================================
-- B3) match_chunks_hybrid — tenant hybrid search
--
-- Unchanged from migration 010 except for the extra returned
-- column. is_reference_list is COALESCEd from the vector branch to
-- the base table so a chunk found ONLY by the keyword branch still
-- reports its true flag instead of defaulting to FALSE.
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
    is_reference_list BOOLEAN,
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
            dc.is_reference_list,
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
        COALESCE(v.is_reference_list, dc2.is_reference_list)  AS is_reference_list,
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
  'As migration 010, plus is_reference_list in the result set so rerank_chunks() can demote bibliography chunks.';


-- ============================================================
-- B4) match_global_chunks_hybrid — global hybrid search
--
-- Unchanged from migration 010 except for the extra returned
-- column, COALESCEd as in B3.
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
    is_reference_list BOOLEAN,
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
            dc.is_reference_list,
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
        COALESCE(v.is_reference_list, dc2.is_reference_list)  AS is_reference_list,
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
  'As migration 010, plus is_reference_list in the result set so rerank_chunks() can demote bibliography chunks.';
