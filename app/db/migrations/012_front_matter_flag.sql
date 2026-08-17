    -- ============================================================
    -- 012_front_matter_flag.sql
    --
    -- Adds is_front_matter to document_chunks so cover pages, tables of
    -- contents, copyright notices and committee credit lists can be
    -- DEMOTED at rerank time.
    --
    -- THE PROBLEM THIS SOLVES:
    -- Front matter shares no substance with user questions but collides
    -- with them lexically and semantically. Measured case: for the query
    -- "¿Qué normativa de gobernanza de IA aplica en Costa Rica?" the
    -- ISO 42001 committee credits — a list of Costa Rican universities —
    -- outscored substantive content purely because the country name
    -- repeats through the names of the member institutions, and the
    -- generated answer was flagged as hallucinated by the judge.
    --
    -- WHY THE FLAG IS EXACT (unlike heuristics measured and rejected):
    -- The chunker now detects the document's body-start heading
    -- ("0 INTRODUCCIÓN", "Executive Summary"...) in the RAW text and
    -- forces a HARD chunk cut at that boundary, so no chunk mixes
    -- credits with introduction. Chunks before the boundary are front
    -- matter by construction. Documents without a detectable heading get
    -- no boundary and no flags — under-flagging is the safe direction.
    --
    -- WHY A FLAG AND NOT A DELETE:
    -- Same design as is_reference_list (migration 011): "¿quién participó
    -- en el comité técnico de la norma?" is a legitimate question whose
    -- answer IS the credits list. The flag is returned by the RPCs so the
    -- Python reranker can demote — never SQL-filter — these chunks.
    --
    -- WHAT THIS MIGRATION DOES:
    --   A) is_front_matter BOOLEAN NOT NULL DEFAULT FALSE on
    --      document_chunks. Existing rows stay FALSE ("ordinary content")
    --      until their document is reprocessed.
    --      No index: never a WHERE predicate.
    --   B) The four retrieval RPCs additionally RETURN is_front_matter.
    --      Argument lists are UNCHANGED from migrations 010/011.
    --
    -- IMPORTANT: CREATE OR REPLACE cannot change a function's return
    -- type, and RETURNS TABLE gains a column here — so each function is
    -- DROPped and recreated, exactly as migration 011 did.
    --
    -- ORDERING: apply AFTER 011_reference_list_flag.sql.
    --
    -- HOW TO RUN:
    --   Supabase Dashboard -> SQL Editor -> paste this file -> Run.
    --   Idempotent: the column uses IF NOT EXISTS, the drops IF EXISTS.
    -- ============================================================


    -- ============================================================
    -- A) Add the column
    -- ============================================================
    ALTER TABLE document_chunks
        ADD COLUMN IF NOT EXISTS is_front_matter BOOLEAN NOT NULL DEFAULT FALSE;

    COMMENT ON COLUMN document_chunks.is_front_matter IS
    'TRUE when the chunk lies entirely before the document body-start boundary (cover, TOC, copyright, committee credits). Set at chunking time by find_body_start (app/domain/documents/front_matter.py) plus a hard chunk cut at the boundary. Read by the reranker to demote such chunks. FALSE = ordinary content, and also the not-yet-classified default.';


    -- ============================================================
    -- B0) Drop the migration-011 signatures (return type changes)
    -- ============================================================
    DROP FUNCTION IF EXISTS match_chunks(vector, TEXT, INT, TEXT[], TEXT[], TEXT[]);
    DROP FUNCTION IF EXISTS match_global_chunks(vector, INT, TEXT, TEXT[], TEXT[], TEXT[]);
    DROP FUNCTION IF EXISTS match_chunks_hybrid(vector, TEXT, TEXT, INT, TEXT[], TEXT[], TEXT[]);
    DROP FUNCTION IF EXISTS match_global_chunks_hybrid(vector, TEXT, INT, TEXT, TEXT[], TEXT[], TEXT[]);



    -- ============================================================
    -- B1) match_chunks — as migration 011 plus is_front_matter in the
    --     result set. Arguments and filtering identical.
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
        is_front_matter BOOLEAN,
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
                dc.is_front_matter,
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
    'As migration 011, plus is_front_matter in the result set so the reranker can demote cover/TOC/credits chunks.';


    -- ============================================================
    -- B2) match_global_chunks — as migration 011 plus the extra column.
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
        is_front_matter BOOLEAN,
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
                dc.is_front_matter,
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
    'As migration 011, plus is_front_matter in the result set so the reranker can demote cover/TOC/credits chunks.';


    -- ============================================================
    -- B3) match_chunks_hybrid — as migration 011 plus the extra column,
    --     COALESCEd so keyword-only hits still report their true flag.
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
        is_front_matter BOOLEAN,
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
                dc.is_front_matter,
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
            COALESCE(v.is_front_matter, dc2.is_front_matter)      AS is_front_matter,
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
    'As migration 011, plus is_front_matter in the result set so the reranker can demote cover/TOC/credits chunks.';


    -- ============================================================
    -- B4) match_global_chunks_hybrid — as B3, global scope.
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
        is_front_matter BOOLEAN,
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
                dc.is_front_matter,
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
            COALESCE(v.is_front_matter, dc2.is_front_matter)      AS is_front_matter,
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
    'As migration 011, plus is_front_matter in the result set so the reranker can demote cover/TOC/credits chunks.';
