-- ============================================================
-- 006_rag_governance.sql — RAG Governance Hardening
--
-- This migration extends the existing schema with everything
-- needed for enterprise-grade governance, audit, and security.
-- It NEVER drops or modifies existing columns; only adds.
--
-- WHAT THIS MIGRATION ADDS:
--
--   A) Cost + tokens + risk on traces
--      Each AI call now carries the cost in USD, the token
--      counts, the per-response risk_level, and a flag for
--      legal/compliance disclaimers.
--
--   B) Eight evaluation metrics on `evaluations`
--      retrieval_precision, groundedness_score,
--      context_adherence_score, citation_score,
--      completeness_score, instruction_adherence_score,
--      hallucination, should_have_said_idk +
--      claim counts and the LLM-judge raw JSON.
--
--   C) `response_risks` table (1-1 with traces)
--      Per-response risk register: level, confidence,
--      reasons, retrieval confidence.
--
--   D) Hybrid retrieval (vector + tsvector + RRF)
--      `content_tsv` GENERATED column on document_chunks
--      + GIN index. Two new RPCs return both vector and
--      keyword ranks so the application can run
--      Reciprocal Rank Fusion.
--
--   E) `rag_metrics_v` view
--      Pre-aggregated metrics for /metrics/rag.
--
-- HOW TO RUN:
--   Supabase Dashboard → SQL Editor → paste and Run.
--
-- NOTE: This migration is idempotent. Every ALTER uses IF NOT
--       EXISTS and every CREATE uses OR REPLACE / IF NOT EXISTS.
-- ============================================================


-- ============================================================
-- A) traces — cost, tokens, risk, client_id alias
-- ============================================================

ALTER TABLE traces
    ADD COLUMN IF NOT EXISTS prompt_tokens       INTEGER,
    ADD COLUMN IF NOT EXISTS completion_tokens   INTEGER,
    ADD COLUMN IF NOT EXISTS cost_estimate_usd   NUMERIC(10, 6),
    ADD COLUMN IF NOT EXISTS risk_level          TEXT,
    ADD COLUMN IF NOT EXISTS requires_disclaimer BOOLEAN DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS client_id           TEXT;

-- client_id is a denormalized alias of tenant_id. The spec uses
-- the term `client_id` while the code base uses `tenant_id`. We
-- keep both columns in sync via a trigger so existing code keeps
-- working and new code can filter by either name.
CREATE OR REPLACE FUNCTION sync_client_id_from_tenant()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.client_id IS NULL THEN
        NEW.client_id := NEW.tenant_id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_traces_sync_client_id ON traces;
CREATE TRIGGER trg_traces_sync_client_id
    BEFORE INSERT OR UPDATE ON traces
    FOR EACH ROW EXECUTE FUNCTION sync_client_id_from_tenant();

-- Backfill existing rows.
UPDATE traces SET client_id = tenant_id WHERE client_id IS NULL;

CREATE INDEX IF NOT EXISTS idx_traces_client_id_created
    ON traces (client_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_traces_risk_level
    ON traces (risk_level) WHERE risk_level IS NOT NULL;

COMMENT ON COLUMN traces.cost_estimate_usd IS
  'Estimated USD cost of this LLM call computed by app/domain/observability/cost.py';
COMMENT ON COLUMN traces.client_id IS
  'Alias of tenant_id maintained by trigger for spec compatibility';


-- ============================================================
-- B) evaluations — 8 governance metrics + judge raw output
-- ============================================================

ALTER TABLE evaluations
    ADD COLUMN IF NOT EXISTS retrieval_precision         NUMERIC(5, 4),
    ADD COLUMN IF NOT EXISTS groundedness_score          NUMERIC(5, 4),
    ADD COLUMN IF NOT EXISTS context_adherence_score     NUMERIC(5, 4),
    ADD COLUMN IF NOT EXISTS citation_score              NUMERIC(5, 4),
    ADD COLUMN IF NOT EXISTS completeness_score          NUMERIC(5, 4),
    ADD COLUMN IF NOT EXISTS instruction_adherence_score NUMERIC(5, 4),
    ADD COLUMN IF NOT EXISTS hallucination               BOOLEAN,
    ADD COLUMN IF NOT EXISTS should_have_said_idk        BOOLEAN,
    ADD COLUMN IF NOT EXISTS claims_total                INTEGER,
    ADD COLUMN IF NOT EXISTS claims_supported            INTEGER,
    ADD COLUMN IF NOT EXISTS judge_raw_json              JSONB,
    ADD COLUMN IF NOT EXISTS evaluator_version           TEXT DEFAULT 'v2.governance';

CREATE INDEX IF NOT EXISTS idx_evaluations_hallucination
    ON evaluations (hallucination) WHERE hallucination IS NOT NULL;

CREATE INDEX IF NOT EXISTS idx_evaluations_context_adherence
    ON evaluations (context_adherence_score);

COMMENT ON COLUMN evaluations.context_adherence_score IS
  'Fraction of response claims supported by retrieved chunks. Computed by LLM-as-judge in app/domain/evals/llm_judge.py.';
COMMENT ON COLUMN evaluations.hallucination IS
  'True when the LLM judge found at least one unsupported claim that was stated as fact.';
COMMENT ON COLUMN evaluations.should_have_said_idk IS
  'True when the model should have abstained ("I do not know") given the retrieved context.';


-- ============================================================
-- C) response_risks — per-response risk register
-- ============================================================

CREATE TABLE IF NOT EXISTS response_risks (
    id                     UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trace_id               UUID NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
    tenant_id              TEXT NOT NULL,
    client_id              TEXT NOT NULL,

    -- One of: low | medium | high
    risk_level             TEXT NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),

    -- Aggregate confidence in [0,1]. Higher = more confident the response is safe.
    confidence_score       NUMERIC(5, 4) NOT NULL,

    -- Confidence in retrieval. Derived from average chunk similarity and chunk count.
    retrieval_confidence   NUMERIC(5, 4),

    -- List of human-readable reasons explaining the level. Stored as JSON for
    -- structured filtering in dashboards.
    risk_reasons           JSONB NOT NULL DEFAULT '[]'::jsonb,

    created_at             TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_response_risks_trace_id
    ON response_risks (trace_id);

CREATE INDEX IF NOT EXISTS idx_response_risks_tenant_level
    ON response_risks (tenant_id, risk_level, created_at DESC);

CREATE UNIQUE INDEX IF NOT EXISTS uq_response_risks_trace_id
    ON response_risks (trace_id);

COMMENT ON TABLE response_risks IS
  'Per-response risk register. One row per trace. Computed by app/domain/response_risk/scorer.py from evaluation + guardrails + metadata.';


-- ============================================================
-- D) Hybrid retrieval — content_tsv + GIN index + RPCs
-- ============================================================

-- The 'simple' configuration is language-agnostic and keeps the
-- migration portable across Spanish/English corpora. Switch to
-- 'spanish' or 'english' if the corpus becomes monolingual.
ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS content_tsv tsvector
    GENERATED ALWAYS AS (to_tsvector('simple', coalesce(content, ''))) STORED;

CREATE INDEX IF NOT EXISTS idx_document_chunks_content_tsv
    ON document_chunks USING GIN (content_tsv);

COMMENT ON COLUMN document_chunks.content_tsv IS
  'tsvector for hybrid (vector + keyword) retrieval. Auto-generated from content.';


-- Drop any prior versions so the new signature is unambiguous.
DROP FUNCTION IF EXISTS match_chunks_hybrid(vector, TEXT, TEXT, INT, TEXT[]);
DROP FUNCTION IF EXISTS match_global_chunks_hybrid(vector, TEXT, INT, TEXT, TEXT[]);


-- ------------------------------------------------------------
-- match_chunks_hybrid — tenant-scoped hybrid search
--
-- Returns BOTH a vector_rank (1 = best vector match) and a
-- keyword_rank (1 = best keyword match). The application layer
-- combines them via Reciprocal Rank Fusion:
--
--     rrf(d) = sum( 1 / (k + rank_i(d)) )
--
-- where k=60 is the standard RRF constant.
--
-- NULL ranks indicate the chunk did NOT appear in that side of
-- the search. The application treats them as "infinite rank"
-- (zero contribution to RRF).
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

    -- websearch_to_tsquery accepts natural-language queries safely.
    -- It strips operators, supports phrases, and never raises on
    -- malformed input — perfect for untrusted user text.
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

COMMENT ON FUNCTION match_chunks_hybrid IS
  'Hybrid retrieval (vector + keyword tsvector). Returns ranks so the application can combine them via RRF.';


-- ------------------------------------------------------------
-- match_global_chunks_hybrid — global library hybrid search
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

COMMENT ON FUNCTION match_global_chunks_hybrid IS
  'Hybrid retrieval over the global library. Returns vector + keyword ranks for RRF in the application.';


-- ============================================================
-- E) rag_metrics_v — pre-aggregated metrics view
--
-- One row per (client_id, day) with averages and counts that
-- /metrics/rag returns. Materialising it as a view (not a table)
-- keeps it always-fresh; if performance becomes an issue we can
-- swap to MATERIALIZED VIEW + scheduled REFRESH.
-- ============================================================
CREATE OR REPLACE VIEW rag_metrics_v AS
SELECT
    t.client_id,
    DATE_TRUNC('day', t.created_at)               AS day,
    COUNT(*)                                       AS total_calls,
    AVG(t.latency_ms)                              AS avg_latency_ms,
    SUM(t.cost_estimate_usd)                       AS total_cost_usd,
    AVG(t.cost_estimate_usd)                       AS avg_cost_usd,
    AVG(e.retrieval_precision)                     AS avg_retrieval_precision,
    AVG(e.groundedness_score)                      AS avg_groundedness,
    AVG(e.context_adherence_score)                 AS avg_context_adherence,
    AVG(e.citation_score)                          AS avg_citation_score,
    AVG(e.completeness_score)                      AS avg_completeness,
    AVG(e.instruction_adherence_score)             AS avg_instruction_adherence,
    SUM(CASE WHEN e.hallucination THEN 1 ELSE 0 END)::FLOAT
        / NULLIF(COUNT(e.hallucination), 0)        AS hallucination_rate,
    SUM(CASE WHEN r.risk_level = 'high' THEN 1 ELSE 0 END)   AS high_risk_count,
    SUM(CASE WHEN r.risk_level = 'medium' THEN 1 ELSE 0 END) AS medium_risk_count,
    SUM(CASE WHEN r.risk_level = 'low' THEN 1 ELSE 0 END)    AS low_risk_count
FROM traces t
LEFT JOIN evaluations e   ON e.trace_id = t.id
LEFT JOIN response_risks r ON r.trace_id = t.id
GROUP BY t.client_id, DATE_TRUNC('day', t.created_at);

COMMENT ON VIEW rag_metrics_v IS
  'Per-client per-day aggregated metrics for /metrics/rag. Uses LEFT JOINs so traces without evaluations are still counted.';
