-- ============================================================
-- 004_projects_memory_framework.sql
--
-- Consolidates three concerns in a single migration:
--
--   A) Project expediente (L2 memory)
--        Table `projects` stores the structured state of every AI
--        project the tenant is governing: profile, risk register,
--        policy document. Traces and threads link back to a project
--        so that A/B/C capabilities share a common context.
--
--   B) Session memory (L1)
--        Tables `threads` + `messages` give the chat and the
--        conversational risk-intake persistent continuity across
--        calls to POST /ai/query. Without them every call is
--        amnesic.
--
--   C) Framework tagging
--        Documents are organised by normative framework in
--        docs/admin/<Framework>/... The `framework` column
--        propagates that metadata to documents and chunks so the
--        retrieval layer can optionally scope search (e.g. "only
--        ISO 42001 when drafting an AIMS policy").
--
-- All three are prerequisites for the risk assessment and policy
-- generation orchestrators added in the same sprint. Keeping them
-- in one SQL file means they always apply together — the Python
-- code assumes the full schema.
--
-- HOW TO RUN:
--   Supabase Dashboard → SQL Editor → paste and Run.
-- ============================================================


-- ============================================================
-- A) projects — L2 "expediente"
-- ============================================================
CREATE TABLE IF NOT EXISTS projects (
    id                 UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id          TEXT NOT NULL,
    name               TEXT NOT NULL,
    description        TEXT,

    -- Structured profile filled during risk intake (B1 + B2).
    -- Stored as JSONB for flexibility: fields evolve as prompts evolve
    -- and we don't want a migration every time we add an attribute.
    -- Expected keys (documented in app/domain/projects/models.py):
    --   project_name, ai_type, data_used, industry, affected_audience,
    --   autonomy_level, deployment, country, existing_vendor
    project_profile    JSONB NOT NULL DEFAULT '{}'::jsonb,

    -- Output of the risk orchestrator (B3). Shape documented in
    -- app/domain/risk/models.py → RiskRegister.
    risk_register      JSONB,

    -- Output of the policy orchestrator (C1). Shape documented in
    -- app/domain/policies/models.py → Policy.
    policy             JSONB,

    -- Output of the controls orchestrator (C2). Separate from policy
    -- because controls can be iterated independently.
    controls           JSONB,

    -- Lifecycle:
    --   draft            → profile incomplete
    --   intake_complete  → profile ready, risk not yet run
    --   risks_analyzed   → risk_register populated
    --   policy_generated → policy populated
    --   archived         → project retired
    status             TEXT NOT NULL DEFAULT 'draft',

    created_at         TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_projects_tenant_id
    ON projects (tenant_id);

CREATE INDEX IF NOT EXISTS idx_projects_status
    ON projects (status);

COMMENT ON TABLE projects IS
  'AI governance projects. Central "expediente" that risk, policies, threads and traces attach to.';


-- ============================================================
-- B) threads + messages — L1 session memory
-- ============================================================
CREATE TABLE IF NOT EXISTS threads (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id   TEXT NOT NULL,
    user_id     TEXT NOT NULL,

    -- Optional link to a project. NULL = free chat not tied to
    -- any project (e.g. a consultant asking about ISO 42001 before
    -- creating a project). When set, AIService hydrates the prompt
    -- with project_profile + risk_register excerpts.
    project_id  UUID REFERENCES projects(id) ON DELETE SET NULL,

    -- Which capability this thread serves:
    --   chatbot           → free Q&A
    --   risk_intake       → guided interview for the project profile
    --   executive_summary → one-off summarisation
    --   policy_generation → policy discovery
    -- The use_case scopes how AIService builds the system prompt.
    use_case    TEXT NOT NULL,

    -- Short human-readable label shown in the UI sidebar.
    -- Populated from the first user message (truncated) or edited.
    title       TEXT NOT NULL DEFAULT 'Nuevo hilo',

    status      TEXT NOT NULL DEFAULT 'open',   -- open | archived
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_threads_tenant_user
    ON threads (tenant_id, user_id, updated_at DESC);

CREATE INDEX IF NOT EXISTS idx_threads_project_id
    ON threads (project_id);

COMMENT ON TABLE threads IS
  'Chat threads. Group messages and provide persistent session memory for every AI use case.';


CREATE TABLE IF NOT EXISTS messages (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    thread_id   UUID NOT NULL REFERENCES threads(id) ON DELETE CASCADE,

    -- Who produced this message. "system" is reserved for system-injected
    -- notices (e.g. "risk register updated") and normally is not shown.
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),

    content     TEXT NOT NULL,

    -- When role='assistant', trace_id points to the trace that produced
    -- the message. This is the bridge that makes the "why did you say X?"
    -- lookup possible: from a message → trace → prompt + retrieved chunks.
    trace_id    UUID REFERENCES traces(id) ON DELETE SET NULL,

    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_messages_thread_created
    ON messages (thread_id, created_at);

COMMENT ON TABLE messages IS
  'Individual turns within a thread. Assistant messages reference the trace that generated them.';


-- ============================================================
-- Extend traces with project_id and thread_id
--
-- These are nullable so traces from the old pipeline (no project,
-- no thread) still work. Once the routes always pass thread_id they
-- can become NOT NULL in a later migration.
-- ============================================================
ALTER TABLE traces
    ADD COLUMN IF NOT EXISTS project_id UUID REFERENCES projects(id) ON DELETE SET NULL;

ALTER TABLE traces
    ADD COLUMN IF NOT EXISTS thread_id UUID REFERENCES threads(id) ON DELETE SET NULL;

CREATE INDEX IF NOT EXISTS idx_traces_project_id
    ON traces (project_id, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_traces_thread_id
    ON traces (thread_id, created_at);


-- ============================================================
-- C) Framework tagging on documents and chunks
--
-- Seed_documents.py reads docs/admin/<Framework>/<Country>/file.*.
-- We store that <Framework> slug on both documents and chunks so
-- the retrieval layer can filter by framework when the caller
-- passes framework_filter (e.g. ["iso_42001", "nist_airmf"]).
--
-- NULL framework means the document is not scoped to any framework
-- (general org documents, contracts, policies, etc.) and is always
-- eligible for retrieval.
-- ============================================================
ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS framework TEXT;

ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS framework TEXT;

CREATE INDEX IF NOT EXISTS idx_documents_framework
    ON documents (framework);

CREATE INDEX IF NOT EXISTS idx_document_chunks_framework
    ON document_chunks (framework);


-- ============================================================
-- Updated match_chunks — supports framework_filter
--
-- match_frameworks = NULL  → no framework restriction (current behaviour)
-- match_frameworks = []    → no framework restriction (empty array means "any")
-- match_frameworks = [...] → return chunks whose framework is in the list,
--                            plus chunks with NULL framework (always eligible)
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
    similarity      FLOAT
)
LANGUAGE plpgsql
VOLATILE
AS $$
BEGIN
    SET LOCAL ivfflat.probes = 100;

    RETURN QUERY
    SELECT
        dc.id,
        dc.document_id,
        dc.tenant_id,
        dc.content,
        dc.chunk_index,
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    WHERE
        dc.tenant_id = match_tenant_id
        AND dc.embedding IS NOT NULL
        AND (
            -- Framework filter: NULL / empty array → no restriction.
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
  'Vector similarity search for RAG retrieval, with optional framework filter. Sets ivfflat.probes=100.';


-- ============================================================
-- Updated match_global_chunks — supports framework_filter
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
    similarity      FLOAT
)
LANGUAGE plpgsql
VOLATILE
AS $$
BEGIN
    SET LOCAL ivfflat.probes = 100;

    RETURN QUERY
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
  'Vector similarity search over the global library, with optional country and framework filters. Sets ivfflat.probes=100.';
