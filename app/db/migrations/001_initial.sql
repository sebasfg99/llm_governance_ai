-- ============================================================
-- 001_initial.sql — Complete database schema for ai-core
--
-- WHY THIS FILE EXISTS:
-- This is the single SQL script that creates the entire
-- database schema. Run it once against your Supabase database
-- (or local Docker Postgres) to set up all tables, indexes,
-- and helper functions.
--
-- HOW TO RUN:
--   Option A — Supabase Dashboard:
--     1. Go to your Supabase project
--     2. Click "SQL Editor" in the left sidebar
--     3. Paste this entire file and click "Run"
--
--   Option B — Local Docker Postgres:
--     psql -h localhost -U postgres -d aicore -f app/db/migrations/001_initial.sql
--
-- ARCHITECTURE NOTES:
--   - ALL tables (except `tenants`) have a tenant_id column
--   - tenant_id = 'global' is reserved for the shared library
--   - Tenant isolation is enforced at the application layer
--     (each query WHERE tenant_id = ?)
--   - TODO: Add Row Level Security (RLS) policies before production
-- ============================================================


-- ============================================================
-- EXTENSIONS
-- Enable pgvector — the Postgres extension that adds vector
-- storage and similarity search. Must be enabled before using
-- the `vector` data type.
-- ============================================================
CREATE EXTENSION IF NOT EXISTS vector;

-- Enable UUID generation — we use UUIDs as primary keys
-- because they're globally unique and don't reveal row counts
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";


-- ============================================================
-- TABLE: tenants
--
-- Represents the top-level organizations using the platform.
-- Every other table links back to a tenant via tenant_id.
--
-- In the MVP, tenant records are created manually or via a
-- seed script. Future: add a self-service onboarding flow.
-- ============================================================
CREATE TABLE IF NOT EXISTS tenants (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        TEXT NOT NULL,                  -- Display name, e.g. "Acme Corp"
    slug        TEXT UNIQUE NOT NULL,           -- URL-safe identifier, e.g. "acme-corp"
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Pre-insert the dev tenant for local development
-- ON CONFLICT DO NOTHING makes this script re-runnable (idempotent)
INSERT INTO tenants (id, name, slug)
VALUES ('00000000-0000-0000-0000-000000000001', 'Dev Tenant', 'dev-tenant')
ON CONFLICT (slug) DO NOTHING;

-- Pre-insert the global tenant — used for the shared knowledge library
-- tenant_id = 'global' means: available to ALL tenants in queries
INSERT INTO tenants (id, name, slug)
VALUES ('00000000-0000-0000-0000-000000000000', 'Global Library', 'global')
ON CONFLICT (slug) DO NOTHING;

COMMENT ON TABLE tenants IS 'Top-level organizations. Every resource belongs to a tenant.';


-- ============================================================
-- TABLE: documents
--
-- Tracks every uploaded document — metadata only.
-- The actual file bytes live on disk (or S3 in production).
-- The extracted text chunks live in document_chunks.
--
-- document lifecycle:
--   pending → processing → ready  (success path)
--   pending → processing → failed (error path)
-- ============================================================
CREATE TABLE IF NOT EXISTS documents (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id       TEXT NOT NULL,              -- Which tenant owns this document
    filename        TEXT NOT NULL,              -- Original uploaded filename
    file_type       TEXT NOT NULL,              -- "pdf" or "docx"
    file_path       TEXT NOT NULL,              -- Path on disk: uploads/{tenant_id}/{filename}
    status          TEXT NOT NULL DEFAULT 'pending',
                                                -- pending | processing | ready | failed
    error_message   TEXT,                       -- Set if status = 'failed'; stores the error
    page_count      INTEGER,                    -- Number of pages (populated after processing)
    chunk_count     INTEGER,                    -- Number of chunks produced (populated after processing)
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    processed_at    TIMESTAMPTZ                 -- Set when processing completes (success or failure)
);

-- Index for fast tenant-scoped document queries
CREATE INDEX IF NOT EXISTS idx_documents_tenant_id
    ON documents (tenant_id);

-- Index for filtering by status (e.g., find all pending docs to process)
CREATE INDEX IF NOT EXISTS idx_documents_status
    ON documents (status);

COMMENT ON TABLE documents IS 'Document metadata. File bytes are stored on disk; text chunks in document_chunks.';
COMMENT ON COLUMN documents.status IS 'Processing lifecycle: pending → processing → ready | failed';


-- ============================================================
-- TABLE: document_chunks
--
-- Stores the text chunks extracted from each document,
-- along with their vector embeddings for similarity search.
--
-- This is the core RAG table. The vector similarity search
-- (implemented as the match_chunks function below) queries
-- this table to find relevant content for a user's question.
--
-- WHY STORE CHUNKS SEPARATELY FROM DOCUMENTS:
-- A document (e.g., a 50-page PDF) is split into many small
-- chunks (~500 tokens each). We store each chunk separately
-- so we can return only the relevant parts of a document,
-- not the entire thing, which would exceed the LLM's context.
-- ============================================================
CREATE TABLE IF NOT EXISTS document_chunks (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id     UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    tenant_id       TEXT NOT NULL,              -- Denormalized from document for faster queries
    content         TEXT NOT NULL,              -- The actual text of this chunk
    chunk_index     INTEGER NOT NULL,           -- Position within the document (0-based)
    embedding       vector(1536),               -- 1536-dim vector from text-embedding-3-small
                                                -- NULL until embeddings are generated
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for filtering chunks by tenant before vector search
-- This is critical: without it, every vector search scans ALL tenants' data
CREATE INDEX IF NOT EXISTS idx_document_chunks_tenant_id
    ON document_chunks (tenant_id);

-- Index for filtering chunks by document (e.g., to delete when a doc is removed)
CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
    ON document_chunks (document_id);

-- IVFFlat index for approximate nearest-neighbor vector search
-- WHY IVFFlat: It's the standard pgvector index for cosine similarity.
-- It's much faster than exact search on large datasets, with minimal accuracy loss.
-- lists=100 is appropriate for up to ~1M vectors; increase for larger datasets.
-- TODO: Tune the `lists` parameter as data grows
CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
    ON document_chunks USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

COMMENT ON TABLE document_chunks IS 'Text chunks from documents with vector embeddings for RAG retrieval.';
COMMENT ON COLUMN document_chunks.embedding IS '1536-dim vector from OpenAI text-embedding-3-small. Used for cosine similarity search.';


-- ============================================================
-- TABLE: prompts
--
-- The prompt library — stores versioned system prompts
-- for different use cases (Q&A, summarization, drafting, etc.)
--
-- WHY STORE PROMPTS IN THE DB:
-- Storing prompts in the DB means you can update them without
-- deploying new code. You can A/B test different versions,
-- roll back to a previous version, and track what prompt
-- was active for any given trace.
--
-- Prompt versioning:
--   - Multiple rows can share the same (name, use_case) pair
--   - is_active = true marks the currently-used version
--   - Only ONE prompt per use_case should have is_active = true
-- ============================================================
CREATE TABLE IF NOT EXISTS prompts (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name        TEXT NOT NULL,                  -- Human-readable name, e.g. "Default RAG Prompt"
    version     INTEGER NOT NULL DEFAULT 1,     -- Version number, increments on each update
    use_case    TEXT NOT NULL,                  -- The use case this prompt serves
                                                -- e.g. "default_rag", "document_summary", "risk_analysis"
    content     TEXT NOT NULL,                  -- The full system prompt text
    is_active   BOOLEAN NOT NULL DEFAULT true,  -- Only one active prompt per use_case at a time
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for fast lookup by use_case (the most common query pattern)
CREATE INDEX IF NOT EXISTS idx_prompts_use_case
    ON prompts (use_case, is_active);

COMMENT ON TABLE prompts IS 'Versioned system prompt library. One active prompt per use_case at a time.';


-- ============================================================
-- TABLE: traces
--
-- The full audit log of every AI request-response cycle.
-- Every call to POST /ai/query creates one trace record.
--
-- WHY TRACE EVERYTHING:
--   1. Debugging: reproduce exactly what happened for any query
--   2. Fine-tuning: export high-quality (prompt, response) pairs
--   3. Billing: measure usage per tenant
--   4. Compliance: demonstrate what the AI said and when
--   5. Evaluation: feed traces into the eval pipeline
-- ============================================================
CREATE TABLE IF NOT EXISTS traces (
    id                  UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tenant_id           TEXT NOT NULL,          -- Which tenant made this request
    user_id             TEXT NOT NULL,          -- Which user within that tenant
    prompt_text         TEXT NOT NULL,          -- The final assembled prompt sent to the LLM
                                                -- (system prompt + context + user query)
    user_query          TEXT NOT NULL,          -- The raw user question (before augmentation)
    use_case            TEXT NOT NULL,          -- Which use case was invoked
    model_provider      TEXT NOT NULL,          -- "openai" or "gemini"
    model_name          TEXT NOT NULL,          -- Specific model, e.g. "gpt-4o-mini"
    retrieved_chunk_ids JSONB,                  -- Array of chunk UUIDs that were retrieved
    retrieved_content   TEXT,                   -- Concatenated text of retrieved chunks (for auditing)
    response_text       TEXT NOT NULL,          -- The LLM's response
    response_mode       TEXT NOT NULL DEFAULT 'text',
                                                -- "text" or "structured"
    latency_ms          INTEGER,                -- End-to-end response time in milliseconds
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Index for tenant-scoped trace feed (newest first)
CREATE INDEX IF NOT EXISTS idx_traces_tenant_created
    ON traces (tenant_id, created_at DESC);

-- Index for user-level trace lookup
CREATE INDEX IF NOT EXISTS idx_traces_user_id
    ON traces (user_id);

COMMENT ON TABLE traces IS 'Complete audit log of every AI query. Used for debugging, fine-tuning, and compliance.';


-- ============================================================
-- TABLE: feedback
--
-- User ratings and comments on AI responses.
-- Each feedback record links to a trace (the specific response
-- the user is rating).
--
-- WHY COLLECT FEEDBACK:
--   1. Identify bad responses for review and retraining
--   2. Build a dataset of good responses for fine-tuning
--   3. Track quality trends over time
-- ============================================================
CREATE TABLE IF NOT EXISTS feedback (
    id          UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trace_id    UUID NOT NULL REFERENCES traces(id) ON DELETE CASCADE,
    tenant_id   TEXT NOT NULL,                  -- Denormalized for fast tenant-level queries
    user_id     TEXT NOT NULL,                  -- Who submitted the feedback
    rating      INTEGER NOT NULL,               -- 1 (bad) to 5 (excellent)
    comment     TEXT,                           -- Optional free-text comment
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Prevent a user from rating the same response twice
    CONSTRAINT feedback_unique_per_trace_user UNIQUE (trace_id, user_id),

    -- Enforce valid rating range
    CONSTRAINT feedback_rating_range CHECK (rating BETWEEN 1 AND 5)
);

-- Index for tenant-level feedback queries
CREATE INDEX IF NOT EXISTS idx_feedback_tenant_id
    ON feedback (tenant_id);

-- Index for looking up feedback by trace
CREATE INDEX IF NOT EXISTS idx_feedback_trace_id
    ON feedback (trace_id);

COMMENT ON TABLE feedback IS 'User ratings on AI responses. 1-5 scale. Linked to traces.';


-- ============================================================
-- TABLE: evaluations
--
-- Automated quality checks run after each AI response.
-- Each evaluation record links to a trace.
--
-- In the MVP, evaluations are rule-based (no API calls):
--   - grounding_score: does the response reference retrieved text?
--   - length_ok: is the response an appropriate length?
--   - has_citations: does the response cite sources?
--   - passed: did the response meet the overall quality bar?
--
-- TODO: Add LLM-as-judge evaluation (optional, costs per eval)
-- ============================================================
CREATE TABLE IF NOT EXISTS evaluations (
    id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    trace_id        UUID NOT NULL REFERENCES traces(id) ON DELETE CASCADE,

    -- Heuristic grader results
    grounding_score FLOAT,          -- 0.0 to 1.0: word overlap between response and retrieved chunks
    length_ok       BOOLEAN,        -- True if response word count is within acceptable range
    has_citations   BOOLEAN,        -- True if response contains citation markers like [1] or "Source:"

    -- Overall pass/fail based on the above
    passed          BOOLEAN NOT NULL DEFAULT false,

    -- Optional: store the full grader output for debugging
    grader_details  JSONB,          -- {"grounding": 0.42, "word_count": 150, ...}

    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),

    -- Only one evaluation record per trace
    CONSTRAINT evaluations_unique_per_trace UNIQUE (trace_id)
);

-- Index for looking up evaluations by trace
CREATE INDEX IF NOT EXISTS idx_evaluations_trace_id
    ON evaluations (trace_id);

-- Index for filtering by pass/fail (e.g., find all failed evals)
CREATE INDEX IF NOT EXISTS idx_evaluations_passed
    ON evaluations (passed);

COMMENT ON TABLE evaluations IS 'Automated quality checks on AI responses. Rule-based heuristics in MVP.';


-- ============================================================
-- FUNCTION: match_chunks
--
-- Postgres function that performs vector similarity search.
-- Called via Supabase RPC: db.rpc("match_chunks", {...})
--
-- WHY A FUNCTION INSTEAD OF A DIRECT QUERY:
-- The Supabase Python client's query builder doesn't support
-- the pgvector <=> operator directly. Using an RPC function
-- lets us write the vector query in SQL and call it cleanly
-- from Python.
--
-- HOW IT WORKS:
--   1. Takes a query_embedding vector as input
--   2. Filters chunks by tenant_id (tenant isolation)
--   3. Computes cosine similarity: 1 - (embedding <=> query_embedding)
--      (1 = identical, 0 = unrelated, -1 = opposite)
--   4. Returns the top match_count most similar chunks
--
-- PARAMETERS:
--   query_embedding  — the vector for the user's question
--   match_tenant_id  — only search this tenant's chunks
--   match_count      — how many chunks to return (default 5)
--
-- USAGE FROM PYTHON:
--   result = db.rpc("match_chunks", {
--       "query_embedding": [0.1, 0.2, ...],  # 1536 floats
--       "match_tenant_id": "acme-corp",
--       "match_count": 5
--   }).execute()
-- ============================================================
CREATE OR REPLACE FUNCTION match_chunks(
    query_embedding  vector(1536),
    match_tenant_id  TEXT,
    match_count      INT DEFAULT 5
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
STABLE  -- This function always returns the same result for the same inputs (helps optimizer)
AS $$
    SELECT
        dc.id,
        dc.document_id,
        dc.tenant_id,
        dc.content,
        dc.chunk_index,
        -- Cosine similarity: 1 minus cosine distance
        -- Higher value = more similar to the query
        1 - (dc.embedding <=> query_embedding) AS similarity
    FROM document_chunks dc
    WHERE
        -- CRITICAL: Only search within the specified tenant's chunks
        -- This is the primary tenant isolation mechanism for RAG
        dc.tenant_id = match_tenant_id
        -- Only return chunks that have embeddings generated
        AND dc.embedding IS NOT NULL
    ORDER BY
        -- Order by cosine distance (ascending = most similar first)
        dc.embedding <=> query_embedding
    LIMIT match_count;
$$;

COMMENT ON FUNCTION match_chunks IS 'Vector similarity search for RAG retrieval. Returns top-k chunks by cosine similarity, filtered by tenant.';


-- ============================================================
-- FUNCTION: match_global_chunks
--
-- Same as match_chunks but searches the global shared library.
-- Called separately and merged with tenant-specific results
-- in the retrieval service when include_global = true.
-- ============================================================
CREATE OR REPLACE FUNCTION match_global_chunks(
    query_embedding  vector(1536),
    match_count      INT DEFAULT 3  -- Fewer global chunks than tenant chunks by default
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
        -- 'global' is the reserved tenant_id for the shared library
        dc.tenant_id = 'global'
        AND dc.embedding IS NOT NULL
    ORDER BY
        dc.embedding <=> query_embedding
    LIMIT match_count;
$$;

COMMENT ON FUNCTION match_global_chunks IS 'Vector similarity search over the global shared knowledge library.';
