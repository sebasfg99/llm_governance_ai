-- ============================================================
-- 003_fix_ivfflat_probes.sql — Fix IVFFlat cluster invisibility
--
-- WHY THIS MIGRATION EXISTS:
-- The vector similarity search in RAG was silently dropping
-- results from documents with relatively few chunks. Runtime
-- evidence (debug session aa53d3):
--
--   - OWASP checklist: 31 chunks, target chunk similarity 0.52
--   - Production RPC returned 0 hits for that document
--   - RPC with ivfflat.probes=100 returned 3 hits (top sim 0.52)
--
-- The IVFFlat index created in 001_initial.sql uses lists=100
-- (100 clusters) but pgvector defaults ivfflat.probes=1, which
-- scans only 1 cluster per query. Small documents end up in
-- unprobed clusters and become invisible.
--
-- FIX:
-- Convert both match functions to PL/pgSQL and SET LOCAL
-- ivfflat.probes = 100 at the start. SET LOCAL only affects the
-- current transaction (the RPC call), so no session state leaks
-- to other queries.
--
-- WHY probes=100 (exhaustive) INSTEAD OF sqrt(lists)=10:
-- Empirical runs with probes=10 and probes=20 both failed to
-- surface a target document whose 31 chunks clustered together:
--   - probes=10: the target doc was never retrieved.
--   - probes=20: the target doc appeared only occasionally.
-- At this corpus size (low thousands of vectors) the cost of a
-- full scan is a few ms — negligible — while the recall benefit
-- is large. probes=lists=100 = exhaustive scan = matches the
-- behaviour of a brute-force cosine search, which is the ground
-- truth we validated in the debug scripts (top target similarity
-- ≈ 0.52).
--
-- WHEN TO REVISIT:
-- Once the corpus grows past ~10k vectors per tenant, reducing
-- probes back to sqrt(lists) or switching the index to HNSW
-- (pgvector ≥ 0.5) becomes worthwhile for latency reasons.
--
-- ROLLBACK:
-- Re-apply the function definitions from 001_initial.sql to
-- restore the SQL-language versions without probe tuning.
-- ============================================================

-- ------------------------------------------------------------
-- FUNCTION: match_chunks (tenant-specific)
-- ------------------------------------------------------------
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
LANGUAGE plpgsql
VOLATILE  -- Must be VOLATILE (not STABLE) to allow SET LOCAL inside
           -- the function body. PostgreSQL rejects SET in STABLE /
           -- IMMUTABLE functions with error 0A000 ("SET is not
           -- allowed in a non-volatile function"). VOLATILE only
           -- disables the planner's result cache for this call,
           -- which is harmless here since the query is executed
           -- fresh per RPC invocation anyway.
AS $$
BEGIN
    -- Inspect ALL 100 IVFFlat clusters per query (exhaustive scan).
    -- At this corpus size, the latency cost is negligible and it
    -- guarantees the same recall as a brute-force cosine search.
    -- SET LOCAL is scoped to the current transaction and does not
    -- persist across pooled connections.
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
    ORDER BY
        dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_chunks IS
  'Vector similarity search for RAG retrieval. Sets ivfflat.probes=100 (exhaustive) to guarantee recall at this corpus size.';


-- ------------------------------------------------------------
-- FUNCTION: match_global_chunks (shared library)
-- ------------------------------------------------------------
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
LANGUAGE plpgsql
VOLATILE  -- See match_chunks above for the same reason: SET LOCAL
           -- is not allowed in STABLE/IMMUTABLE functions.
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
    ORDER BY
        dc.embedding <=> query_embedding
    LIMIT match_count;
END;
$$;

COMMENT ON FUNCTION match_global_chunks IS
  'Vector similarity search over the global shared knowledge library, with optional country filter. Sets ivfflat.probes=100 (exhaustive) to guarantee recall at this corpus size.';
