-- ============================================================
-- 008_industry_codes.sql
-- Adds industry_codes column to documents and document_chunks
-- for Layer-3 sector document filtering.
--
-- Layer rules:
--   industry_codes = '{ALL}' (or NULL)  → always retrieved
--   industry_codes = '{bank,cooperative}' → only for matching institutions
--
-- RUN:
--   Apply in Supabase SQL editor or via Supabase CLI.
--   After applying, re-run seed_documents.py (--force) to tag
--   existing sector documents.
-- ============================================================

ALTER TABLE documents
    ADD COLUMN IF NOT EXISTS industry_codes TEXT[] DEFAULT ARRAY['ALL'];

ALTER TABLE document_chunks
    ADD COLUMN IF NOT EXISTS industry_codes TEXT[] DEFAULT ARRAY['ALL'];

-- Backfill sector documents already in the DB by framework slug.
-- SUGEF and CONASSIF apply to bank + cooperative institutions.
UPDATE documents
    SET industry_codes = ARRAY['bank', 'cooperative']
    WHERE framework IN ('sugef_10_07', 'sugef_18_16', 'sugef_24_22', 'conassif')
      AND (industry_codes IS NULL OR industry_codes = ARRAY['ALL']);

-- Propagate to their chunks.
UPDATE document_chunks dc
    SET industry_codes = d.industry_codes
    FROM documents d
    WHERE dc.document_id = d.id
      AND d.framework IN ('sugef_10_07', 'sugef_18_16', 'sugef_24_22', 'conassif');
