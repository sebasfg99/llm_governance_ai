#!/usr/bin/env python3
# ============================================================
# scripts/migrate_legacy_govai.py — one-time re-ingest of
# GovAI's legacy regulatory_documents into the Python sidecar.
#
# WHY THIS SCRIPT EXISTS:
# GovAI previously stored regulatory documents in two places:
#   1. `regulatory_documents` table (metadata)
#   2. `regulatory-docs` Supabase Storage bucket (PDFs)
#
# When the GovAI ↔ `llm_governance_ai` sidecar cutover ran the
# rename migration, those tables became `legacy_regulatory_documents`
# and `legacy_document_chunks`. The Python sidecar uses fresh
# `documents` + `document_chunks` tables with OpenAI 1536-dim
# embeddings — the legacy Gemini 768-dim chunks can't be migrated
# directly, so we re-ingest from source.
#
# This script:
#   1. Reads every legacy_regulatory_documents row (active=true).
#   2. Skips rows whose source filename + tenant + country already
#      exist as a `documents` row (idempotent — safe to re-run).
#   3. Downloads the original PDF from the legacy `regulatory-docs`
#      Storage bucket.
#   4. Uploads it to the new `rag-documents` bucket via the existing
#      DocumentService.upload_document() — same code path Admins
#      will use once the Edge Function flag flips on.
#   5. Runs DocumentService.process_document() to chunk + embed
#      with text-embedding-3-small.
#
# HOW TO RUN (from llm_governance_ai/):
#   # Dry-run first — lists what would be ingested, no writes.
#   PYTHONPATH=. python scripts/migrate_legacy_govai.py --dry-run
#
#   # Real run.
#   PYTHONPATH=. python scripts/migrate_legacy_govai.py
#
#   # Limit to one institution (test before full backfill).
#   PYTHONPATH=. python scripts/migrate_legacy_govai.py \
#       --institution-id <uuid> --limit 5
#
# The script is idempotent — re-running after a partial failure
# only ingests the documents that weren't picked up the first time.
# ============================================================

import sys
import os
import argparse
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.supabase import get_supabase_client
from app.domain.documents.service import DocumentService
from app.core.logging import get_logger

logger = get_logger(__name__)


# Source bucket the GovAI app uses today. The legacy
# `regulatory_documents.storage_path` column points into this bucket.
LEGACY_STORAGE_BUCKET = "regulatory-docs"

# GovAI tenant scheme: shared/global library documents (no
# institution_id on the legacy row) use tenant_id='global' in the
# Python schema; institution-scoped ones use institution.id::text.
GLOBAL_TENANT_ID = "global"


# Country code → slug mapping. Mirrors src/lib/country-slug-map.ts.
# Adding a country requires updating BOTH this dict AND the TS
# registry so backfill and live ingest stay in sync.
COUNTRY_CODE_TO_SLUG: dict[str, str] = {
    "CR": "costa-rica",
    "PA": "panama",
    "DO": "republica-dominicana",
    # "GT": "guatemala",
    # "SV": "el-salvador",
    # "HN": "honduras",
    # "NI": "nicaragua",
    # Add new institutions here, then in src/lib/country-slug-map.ts.
}


def map_country_codes_to_slug(codes: Optional[list[str]]) -> Optional[str]:
    """
    Convert the legacy `country_codes` array into a single Python
    country slug.

    The legacy row carries an array because a single PDF could
    apply to multiple jurisdictions ("CR" + "PA"). The Python
    schema picks ONE slug per document: we use the first known
    code. Documents tagged with 'ALL' or no code are treated as
    global (country=NULL).

    Returns None for global documents.
    """
    if not codes:
        return None

    for code in codes:
        if not code:
            continue
        upper = code.strip().upper()
        if upper == "ALL":
            return None  # explicit global tag
        slug = COUNTRY_CODE_TO_SLUG.get(upper)
        if slug:
            return slug

    # Unknown codes — log so the operator can extend the map.
    logger.warning(
        f"No country mapping for codes {codes!r}. "
        "Document will be ingested as global (country=NULL)."
    )
    return None


def resolve_tenant_id(legacy_row: dict, fallback_institution_id: Optional[str]) -> str:
    """
    Decide which tenant_id the new `documents` row should live under.

    The legacy schema does not have a per-row institution_id (the
    shared library is one big table). We treat every row as global
    unless --institution-id was passed to this script, in which case
    the operator is explicitly back-filling that institution's docs.
    """
    if fallback_institution_id:
        return fallback_institution_id
    return GLOBAL_TENANT_ID


def already_ingested(
    db,
    *,
    tenant_id: str,
    filename: str,
    country: Optional[str],
    framework: Optional[str],
) -> Optional[str]:
    """
    Return the existing `documents.id` (as str) if this file has
    already been re-ingested for the same tenant+country+framework
    combination, else None.

    Match key is intentionally loose (filename + country + framework)
    because the legacy rows don't carry a stable foreign key into
    the new schema — we'd rather skip a duplicate than ingest the
    same PDF twice.
    """
    query = (
        db.table("documents")
        .select("id, status")
        .eq("tenant_id", tenant_id)
        .eq("filename", filename)
        .limit(1)
    )
    if country is None:
        query = query.is_("country", "null")
    else:
        query = query.eq("country", country)
    if framework is None:
        query = query.is_("framework", "null")
    else:
        query = query.eq("framework", framework)

    result = query.execute()
    return result.data[0]["id"] if result.data else None


def download_legacy_pdf(db, storage_path: str) -> Optional[bytes]:
    """
    Pull the PDF bytes out of the legacy regulatory-docs bucket.
    Returns None if the object is missing — the calling loop logs
    and skips so a single missing file doesn't abort the backfill.
    """
    if not storage_path:
        return None
    try:
        return db.storage.from_(LEGACY_STORAGE_BUCKET).download(storage_path)
    except Exception as exc:
        logger.warning(
            f"Storage download failed for '{storage_path}' "
            f"in bucket '{LEGACY_STORAGE_BUCKET}': {exc}"
        )
        return None


def fetch_legacy_documents(db, *, limit: Optional[int]) -> list[dict]:
    """Read active legacy regulatory_documents rows."""
    query = (
        db.table("legacy_regulatory_documents")
        .select("*")
        .eq("active", True)
        .order("created_at", desc=False)
    )
    if limit:
        query = query.limit(limit)
    return query.execute().data or []


def migrate(
    *,
    dry_run: bool = False,
    institution_id: Optional[str] = None,
    limit: Optional[int] = None,
) -> dict:
    """
    Run the migration. Returns a dict with counts so callers
    (tests, the operator) can assert on the outcome.
    """
    db = get_supabase_client()
    service = DocumentService(db)

    rows = fetch_legacy_documents(db, limit=limit)
    logger.info(
        f"Found {len(rows)} active legacy regulatory_documents "
        f"(limit={limit}, institution_id={institution_id or 'global'})."
    )
    if dry_run:
        logger.info("--dry-run: no writes will be made.")

    processed = 0
    skipped = 0
    missing = 0
    failed = 0

    for row in rows:
        legacy_id = row.get("id")
        filename = row.get("source_filename") or row.get("title") or "document.pdf"
        framework = row.get("framework")
        country = map_country_codes_to_slug(row.get("country_codes"))
        storage_path = row.get("storage_path")
        tenant_id = resolve_tenant_id(row, institution_id)
        label = (
            f"id={legacy_id} "
            f"file='{filename}' "
            f"tenant='{tenant_id}' "
            f"country={country or 'global'} "
            f"framework={framework or 'any'}"
        )

        # Idempotency: skip if the same file already exists in the
        # new schema with the same (tenant, country, framework).
        existing_id = already_ingested(
            db,
            tenant_id=tenant_id,
            filename=filename,
            country=country,
            framework=framework,
        )
        if existing_id:
            logger.info(f"SKIP (already ingested as {existing_id}): {label}")
            skipped += 1
            continue

        if dry_run:
            logger.info(f"WOULD INGEST: {label}")
            processed += 1
            continue

        file_bytes = download_legacy_pdf(db, storage_path)
        if file_bytes is None:
            logger.warning(f"MISSING source PDF for {label}")
            missing += 1
            continue

        try:
            doc = service.upload_document(
                tenant_id=tenant_id,
                filename=filename,
                file_bytes=file_bytes,
                country=country,
                framework=framework,
            )
            service.process_document(str(doc.id))
            logger.info(f"OK: {label} → new_id={doc.id}")
            processed += 1
        except Exception as exc:
            logger.error(f"FAILED: {label} — {exc}")
            failed += 1

    summary = {
        "total": len(rows),
        "processed": processed,
        "skipped": skipped,
        "missing": missing,
        "failed": failed,
    }
    logger.info(
        "Migration complete: "
        f"{summary['processed']} processed, "
        f"{summary['skipped']} skipped, "
        f"{summary['missing']} missing source, "
        f"{summary['failed']} failed."
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Re-ingest GovAI's legacy regulatory_documents into the "
            "Python sidecar's documents + document_chunks tables."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List what would be ingested without writing anything.",
    )
    parser.add_argument(
        "--institution-id",
        type=str,
        default=None,
        help=(
            "Target tenant_id for ingestion. Defaults to 'global'. "
            "Pass an institution UUID to migrate that institution's "
            "private docs instead."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N legacy rows (useful for staged backfills).",
    )
    args = parser.parse_args()

    migrate(
        dry_run=args.dry_run,
        institution_id=args.institution_id,
        limit=args.limit,
    )
