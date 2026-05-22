#!/usr/bin/env python3
# ============================================================
# scripts/seed_documents.py — Bulk document loader for admins
#
# WHY THIS SCRIPT EXISTS:
# Admins need to seed the global knowledge base with documents
# (PDF, DOCX, TXT) without calling the HTTP API manually.
# This script recursively reads files from docs/admin/, detects
# the country from the folder structure, and runs each file
# through the full ingestion pipeline under tenant_id='global'.
#
# FOLDER STRUCTURE RULES:
#   docs/admin/{topic}/{country}/file.pdf  → country-specific document
#   docs/admin/{topic}/file.pdf            → global document (all countries)
#
# COUNTRY NORMALIZATION:
#   "Costa Rica"          → "costa-rica"
#   "República Dominicana" → "republica-dominicana"
#   "Panamá"              → "panama"
#
# HOW TO RUN (from project root):
#   PYTHONPATH=. python scripts/seed_documents.py
#
# With a custom folder:
#   PYTHONPATH=. python scripts/seed_documents.py --folder /path/to/docs
#
# To re-chunk every existing document with the section-aware
# chunker (required after migration 005):
#   PYTHONPATH=. python scripts/seed_documents.py --force
#
# This script is idempotent — safe to run multiple times.
# Files already processed (status='ready') with the same filename
# and country are skipped unless --force is set.
#
# PREREQUISITES FOR --force AFTER MIGRATION 005:
#   1. Run app/db/migrations/005_section_ref_hnsw.sql in Supabase
#      (adds section_ref column, drops the ivfflat index, creates
#      the HNSW index, recreates match_chunks/match_global_chunks).
#   2. Then run this script with --force. The DocumentService will
#      use chunk_text_with_sections() and persist section_ref for
#      every chunk, so the LLM's chunk header gains the
#      "Section: <ref>" attribution.
# ============================================================

import sys
import os
import argparse
import unicodedata
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.supabase import get_supabase_client
from app.domain.documents.service import DocumentService
from app.core.logging import get_logger

logger = get_logger(__name__)

GLOBAL_TENANT_ID = "global"
DEFAULT_DOCS_FOLDER = Path(__file__).parent.parent / "docs" / "admin"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


# Maps the topic folder (docs/admin/<Topic>/...) to the framework slug
# stored on documents.framework. The slugs match the ones that
# AIQueryRequest.framework_filter accepts (see plan section 5 +
# scripts/seed_prompts.py use_case == risk_analysis expectations).
#
# The normalization is tolerant: we uppercase/strip accents/whitespace
# on the folder name before lookup, so minor renamings don't break
# the mapping.
FRAMEWORK_FOLDER_MAP = {
    "ISO 42001": "iso_42001",
    "NIST": "nist_airmf",
    "EU ACT": "eu_ai_act",
    "GDPR": "gdpr",
    "OWASP": "owasp_llm_top10",
    "MITRE-ATLAS": "mitre_atlas",
    "PCI": "pci_dss",
    "CIBERSEGURIDAD": "cybersecurity",
    "GOBIERNO CORPORATIVO": "corporate_governance",
    "PROTECCION DE DATOS PERSONALES": "local_data_protection",
}


def _normalize_topic_key(name: str) -> str:
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    return ascii_str.strip().upper()


def framework_for_topic(topic_name: str) -> str | None:
    """Look up the framework slug for a topic folder name."""
    key = _normalize_topic_key(topic_name)
    return FRAMEWORK_FOLDER_MAP.get(key)


def normalize_country(name: str) -> str:
    """
    Normalize a country folder name to a URL-safe slug.

    Examples:
        "Costa Rica"           → "costa-rica"
        "República Dominicana" → "republica-dominicana"
        "Panamá"               → "panama"
        "El Salvador"          → "el-salvador"
    """
    # Strip accents by decomposing unicode and keeping only ASCII chars
    nfkd = unicodedata.normalize("NFKD", name)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    # Lowercase and replace spaces with hyphens
    return ascii_str.lower().strip().replace(" ", "-")


def find_existing_document(db, filename: str, country: str | None) -> dict | None:
    """Return the most recent document record with this filename and country, or None."""
    query = (
        db.table("documents")
        .select("id, status")
        .eq("tenant_id", GLOBAL_TENANT_ID)
        .eq("filename", filename)
        .order("created_at", desc=True)
        .limit(1)
    )
    if country is None:
        query = query.is_("country", "null")
    else:
        query = query.eq("country", country)

    result = query.execute()
    return result.data[0] if result.data else None


def collect_files(folder: Path) -> list[tuple[Path, str | None, str | None]]:
    """
    Recursively scan a folder and return (file_path, country, framework) tuples.

    Structure:
      folder/{topic}/{country}/file.ext  → country = normalize(country_folder_name)
      folder/{topic}/file.ext            → country = None (global)
      folder/file.ext                    → country = None, framework = None

    The framework slug is derived from {topic} via FRAMEWORK_FOLDER_MAP.
    Unknown topics fall back to framework=None (framework-agnostic).
    """
    files: list[tuple[Path, str | None, str | None]] = []

    # Files directly under folder/ (no topic) → framework-agnostic, global.
    # Handled before walking topics so we don't miss them.
    for item in sorted(folder.iterdir()):
        if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append((item, None, None))

    for topic_dir in sorted(folder.iterdir()):
        if not topic_dir.is_dir() or topic_dir.name.startswith("."):
            continue

        framework = framework_for_topic(topic_dir.name)
        if framework is None:
            logger.warning(
                f"Unknown topic folder '{topic_dir.name}' — documents will be "
                "ingested with framework=NULL (framework-agnostic)."
            )

        for item in sorted(topic_dir.iterdir()):
            if item.name.startswith("."):
                continue

            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                # File directly under topic folder → framework-tagged, global country.
                files.append((item, None, framework))

            elif item.is_dir():
                # Subdirectory = country folder
                country = normalize_country(item.name)
                for file_path in sorted(item.iterdir()):
                    if (
                        file_path.is_file()
                        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
                        and not file_path.name.startswith(".")
                    ):
                        files.append((file_path, country, framework))

    return files


def seed_documents(folder: Path, force: bool = False) -> None:
    if not folder.exists():
        logger.error(f"Folder not found: {folder}")
        sys.exit(1)

    files = collect_files(folder)

    if not files:
        logger.info(f"No PDF, DOCX, or TXT files found in: {folder}")
        return

    logger.info(f"Found {len(files)} file(s) in {folder}")
    if force:
        logger.info("--force: ready documents will be reprocessed with current settings")

    db = get_supabase_client()
    service = DocumentService(db)

    processed = 0
    skipped = 0
    failed = 0

    for file_path, country, framework in files:
        filename = file_path.name
        country_label = country or "global"
        framework_label = framework or "any"
        label = f"{country_label}/{framework_label}"

        existing = find_existing_document(db, filename, country)

        if existing and existing["status"] == "ready" and not force:
            logger.info(f"SKIP: '{filename}' [{label}] already exists as ready")
            skipped += 1
            continue

        try:
            if existing:
                logger.info(
                    f"REPROCESS: '{filename}' [{label}] "
                    f"(prev status={existing['status']})"
                )
                # Patch the framework on the document row BEFORE reprocessing.
                # process_document() reads doc.framework via get_document() and
                # propagates it to every chunk — if the row still has framework=NULL
                # (set by the ALTER TABLE default in migration 004) all chunks will
                # inherit NULL even though we now know the correct slug.
                db.table("documents").update(
                    {"framework": framework}
                ).eq("id", existing["id"]).execute()
                service.process_document(existing["id"])
                logger.info(f"OK: '{filename}' [{label}] reprocessed (doc_id={existing['id']})")
            else:
                logger.info(f"PROCESSING: '{filename}' [{label}]")
                file_bytes = file_path.read_bytes()
                doc = service.upload_document(
                    tenant_id=GLOBAL_TENANT_ID,
                    filename=filename,
                    file_bytes=file_bytes,
                    country=country,
                    framework=framework,
                )
                service.process_document(str(doc.id))
                logger.info(f"OK: '{filename}' [{label}] ingested (doc_id={doc.id})")
            processed += 1
        except Exception as e:
            logger.error(f"FAILED: '{filename}' [{label}] — {e}")
            failed += 1

    logger.info(
        f"\nSeeding complete: {processed} processed, {skipped} skipped, {failed} failed."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Seed global knowledge base with documents from a local folder."
    )
    parser.add_argument(
        "--folder",
        type=Path,
        default=DEFAULT_DOCS_FOLDER,
        help=f"Path to folder containing documents to ingest (default: {DEFAULT_DOCS_FOLDER})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reprocess all documents, including those already marked as ready",
    )
    args = parser.parse_args()

    seed_documents(args.folder, force=args.force)
