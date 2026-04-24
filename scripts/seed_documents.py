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
# This script is idempotent — safe to run multiple times.
# Files already processed (status='ready') with the same filename
# and country are skipped.
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


def collect_files(folder: Path) -> list[tuple[Path, str | None]]:
    """
    Recursively scan a folder and return (file_path, country) pairs.

    Structure:
      folder/{topic}/{country}/file.ext  → country = normalize(country_folder_name)
      folder/{topic}/file.ext            → country = None (global)

    Returns a sorted list of (Path, country_or_None) tuples.
    """
    files: list[tuple[Path, str | None]] = []

    for topic_dir in sorted(folder.iterdir()):
        if not topic_dir.is_dir() or topic_dir.name.startswith("."):
            continue

        for item in sorted(topic_dir.iterdir()):
            if item.name.startswith("."):
                continue

            if item.is_file() and item.suffix.lower() in SUPPORTED_EXTENSIONS:
                # File directly under topic folder → global document
                files.append((item, None))

            elif item.is_dir():
                # Subdirectory = country folder
                country = normalize_country(item.name)
                for file_path in sorted(item.iterdir()):
                    if (
                        file_path.is_file()
                        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
                        and not file_path.name.startswith(".")
                    ):
                        files.append((file_path, country))

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

    for file_path, country in files:
        filename = file_path.name
        country_label = country or "global"

        existing = find_existing_document(db, filename, country)

        if existing and existing["status"] == "ready" and not force:
            logger.info(f"SKIP: '{filename}' [{country_label}] already exists as ready")
            skipped += 1
            continue

        try:
            if existing:
                logger.info(f"REPROCESS: '{filename}' [{country_label}] (prev status={existing['status']})")
                service.process_document(existing["id"])
                logger.info(f"OK: '{filename}' [{country_label}] reprocessed (doc_id={existing['id']})")
            else:
                logger.info(f"PROCESSING: '{filename}' [{country_label}]")
                file_bytes = file_path.read_bytes()
                doc = service.upload_document(
                    tenant_id=GLOBAL_TENANT_ID,
                    filename=filename,
                    file_bytes=file_bytes,
                    country=country,
                )
                service.process_document(str(doc.id))
                logger.info(f"OK: '{filename}' [{country_label}] ingested (doc_id={doc.id})")
            processed += 1
        except Exception as e:
            logger.error(f"FAILED: '{filename}' [{country_label}] — {e}")
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
