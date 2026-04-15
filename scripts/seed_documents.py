#!/usr/bin/env python3
# ============================================================
# scripts/seed_documents.py — Bulk document loader for admins
#
# WHY THIS SCRIPT EXISTS:
# Admins need to seed the global knowledge base with documents
# (PDF, DOCX, TXT) without calling the HTTP API manually.
# This script reads files from docs/admin/, checks for duplicates,
# and runs each file through the full ingestion pipeline
# (parse → chunk → embed → store in Supabase) under tenant_id='global'.
#
# HOW TO RUN (from project root):
#   PYTHONPATH=. python scripts/seed_documents.py
#
# With a custom folder:
#   PYTHONPATH=. python scripts/seed_documents.py --folder /path/to/docs
#
# This script is idempotent — safe to run multiple times.
# Files already processed (status='ready') are skipped.
# ============================================================

import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.supabase import get_supabase_client
from app.domain.documents.service import DocumentService
from app.core.logging import get_logger

logger = get_logger(__name__)

GLOBAL_TENANT_ID = "global"
DEFAULT_DOCS_FOLDER = Path(__file__).parent.parent / "docs" / "admin"
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt"}


def already_processed(db, filename: str) -> bool:
    """Return True if a 'ready' document with this filename already exists for the global tenant."""
    result = (
        db.table("documents")
        .select("id, status")
        .eq("tenant_id", GLOBAL_TENANT_ID)
        .eq("filename", filename)
        .eq("status", "ready")
        .execute()
    )
    return bool(result.data)


def seed_documents(folder: Path) -> None:
    if not folder.exists():
        logger.error(f"Folder not found: {folder}")
        sys.exit(1)

    files = [f for f in sorted(folder.iterdir()) if f.suffix.lower() in SUPPORTED_EXTENSIONS]

    if not files:
        logger.info(f"No PDF, DOCX, or TXT files found in: {folder}")
        return

    logger.info(f"Found {len(files)} file(s) in {folder}")

    db = get_supabase_client()
    service = DocumentService(db)

    processed = 0
    skipped = 0
    failed = 0

    for file_path in files:
        filename = file_path.name

        if already_processed(db, filename):
            logger.info(f"SKIP: '{filename}' already exists as a ready document in tenant=global")
            skipped += 1
            continue

        logger.info(f"PROCESSING: '{filename}'")
        try:
            file_bytes = file_path.read_bytes()
            doc = service.upload_document(
                tenant_id=GLOBAL_TENANT_ID,
                filename=filename,
                file_bytes=file_bytes,
            )
            service.process_document(str(doc.id))
            logger.info(f"OK: '{filename}' ingested successfully (doc_id={doc.id})")
            processed += 1
        except Exception as e:
            logger.error(f"FAILED: '{filename}' — {e}")
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
    args = parser.parse_args()

    seed_documents(args.folder)
