#!/usr/bin/env python3
# ============================================================
# scripts/backfill_reference_flags.py — Classify existing chunks
#
# WHY THIS SCRIPT EXISTS:
# Migration 011 adds document_chunks.is_reference_list, defaulted to
# FALSE. Every existing row therefore says "ordinary content" until
# something classifies it.
#
# The obvious way to fix that would be to reprocess each document
# through DocumentService.process_document(), but that DELETES and
# re-inserts every chunk, which means re-embedding the whole corpus:
# ~3,600 OpenAI calls to recompute vectors that have not changed.
#
# Nothing about this feature changes chunk TEXT — the chunker's
# boundary logic is untouched, so the same document produces the
# same chunks in the same order. Only the new flag is missing. So
# this script re-parses each document, re-chunks it in memory,
# verifies the recomputed text still matches what is stored, and
# UPDATEs is_reference_list alone. Embeddings are never read or
# written.
#
# The text check is what makes that safe: if a chunk's recomputed
# text does not match the stored text, the document's boundaries
# HAVE drifted (someone changed chunk_size, or the parser output
# moved), the chunk_index join is no longer meaningful, and the
# document is skipped and reported instead of being written with
# flags that belong to different text.
#
# HOW TO RUN (from project root):
#   PYTHONPATH=. .venv/bin/python scripts/backfill_reference_flags.py --dry-run
#   PYTHONPATH=. .venv/bin/python scripts/backfill_reference_flags.py
#   PYTHONPATH=. .venv/bin/python scripts/backfill_reference_flags.py --document-id <uuid>
#
# Requires migration 011_reference_list_flag.sql to be applied first.
# ============================================================

from __future__ import annotations

import argparse
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.core.logging import get_logger
from app.db.supabase import get_supabase_client
from app.domain.documents.chunking import chunk_text_with_sections
from app.domain.documents.parser_docx import parse_docx
from app.domain.documents.parser_pdf import parse_pdf
from app.domain.documents.parser_txt import parse_txt

logger = get_logger(__name__)

_PARSERS = {"pdf": parse_pdf, "docx": parse_docx, "txt": parse_txt}


def _parse_document(db, doc: dict) -> str | None:
    """Download a document from storage and return its text."""
    parser = _PARSERS.get((doc.get("file_type") or "").lower())
    if parser is None:
        logger.warning(f"{doc['filename']}: unsupported file_type — skipping")
        return None
    if not doc.get("file_path"):
        logger.warning(f"{doc['filename']}: no file_path — skipping")
        return None

    blob = db.storage.from_(settings.storage_bucket).download(doc["file_path"])
    fd, tmp = tempfile.mkstemp(suffix=f".{doc['file_type']}", prefix="reflag_")
    with os.fdopen(fd, "wb") as fh:
        fh.write(blob)
    try:
        text, _pages = parser(tmp)
        return text
    finally:
        os.unlink(tmp)


def _has_flag_column(db) -> bool:
    """
    Whether migration 011 has been applied. Probed rather than assumed
    so --dry-run can preview the classification BEFORE the migration
    is run, which is the natural order of operations.
    """
    try:
        db.table("document_chunks").select("is_reference_list").limit(1).execute()
        return True
    except Exception as e:  # noqa: BLE001
        if "is_reference_list" in str(e):
            return False
        raise


def backfill(document_id: str | None = None, dry_run: bool = False) -> None:
    db = get_supabase_client()

    has_column = _has_flag_column(db)
    if not has_column:
        if not dry_run:
            raise SystemExit(
                "document_chunks.is_reference_list does not exist.\n"
                "Apply app/db/migrations/011_reference_list_flag.sql "
                "(Supabase Dashboard -> SQL Editor) before writing flags.\n"
                "You can preview the classification now with --dry-run."
            )
        print(
            "NOTE: migration 011 is not applied yet — previewing the "
            "classification only.\n"
        )

    query = db.table("documents").select(
        "id, filename, file_path, file_type, chunk_count"
    )
    if document_id:
        query = query.eq("id", document_id)
    docs = query.execute().data or []
    print(f"Scanning {len(docs)} document(s)\n")

    total_flagged = 0
    total_updated = 0
    skipped: list[str] = []

    for doc in docs:
        columns = "id, chunk_index, content"
        if has_column:
            columns += ", is_reference_list"
        stored = (
            db.table("document_chunks")
            .select(columns)
            .eq("document_id", doc["id"])
            .order("chunk_index")
            .execute()
            .data
            or []
        )
        if not stored:
            continue

        try:
            text = _parse_document(db, doc)
        except Exception as e:  # noqa: BLE001 — one bad document must not stop the run
            logger.warning(f"{doc['filename']}: parse failed ({e}) — skipping")
            skipped.append(f"{doc['filename']} (parse failed)")
            continue
        if text is None:
            skipped.append(f"{doc['filename']} (unsupported)")
            continue

        chunks = chunk_text_with_sections(text)

        # Guard: the chunk_index join is only valid while the recomputed
        # text still matches what is stored. See the header.
        if len(chunks) != len(stored):
            msg = f"{doc['filename']} (count drift: stored={len(stored)} now={len(chunks)})"
            logger.warning(f"{msg} — skipping")
            skipped.append(msg)
            continue
        drifted = [
            row["chunk_index"]
            for row in stored
            if row["content"] != chunks[row["chunk_index"]].content
        ]
        if drifted:
            msg = f"{doc['filename']} (text drift at {drifted[:5]})"
            logger.warning(f"{msg} — skipping")
            skipped.append(msg)
            continue

        # Only write rows whose flag actually changes.
        changes = [
            (row["id"], chunks[row["chunk_index"]].is_reference_list)
            for row in stored
            if bool(row.get("is_reference_list"))
            != chunks[row["chunk_index"]].is_reference_list
        ]
        flagged = [i for i, c in enumerate(chunks) if c.is_reference_list]
        total_flagged += len(flagged)

        if flagged or changes:
            print(
                f"  {doc['filename'][:52]:52} "
                f"{len(flagged):>3} reference chunk(s) {flagged[:8]}"
                f"{' ...' if len(flagged) > 8 else ''}"
            )

        if dry_run:
            continue
        for chunk_id, value in changes:
            db.table("document_chunks").update(
                {"is_reference_list": value}
            ).eq("id", chunk_id).execute()
        total_updated += len(changes)

    print()
    print(f"{'Would flag' if dry_run else 'Flagged'}: {total_flagged} chunk(s) "
          f"as reference lists")
    if not dry_run:
        print(f"Rows updated: {total_updated}")
    if skipped:
        print(f"\nSkipped {len(skipped)} document(s):")
        for s in skipped:
            print(f"  - {s}")
        print("\nA skipped document keeps its current flags. Reprocess it "
              "through the normal pipeline if its chunking really has changed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Classify existing chunks as reference lists (no re-embedding)"
    )
    parser.add_argument("--document-id", help="Limit to one document")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing",
    )
    args = parser.parse_args()
    backfill(document_id=args.document_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
