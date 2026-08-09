#!/usr/bin/env python3
# ============================================================
# scripts/reprocess_front_matter.py — Apply the front-matter cut
#
# WHY THIS SCRIPT EXISTS:
# The chunker now detects a document's body-start heading and forces
# a hard chunk cut there, flagging everything before it as
# is_front_matter (see app/domain/documents/front_matter.py). Stored
# chunks were produced before that logic existed, so documents WITH a
# detectable boundary must be reprocessed for their chunk boundaries
# and flags to change. Documents WITHOUT a boundary chunk identically
# to before and are skipped — no wasted embeddings.
#
# Unlike backfill_reference_flags.py (which only wrote a flag), this
# RE-CHUNKS: boundaries move, so the affected documents are re-embedded
# via the normal process_document() pipeline. That pipeline is the
# hardened one (insert-new -> verify -> delete-old), so a mid-run
# failure leaves the previous generation intact.
#
# CONSEQUENCE TO BE AWARE OF: chunk_index values shift for reprocessed
# documents. Chunk indexes quoted in earlier diagnostics/docs (e.g.
# "ISO 42001 #69/#70 are the bibliography") refer to the pre-cut
# chunking. is_reference_list is recomputed by the chunker on the new
# boundaries, so the flags stay correct automatically.
#
# HOW TO RUN (from project root):
#   PYTHONPATH=. .venv/bin/python scripts/reprocess_front_matter.py --dry-run
#   PYTHONPATH=. .venv/bin/python scripts/reprocess_front_matter.py
#   PYTHONPATH=. .venv/bin/python scripts/reprocess_front_matter.py --document-id <uuid>
#
# Requires migration 012_front_matter_flag.sql for the real run
# (--dry-run works before it: it only parses and reports).
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
from app.domain.documents.front_matter import find_body_start
from app.domain.documents.parser_docx import parse_docx
from app.domain.documents.parser_pdf import parse_pdf
from app.domain.documents.parser_txt import parse_txt
from app.domain.documents.service import DocumentService

logger = get_logger(__name__)

_PARSERS = {"pdf": parse_pdf, "docx": parse_docx, "txt": parse_txt}


def _parse_document(db, doc: dict) -> str | None:
    parser = _PARSERS.get((doc.get("file_type") or "").lower())
    if parser is None or not doc.get("file_path"):
        return None
    blob = db.storage.from_(settings.storage_bucket).download(doc["file_path"])
    fd, tmp = tempfile.mkstemp(suffix=f".{doc['file_type']}", prefix="fmcut_")
    with os.fdopen(fd, "wb") as fh:
        fh.write(blob)
    try:
        text, _pages = parser(tmp)
        return text
    finally:
        os.unlink(tmp)


def run(document_id: str | None = None, dry_run: bool = False) -> None:
    db = get_supabase_client()
    service = DocumentService(db)

    query = db.table("documents").select(
        "id, filename, file_path, file_type, status, chunk_count"
    )
    if document_id:
        query = query.eq("id", document_id)
    docs = query.execute().data or []
    print(f"Scanning {len(docs)} document(s)\n")
    print(f"{'documento':54} {'boundary':>9} {'chunks':>7} {'flagged':>8}")
    print("-" * 84)

    to_reprocess: list[dict] = []
    skipped: list[str] = []

    for doc in sorted(docs, key=lambda d: d["filename"]):
        try:
            text = _parse_document(db, doc)
        except Exception as e:  # noqa: BLE001 — one bad doc must not stop the scan
            skipped.append(f"{doc['filename']} (parse failed: {e})")
            continue
        if text is None:
            skipped.append(f"{doc['filename']} (unsupported)")
            continue

        offset = find_body_start(text)
        if offset is None:
            continue  # chunking unchanged -> nothing to do

        chunks = chunk_text_with_sections(text)
        flagged = [i for i, c in enumerate(chunks) if c.is_front_matter]
        pct = offset / len(text) * 100
        print(f"{doc['filename'][:54]:54} {pct:>8.1f}% {len(chunks):>7} "
              f"{str(flagged)[:20]:>8}")
        to_reprocess.append(doc)

    print(f"\n{len(to_reprocess)} document(s) have a boundary and would be "
          f"reprocessed; {len(docs) - len(to_reprocess) - len(skipped)} "
          f"unchanged; {len(skipped)} skipped")
    for s in skipped:
        print(f"  skipped: {s}")

    if dry_run or not to_reprocess:
        if dry_run:
            print("\nDry run — nothing written.")
        return

    print("\nReprocessing through the hardened pipeline "
          "(insert-new -> verify -> delete-old)...")
    ok = failed = 0
    for doc in to_reprocess:
        try:
            result = service.process_document(doc["id"])
            print(f"  ok    {doc['filename'][:52]:52} "
                  f"chunks={result.chunk_count}")
            ok += 1
        except Exception as e:  # noqa: BLE001
            print(f"  FAIL  {doc['filename'][:52]:52} {e}")
            failed += 1
    print(f"\nDone: {ok} reprocessed, {failed} failed.")
    if failed:
        print("Failed documents keep their previous chunks (hardened "
              "persistence). Run scripts/audit_corpus_health.py to verify.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reprocess documents whose front-matter boundary is detectable"
    )
    parser.add_argument("--document-id", help="Limit to one document")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would change without writing")
    args = parser.parse_args()
    run(document_id=args.document_id, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
