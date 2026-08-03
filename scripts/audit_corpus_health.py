#!/usr/bin/env python3
# ============================================================
# scripts/audit_corpus_health.py — Corpus integrity check
#
# WHY THIS SCRIPT EXISTS:
# On 2026-07-29 a seed_documents.py --force run lost two documents
# to transient network errors mid-insert, and NOTHING surfaced it.
# The EU AI Act kept 200 of 493 chunks; GDPR kept 0 of 189. Both
# still advertised their old, correct-looking chunk_count, so the
# corpus looked complete from every angle except counting the rows.
#
# Retrieval kept serving the 200 surviving EU AI Act chunks as
# valid candidates, which is the dangerous part: questions about
# the rest of the regulation had no chance of being answered, and
# the system said nothing.
#
# This script counts the rows. It reports:
#
#   MISSING    — status='ready' but zero chunks. Invisible to
#                retrieval while claiming to be healthy.
#   MISMATCH   — chunk_count disagrees with the real row count.
#   NOT_READY  — status is not 'ready'; whatever chunks exist are
#                still being served, so the count matters.
#   GAPS       — chunk_index is not a contiguous 0..n-1 run, i.e.
#                the document has holes in the middle.
#
# Exit code is 1 when anything is reported, so it can gate a
# deploy or run in CI.
#
# HOW TO RUN (from project root):
#   PYTHONPATH=. .venv/bin/python scripts/audit_corpus_health.py
#   PYTHONPATH=. .venv/bin/python scripts/audit_corpus_health.py --verbose
# ============================================================

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.supabase import get_supabase_client

# PostgREST caps rows per response; chunk_index values are paged
# through in slices of this size so large documents are counted in
# full rather than silently truncated at the cap.
_PAGE_SIZE = 1000


def _chunk_indexes(db, document_id: str) -> list[int]:
    """Every chunk_index for a document, paged past the row cap."""
    indexes: list[int] = []
    page = 0
    while True:
        rows = (
            db.table("document_chunks")
            .select("chunk_index")
            .eq("document_id", document_id)
            .range(page * _PAGE_SIZE, page * _PAGE_SIZE + _PAGE_SIZE - 1)
            .execute()
            .data
            or []
        )
        indexes.extend(r["chunk_index"] for r in rows)
        if len(rows) < _PAGE_SIZE:
            return indexes
        page += 1


def audit(verbose: bool = False) -> int:
    db = get_supabase_client()

    docs = (
        db.table("documents")
        .select("id, filename, status, chunk_count, framework, tenant_id")
        .execute()
        .data
        or []
    )
    docs.sort(key=lambda d: d.get("filename") or "")

    findings: list[tuple[str, dict, str]] = []
    total_chunks = 0

    for doc in docs:
        indexes = _chunk_indexes(db, doc["id"])
        actual = len(indexes)
        total_chunks += actual
        claimed = doc.get("chunk_count")
        status = doc.get("status")

        problems: list[str] = []
        if status == "ready" and actual == 0:
            problems.append("MISSING: status=ready but 0 chunks stored")
        if claimed is not None and claimed != actual:
            problems.append(
                f"MISMATCH: chunk_count={claimed} but {actual} rows stored"
            )
        if status != "ready":
            problems.append(f"NOT_READY: status={status!r} with {actual} chunks stored")
        if indexes:
            expected = set(range(max(indexes) + 1))
            missing = sorted(expected - set(indexes))
            if missing:
                preview = missing[:5]
                problems.append(
                    f"GAPS: {len(missing)} missing chunk_index values "
                    f"(e.g. {preview}) up to {max(indexes)}"
                )

        if problems:
            findings.append((doc["filename"], doc, "; ".join(problems)))
        elif verbose:
            print(f"  ok        {actual:>5} chunks  {doc['filename'][:56]}")

    print()
    print("=" * 78)
    print(f"  Corpus health: {len(docs)} documents, {total_chunks} chunks")
    print("=" * 78)

    if not findings:
        print("\n  No inconsistencies found.")
        return 0

    print(f"\n  {len(findings)} document(s) need attention:\n")
    for filename, doc, problem in findings:
        print(f"  - {filename}")
        print(f"      id        : {doc['id']}")
        print(f"      framework : {doc.get('framework')}")
        print(f"      problem   : {problem}")
        print()

    print("  Reprocess a document with:")
    print("    PYTHONPATH=. .venv/bin/python -c \\")
    print("      \"from app.db.supabase import get_supabase_client; \\")
    print("       from app.domain.documents.service import DocumentService; \\")
    print("       DocumentService(get_supabase_client()).process_document('<id>')\"")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report documents whose stored chunks disagree with their metadata"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Also list healthy documents",
    )
    args = parser.parse_args()
    raise SystemExit(audit(verbose=args.verbose))


if __name__ == "__main__":
    main()
