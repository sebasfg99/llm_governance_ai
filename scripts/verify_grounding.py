#!/usr/bin/env python3
# ============================================================
# scripts/verify_grounding.py
#
# Quick grounding check for a single trace.
#
# Reads the full content of every chunk that was retrieved for a
# given trace_id and prints them so we can manually verify whether
# the LLM's response was actually supported by the corpus or was
# fabricated.
#
# Usage:
#   PYTHONPATH=. .venv/bin/python scripts/verify_grounding.py <trace_id>
# ============================================================

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.supabase import get_supabase_client


def main(trace_id: str) -> None:
    db = get_supabase_client()

    trace = (
        db.table("traces")
        .select(
            "id, user_query, response_text, retrieved_chunk_ids, "
            "model_name, latency_ms"
        )
        .eq("id", trace_id)
        .single()
        .execute()
        .data
    )
    if not trace:
        print(f"Trace {trace_id} not found")
        return

    print("=" * 70)
    print("TRACE")
    print("=" * 70)
    print(f"Query:    {trace['user_query']}")
    print(f"Model:    {trace['model_name']}")
    print(f"Latency:  {trace['latency_ms']}ms")
    print()
    print("RESPONSE:")
    print(f"  {trace['response_text']}")
    print()

    raw_ids = trace.get("retrieved_chunk_ids") or []
    chunk_ids: list[str] = []
    for entry in raw_ids:
        if isinstance(entry, str):
            chunk_ids.append(entry)
        elif isinstance(entry, dict):
            cid = entry.get("chunk_id") or entry.get("id")
            if cid:
                chunk_ids.append(cid)

    print("=" * 70)
    print(f"FULL CONTENT OF {len(chunk_ids)} RETRIEVED CHUNKS")
    print("=" * 70)
    chunks = (
        db.table("document_chunks")
        .select("id, document_id, section_ref, content")
        .in_("id", chunk_ids)
        .execute()
        .data
        or []
    )
    by_id = {c["id"]: c for c in chunks}
    for i, cid in enumerate(chunk_ids, start=1):
        c = by_id.get(cid)
        if not c:
            print(f"\n[{i}] {cid}  (chunk no longer exists)")
            continue
        print(f"\n[{i}] section_ref={c['section_ref']}  doc={c['document_id']}")
        print(c["content"])
        print("-" * 70)

    # Look for the literal phrase the LLM cited.
    target = "la organización debe documentar una política"
    print()
    print("=" * 70)
    print(f"GROUNDING CHECK — looking for: {target!r}")
    print("=" * 70)
    found = False
    for c in chunks:
        if target.lower() in (c["content"] or "").lower():
            print(f"  FOUND in chunk {c['id']} (section_ref={c['section_ref']})")
            found = True
    if not found:
        # Try a fuzzier check: the phrase with broken-word artefacts
        loose = "documentar una política"
        for c in chunks:
            if loose.lower() in (c["content"] or "").lower():
                print(
                    f"  FUZZY MATCH on '{loose}' in chunk {c['id']} "
                    f"(section_ref={c['section_ref']})"
                )
                found = True
    if not found:
        print("  NOT FOUND in any retrieved chunk — likely hallucination.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: verify_grounding.py <trace_id>")
        sys.exit(1)
    main(sys.argv[1])
