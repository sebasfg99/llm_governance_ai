#!/usr/bin/env python3
# ============================================================
# scripts/diagnose_iso42001_anexo_a.py
#
# Diagnostic script to answer: "Does the corpus actually contain
# the list of controls from ISO 42001 Anexo A, and if so, can we
# find them?"
#
# Runs four checks:
#   1. Inventory: how many chunks per ISO 42001 document, by
#      section_ref.
#   2. Direct keyword scan: find chunks whose content matches
#      patterns like "A.2.2", "A.6.1.1", "Politica de IA".
#   3. Hybrid retrieval rehearsal: simulate the production query
#      using the new RPC and print top hits.
#   4. Print the chunks that ACTUALLY contain a control identifier
#      so we can verify the chunker did not mangle them.
#
# Run with:
#   PYTHONPATH=. .venv/bin/python scripts/diagnose_iso42001_anexo_a.py
# ============================================================

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.supabase import get_supabase_client


SEPARATOR = "=" * 70


def _print(title: str) -> None:
    print()
    print(SEPARATOR)
    print(f"  {title}")
    print(SEPARATOR)


def main() -> None:
    db = get_supabase_client()

    # ---------- 1. Inventory ----------
    _print("1. ISO 42001 documents in the corpus")
    docs = (
        db.table("documents")
        .select("id, filename, framework, tenant_id, country")
        .eq("framework", "iso_42001")
        .execute()
        .data
        or []
    )
    if not docs:
        # Try a looser filter — maybe the framework slug differs.
        docs = (
            db.table("documents")
            .select("id, filename, framework, tenant_id")
            .ilike("filename", "%42001%")
            .execute()
            .data
            or []
        )
    if not docs:
        print("  (no documents matching ISO 42001)")
        return

    for d in docs:
        chunks = (
            db.table("document_chunks")
            .select("id", count="exact")
            .eq("document_id", d["id"])
            .execute()
        )
        n = getattr(chunks, "count", None) or len(chunks.data or [])
        print(
            f"  - {d['filename']}  framework={d.get('framework')}  "
            f"tenant={d.get('tenant_id')}  chunks={n}"
        )

    doc_ids = [d["id"] for d in docs]

    # ---------- 2. Section_ref breakdown ----------
    _print("2. section_ref distribution (only ISO 42001 chunks)")
    chunks = (
        db.table("document_chunks")
        .select("id, document_id, section_ref")
        .in_("document_id", doc_ids)
        .execute()
        .data
        or []
    )
    by_section: dict[str, int] = {}
    for c in chunks:
        key = c.get("section_ref") or "(no section_ref)"
        by_section[key] = by_section.get(key, 0) + 1

    for ref, n in sorted(by_section.items(), key=lambda x: -x[1])[:30]:
        print(f"  {n:5d}  {ref}")

    # ---------- 3. Direct keyword scan ----------
    _print("3. Chunks whose content mentions a control identifier")
    patterns = [
        "A.2.2",
        "A.2.3",
        "A.6.1",
        "A.6.2",
        "A.7.1",
        "A.8.1",
        "A.9.1",
        "objetivo de control",
        "Política de IA",
        "Politica de IA",
    ]
    for pat in patterns:
        result = (
            db.table("document_chunks")
            .select("id, document_id, section_ref, content")
            .in_("document_id", doc_ids)
            .ilike("content", f"%{pat}%")
            .limit(3)
            .execute()
            .data
            or []
        )
        print(f"\n  pattern: '{pat}' -> {len(result)} hit(s)")
        for r in result:
            preview = (r["content"] or "")[:160].replace("\n", " ")
            print(f"     [{r['section_ref']}] {preview}...")

    # ---------- 4. Hybrid retrieval rehearsal ----------
    _print("4. Hybrid retrieval rehearsal (vector + tsvector + RRF)")
    try:
        from app.domain.documents.embeddings import embed_single

        query = "Lista los controles del Anexo A de ISO 42001 sobre política de IA, recursos y gestión del ciclo de vida"
        emb = embed_single(query)

        # 4a) tenant-scoped hybrid (only useful if a tenant other than 'global' is set)
        tenant_ids = {d["tenant_id"] for d in docs}
        for tenant in tenant_ids:
            try:
                rpc = db.rpc(
                    "match_chunks_hybrid",
                    {
                        "query_embedding": emb,
                        "query_text": query,
                        "match_tenant_id": tenant,
                        "match_count": 10,
                        "match_frameworks": ["iso_42001"],
                    },
                ).execute()
                rows = rpc.data or []
                print(
                    f"\n  tenant={tenant}: {len(rows)} hits via match_chunks_hybrid"
                )
                for r in rows[:8]:
                    sim = r.get("similarity") or 0.0
                    kw = r.get("keyword_score") or 0.0
                    sec = r.get("section_ref")
                    preview = (r.get("content") or "")[:120].replace("\n", " ")
                    print(
                        f"     sim={sim:.3f}  kw={kw:.3f}  [{sec}]  {preview}..."
                    )
            except Exception as e:
                print(f"  tenant={tenant}: match_chunks_hybrid failed -> {e}")

        # 4b) global hybrid — the RPC actually used at runtime when the
        # corpus lives under tenant_id='global'. This is what your
        # production query hits.
        try:
            rpc = db.rpc(
                "match_global_chunks_hybrid",
                {
                    "query_embedding": emb,
                    "query_text": query,
                    "match_count": 10,
                    "match_country": None,
                    "match_frameworks": ["iso_42001"],
                },
            ).execute()
            rows = rpc.data or []
            print(
                f"\n  GLOBAL: {len(rows)} hits via match_global_chunks_hybrid"
            )
            for r in rows[:8]:
                sim = r.get("similarity") or 0.0
                kw = r.get("keyword_score") or 0.0
                sec = r.get("section_ref")
                vr = r.get("vector_rank")
                kr = r.get("keyword_rank")
                preview = (r.get("content") or "")[:140].replace("\n", " ")
                print(
                    f"     sim={sim:.3f}  kw={kw:.3f}  vr={vr} kr={kr}  [{sec}]  {preview}..."
                )
        except Exception as e:
            print(f"  GLOBAL: match_global_chunks_hybrid failed -> {e}")

    except Exception as e:
        print(f"  could not run hybrid rehearsal: {e}")


if __name__ == "__main__":
    main()
