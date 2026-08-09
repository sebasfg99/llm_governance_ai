#!/usr/bin/env python3
# ============================================================
# scripts/diagnose_retrieval_owasp.py
#
# Where does the OWASP GenAI Solutions Reference Guide get lost?
#
# THE OBSERVATION:
# Seguridad Q5 asks for the references cited by that guide, and the
# retrieved top-5 contains ZERO chunks from it — 3 from PCI-DSS-LA,
# 1 from SAFEAI, 1 from a different OWASP document.
#
# THE QUESTION THIS ANSWERS:
# Is the document dropped at the final cut (it reaches the candidate
# pool but loses the top-5 race), or does it never enter the pool at
# all? The fix is completely different in each case:
#   - reaches the pool  -> per-framework diversity cap
#   - never reaches it  -> an embedding/similarity problem
#
# HOW IT WORKS:
# Replays the production pipeline stage by stage using the REAL
# functions from app/domain/retrieval/service.py, capturing the
# intermediate state that production only logs in aggregate. Nothing
# is monkey-patched and no production code is modified — the stages
# are simply called in the same order with the same settings.
#
# Pipeline being replayed (RetrievalService.retrieve ->
# _run_retrieval_pass):
#   1. _extract_search_topic      strip instruction verbs
#   2. generate_bilingual_queries ES + EN variants
#   3. embed each variant
#   4. per variant: vector search, candidate_k rows per scope
#   5. merge into one pool, keyed by chunk_id, keeping max similarity
#   6. rerank_chunks               fused score + reference penalty
#   7. apply_per_document_cap      diversity
#   8. similarity threshold
#   9. slice to top_k
#
# HOW TO RUN (from project root):
#   PYTHONPATH=. .venv/bin/python scripts/diagnose_retrieval_owasp.py
# ============================================================

from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.db.supabase import get_supabase_client
from app.domain.documents.embeddings import embed_single
from app.domain.retrieval.query_expansion import generate_bilingual_queries
from app.domain.retrieval.ranking import apply_per_document_cap, rerank_chunks
from app.domain.retrieval.service import RetrievalService, _extract_search_topic

SEP = "=" * 78

QUERY = (
    "¿Qué referencias bibliográficas cita la Guía de Soluciones de "
    "Seguridad GenAI de OWASP?"
)
TARGET_FILENAME = "OWASP-GenAI-Security-Project-Solutions-Reference-Guide-Q2_Q325.pdf"

AGENT_TAGS = ["seguridad"]
COUNTRY = "costa-rica"
INDUSTRY = "cooperative"


def _print(title: str) -> None:
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)


def main() -> None:
    db = get_supabase_client()
    service = RetrievalService(db)

    docs = {d["id"]: d for d in db.table("documents").select(
        "id, filename, framework").execute().data}
    target_id = next(
        (i for i, d in docs.items() if d["filename"] == TARGET_FILENAME), None
    )
    if target_id is None:
        print(f"Target document not found: {TARGET_FILENAME}")
        return

    def label(chunk) -> str:
        d = docs.get(chunk.document_id, {})
        return f"{d.get('filename', '?')[:38]:38} #{chunk.chunk_index:<4}"

    def framework_of(chunk) -> str:
        return docs.get(chunk.document_id, {}).get("framework") or "?"

    top_k = settings.retrieval_top_k
    candidate_k = max(settings.retrieval_candidate_k, top_k)

    _print("0. Pipeline configuration")
    print(f"  retrieval_top_k              : {top_k}")
    print(f"  retrieval_candidate_k        : {settings.retrieval_candidate_k}"
          f"  -> candidate_k = {candidate_k} per scope, per variant")
    print(f"  agent_max_chunks_per_document: {settings.agent_max_chunks_per_document}"
          f"  (agent-scoped regime)")
    print(f"  similarity_threshold         : {settings.similarity_threshold}")
    print(f"  retrieval_reference_penalty  : {settings.retrieval_reference_penalty}")
    print(f"  retrieval_multilingual       : {settings.retrieval_multilingual}")
    print(f"  retrieval_use_hybrid         : {settings.retrieval_use_hybrid}")

    # ---------- Stages 1-3 ----------
    _print("1. Query preparation")
    topic = _extract_search_topic(QUERY)
    print(f"  original : {QUERY}")
    print(f"  topic    : {topic}")
    variants = generate_bilingual_queries(topic) or [topic]
    for i, v in enumerate(variants, 1):
        print(f"  variant {i}: {v}")
    embedded = [(v, embed_single(v)) for v in variants]

    # ---------- Stage 4-5: build the candidate pool ----------
    _print("2. Candidate pool (BEFORE rerank, cap and threshold)")
    pool: dict[str, object] = {}
    for variant, vector in embedded:
        chunks = service._search_global_chunks(
            query_embedding=vector,
            top_k=candidate_k,
            country=COUNTRY,
            framework_filter=None,
            agent_tags=AGENT_TAGS,
            industry_codes=[INDUSTRY],
        )
        print(f"  variant '{variant[:46]}...' -> {len(chunks)} rows")
        for c in chunks:
            existing = pool.get(c.chunk_id)
            if existing is None or c.similarity > existing.similarity:
                pool[c.chunk_id] = c

    candidates = list(pool.values())
    candidates.sort(key=lambda c: -c.similarity)
    print(f"\n  unique candidates in pool: {len(candidates)}")

    by_framework = Counter(framework_of(c) for c in candidates)
    by_doc = Counter(docs.get(c.document_id, {}).get("filename", "?")
                     for c in candidates)
    print("\n  distribution by framework:")
    for fw, n in by_framework.most_common():
        print(f"    {fw:20} {n:>3}  ({n / len(candidates) * 100:.0f}%)")
    print("\n  distribution by document:")
    for fn, n in by_doc.most_common():
        print(f"    {fn[:52]:52} {n:>3}")

    # ---------- The decisive question ----------
    _print("3. Is the target document in the pool?")
    in_pool = [
        (i, c) for i, c in enumerate(candidates, 1) if c.document_id == target_id
    ]
    if in_pool:
        print(f"  YES — {len(in_pool)} chunk(s) of {TARGET_FILENAME[:40]}...")
        for rank, c in in_pool:
            print(f"    pool rank {rank:>3}/{len(candidates)}  "
                  f"sim={c.similarity:.4f}  #{c.chunk_index}  "
                  f"is_reference_list={c.is_reference_list}")
    else:
        print(f"  NO — zero chunks of {TARGET_FILENAME[:40]}... reached the pool.")

    # For comparison: the PCI chunks that DID make the final cut.
    pci = [(i, c) for i, c in enumerate(candidates, 1)
           if "PCI" in docs.get(c.document_id, {}).get("filename", "")]
    if pci:
        print(f"\n  PCI-DSS chunks in pool: {len(pci)}; best few:")
        for rank, c in pci[:5]:
            print(f"    pool rank {rank:>3}  sim={c.similarity:.4f}  {label(c)}")

    # ---------- Stages 6-9: replay the cut ----------
    _print("4. What the pipeline does to the pool")
    ranked = rerank_chunks(candidates, QUERY)
    max_per_doc = settings.agent_max_chunks_per_document
    capped = apply_per_document_cap(ranked, max_per_doc)
    above = [c for c in capped if c.similarity >= settings.similarity_threshold]
    final = above[:top_k]

    def position(stage: list, doc_id: str) -> str:
        hits = [i for i, c in enumerate(stage, 1) if c.document_id == doc_id]
        return f"positions {hits[:5]}" if hits else "ABSENT"

    print(f"  after rerank      ({len(ranked):>3} chunks): target {position(ranked, target_id)}")
    print(f"  after cap({max_per_doc})      ({len(capped):>3} chunks): target {position(capped, target_id)}")
    print(f"  after threshold   ({len(above):>3} chunks): target {position(above, target_id)}")
    print(f"  final top-{top_k}        ({len(final):>3} chunks): target {position(final, target_id)}")

    print(f"\n  final top-{top_k}:")
    for i, c in enumerate(final, 1):
        print(f"    {i}. sim={c.similarity:.4f} [{framework_of(c):16}] {label(c)}"
              f" ref={c.is_reference_list}")

    # ---------- Direct similarity, bypassing the pipeline ----------
    _print("5. Direct similarity: query vs EVERY chunk of the target document")
    print("  Bypasses retrieval entirely — this is what the vector index sees.")
    import math

    def cos(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        na = math.sqrt(sum(x * x for x in a))
        nb = math.sqrt(sum(y * y for y in b))
        return dot / (na * nb) if na and nb else 0.0

    rows = (
        db.table("document_chunks")
        .select("chunk_index, content, embedding, is_reference_list")
        .eq("document_id", target_id)
        .order("chunk_index")
        .execute()
        .data
        or []
    )
    scored = []
    for variant, vector in embedded:
        for r in rows:
            emb = r.get("embedding")
            if isinstance(emb, str):
                import json
                emb = json.loads(emb)
            if not emb:
                continue
            scored.append((cos(vector, emb), r["chunk_index"],
                           bool(r.get("is_reference_list")), variant))
    best: dict[int, tuple] = {}
    for sim, idx, is_ref, variant in scored:
        if idx not in best or sim > best[idx][0]:
            best[idx] = (sim, idx, is_ref, variant)
    ranked_target = sorted(best.values(), key=lambda t: -t[0])

    print(f"\n  {len(rows)} chunks in the document. Top 8 by similarity:")
    for sim, idx, is_ref, variant in ranked_target[:8]:
        print(f"    sim={sim:.4f}  #{idx:<4} is_reference_list={str(is_ref):5} "
              f"(best variant: '{variant[:34]}...')")

    if candidates:
        cutoff = min(c.similarity for c in candidates)
        print(f"\n  Weakest similarity that DID enter the pool: {cutoff:.4f}")
        best_target = ranked_target[0][0] if ranked_target else 0.0
        print(f"  Best similarity of the target document      : {best_target:.4f}")
        verdict = ("ABOVE the pool cutoff — it should have entered"
                   if best_target > cutoff else
                   "BELOW the pool cutoff — it never had a chance")
        print(f"  -> {verdict}")

    # The flagged chunk specifically, since Q5 is a bibliography query.
    flagged = [t for t in ranked_target if t[2]]
    if flagged:
        sim, idx, _r, _v = flagged[0]
        rank = [i for i, t in enumerate(ranked_target, 1) if t[1] == idx][0]
        print(f"\n  The one chunk flagged is_reference_list (#{idx}): "
              f"sim={sim:.4f}, rank {rank}/{len(ranked_target)} within its own document.")


if __name__ == "__main__":
    main()
