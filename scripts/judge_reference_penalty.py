#!/usr/bin/env python3
# ============================================================
# scripts/judge_reference_penalty.py
#
# Does demoting bibliography chunks actually produce a BETTER
# ANSWER, or does it only move numbers around in the ranking?
#
# Answers that by holding everything constant except the retrieved
# context:
#
#   BEFORE — top-k reranked with the reference penalty disabled
#            (bibliography present, current production behaviour)
#   AFTER  — top-k reranked with the penalty active
#
# Both contexts go through the same production system prompt
# ("Chatbot Documental v5") and the same model, then both answers
# go to the real LLM judge (app/domain/evals/llm_judge.py), which
# reports how many of the answer's claims the context actually
# supports.
#
# Run with:
#   PYTHONPATH=. .venv/bin/python scripts/judge_reference_penalty.py
# ============================================================

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.db.supabase import get_supabase_client
from app.domain.evals.llm_judge import judge_response
from app.domain.llm.openai_client import OpenAIClient
from app.domain.retrieval.ranking import rerank_chunks

from scripts.validate_reference_penalty import (
    CASE_A_QUERIES,
    load_flags,
    retrieve_candidates,
)

PROMPT_NAME = "Chatbot Documental v5 (semantic grounding + cost optimized)"
SEP = "=" * 78


def load_system_prompt(db) -> str:
    rows = (
        db.table("prompts").select("name, content").eq("name", PROMPT_NAME)
        .limit(1).execute().data
        or []
    )
    if not rows:
        raise SystemExit(f"prompt {PROMPT_NAME!r} not found")
    return rows[0]["content"]


def build_context(chunks) -> str:
    """The production context format (see AIService._build_prompt)."""
    body = "\n\n".join(
        f"[Chunk {i+1} | Fuente: ISO42001_Replica.pdf"
        f"{f' | Section: {c.section_ref}' if c.section_ref else ''}"
        f" | Framework: iso_42001 | similarity: {c.similarity:.3f}]\n{c.content}"
        for i, c in enumerate(chunks)
    )
    return "\n\nDocumentos recuperados:\n" + "=" * 50 + f"\n{body}\n" + "=" * 50


def run_variant(client, system_prompt, query, chunks) -> tuple[str, dict]:
    prompt = f"{system_prompt.strip()}{build_context(chunks)}\n\nPregunta del usuario:\n{query}"
    answer, _latency, _pt, _ct = client.generate(prompt, temperature=0.2)
    verdict = judge_response(
        query=query,
        response=answer,
        chunks=[c.content for c in chunks],
        instructions=system_prompt,
    )
    return answer, verdict


def main() -> None:
    db = get_supabase_client()
    client = OpenAIClient()
    system_prompt = load_system_prompt(db)
    flags = load_flags(db)
    top_k = settings.retrieval_top_k

    totals = {"before": [0, 0], "after": [0, 0]}  # [supported, total]

    for query in CASE_A_QUERIES:
        candidates = retrieve_candidates(db, query, flags)
        before = rerank_chunks(query, list(candidates), reference_penalty=1.0)[:top_k]
        after = rerank_chunks(query, list(candidates))[:top_k]

        ref_before = [c.chunk_index for c in before if c.is_reference_list]
        if not ref_before:
            continue  # nothing changed for this query; no comparison to make

        print()
        print(SEP)
        print(f"  {query}")
        print(SEP)
        print(f"  BEFORE context chunks: {[c.chunk_index for c in before]} "
              f"(reference: {ref_before})")
        print(f"  AFTER  context chunks: {[c.chunk_index for c in after]}")

        for label, chunks in (("before", before), ("after", after)):
            answer, verdict = run_variant(client, system_prompt, query, chunks)
            supported = verdict["claims_supported"]
            total = verdict["claims_total"]
            totals[label][0] += supported
            totals[label][1] += total
            print(f"\n  --- {label.upper()} ---")
            print(f"    claims supported : {supported} / {total}")
            print(f"    hallucination    : {verdict['hallucination']}")
            print(f"    should say IDK   : {verdict['should_have_said_idk']}")
            print(f"    judge reasoning  : {verdict['reasoning'][:220]}")
            print(f"    answer           : {answer[:220].strip()}...")

    print()
    print(SEP)
    print("  TOTAL across compared queries")
    print(SEP)
    for label in ("before", "after"):
        sup, tot = totals[label]
        pct = (sup / tot * 100) if tot else 0.0
        print(f"  {label.upper():6}: {sup} / {tot} claims supported ({pct:.0f}%)")


if __name__ == "__main__":
    main()
