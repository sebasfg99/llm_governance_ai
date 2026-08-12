#!/usr/bin/env python3
# ============================================================
# scripts/validate_reference_penalty.py
#
# Validation for the is_reference_list fix.
#
# WHY THIS RUNS OFFLINE:
# Migration 011 has not been applied to Supabase yet (it needs the
# SQL editor), so the RPCs do not return is_reference_list. This
# script reproduces the production ranking faithfully anyway:
#
#   - candidates and their similarities come from the REAL RPC
#     against the REAL corpus, exactly as in production;
#   - the flag is recomputed locally by re-chunking the source PDF
#     and joined on chunk_index. That join is exact — the chunker
#     is deterministic, and this script asserts the re-chunked text
#     is byte-identical to what is stored;
#   - rerank_chunks() is the real production function.
#
# So the only thing simulated is where the flag comes from, which
# is precisely what the migration will change.
#
# Checks:
#   1. Classification — idx 70 flagged, 64/56/68 not.
#      (was 69/70 before the migration-012 reprocess re-cut
#      every boundary in ISO 42001)
#   2. Case A — does the substantive chunk displace the
#      bibliography in the top-k?
#   3. Control — does a question ABOUT the references still
#      retrieve them?
#
# Run with:
#   PYTHONPATH=. .venv/bin/python scripts/validate_reference_penalty.py
# ============================================================

from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.db.supabase import get_supabase_client
from app.domain.documents.chunking import chunk_text_with_sections
from app.domain.documents.embeddings import embed_single
from app.domain.documents.parser_pdf import parse_pdf
from app.domain.retrieval import ranking
from app.domain.retrieval.ranking import ChunkResult, rerank_chunks

ISO_DOC_ID = "a93bc396-6c66-4296-aa8a-0f178998cac0"
ISO_PATH = f"global/{ISO_DOC_ID}/ISO42001_Replica.pdf"

# Case A's exact wording was not recoverable — that diagnosis ran
# offline and left no trace row. These queries reproduce the SAME
# inversion against the same corpus: a question about content whose
# words happen to appear in the titles listed in the Anexo D
# bibliography, so the bibliography outranks the substantive text.
#
# The first is the sharpest reproduction: it matches the entry
# "ISO/IEC TR 24027:2021 — Bias in AI systems and AI aided decision
# making" almost verbatim, and the bibliography takes vector ranks
# 1 and 3 — worse than the 3rd/5th originally reported.
CASE_A_QUERIES = [
    "¿Cómo aborda ISO 42001 el sesgo en los sistemas de IA y la toma de "
    "decisiones asistida?",
    "¿Qué orientación da ISO 42001 sobre calidad de datos, privacidad y "
    "seguridad de la información?",
    "¿Qué marcos de gestión de riesgos y terminología de IA se relacionan "
    "con ISO 42001?",
    "¿Qué dice ISO 42001 sobre la gestión del riesgo y la gobernanza de TI "
    "en el uso de IA?",
]

CONTROL_QUERIES = [
    "¿Qué normas ISO cita el Anexo D de ISO 42001?",
    "Muéstrame la bibliografía de la norma ISO 42001",
    "¿Cuáles son las normas citadas en el documento?",
]

SEP = "=" * 78


def _print(title: str) -> None:
    print()
    print(SEP)
    print(f"  {title}")
    print(SEP)


def load_flags(db) -> dict[int, bool]:
    """Re-chunk the ISO 42001 PDF and return {chunk_index: is_reference_list}."""
    blob = db.storage.from_(settings.storage_bucket).download(ISO_PATH)
    fd, tmp = tempfile.mkstemp(suffix=".pdf")
    with os.fdopen(fd, "wb") as fh:
        fh.write(blob)
    try:
        text, _pages = parse_pdf(tmp)
    finally:
        os.unlink(tmp)

    chunks = chunk_text_with_sections(text)

    # The join below is only valid if re-chunking reproduces what is
    # stored. Assert it rather than assume it.
    stored = (
        db.table("document_chunks")
        .select("chunk_index, content")
        .eq("document_id", ISO_DOC_ID)
        .order("chunk_index")
        .execute()
        .data
        or []
    )
    assert len(stored) == len(chunks), (
        f"chunk count drift: stored={len(stored)} recomputed={len(chunks)}"
    )
    for row in stored:
        assert row["content"] == chunks[row["chunk_index"]].content, (
            f"chunk {row['chunk_index']} text drifted from what is stored"
        )
    print(f"  join verified: {len(chunks)} chunks reproduce stored text exactly")

    return {i: c.is_reference_list for i, c in enumerate(chunks)}


def retrieve_candidates(db, query: str, flags: dict[int, bool]) -> list[ChunkResult]:
    """Real RPC call against the real corpus, with flags joined in."""
    embedding = embed_single(query)
    rows = (
        db.rpc(
            "match_global_chunks",
            {
                "query_embedding": embedding,
                "match_count": settings.retrieval_candidate_k,
                "match_country": None,
                "match_frameworks": ["iso_42001"],
            },
        )
        .execute()
        .data
        or []
    )
    results = []
    for row in rows:
        is_iso = str(row["document_id"]) == ISO_DOC_ID
        results.append(
            ChunkResult(
                chunk_id=str(row["id"]),
                document_id=str(row["document_id"]),
                tenant_id=row["tenant_id"],
                content=row["content"],
                chunk_index=row["chunk_index"],
                similarity=float(row["similarity"]),
                section_ref=row.get("section_ref"),
                is_reference_list=flags.get(row["chunk_index"], False) if is_iso else False,
            )
        )
    return results


def show_ranking(label: str, chunks: list[ChunkResult], top: int = 8) -> None:
    print(f"\n  --- {label} ---")
    print(f"    {'#':>2}  {'idx':>4}  {'sim':>5}  {'ref?':>5}  {'section':10} preview")
    for rank, c in enumerate(chunks[:top], start=1):
        flag = "REF" if c.is_reference_list else "-"
        preview = (c.content or "")[:52].replace("\n", " ")
        print(
            f"    {rank:>2}  {c.chunk_index:>4}  {c.similarity:>5.3f}  {flag:>5}  "
            f"{str(c.section_ref)[:10]:10} {preview}"
        )


def main() -> None:
    db = get_supabase_client()

    # ---------- 1. Classification ----------
    _print("1. Classification of the labelled ISO 42001 chunks")
    flags = load_flags(db)
    # Labelled by reading the chunks, and RE-LABELLED after the
    # front-matter reprocess (migration 012) moved every boundary in
    # this document. The previous set was {69: True, 70: True, 64:
    # False, 56: False, 68: False}; #69 is now substantive prose and
    # only #70 carries the bibliography, which is a precision
    # improvement — the old #69 was a carry-forward false positive.
    #
    # If this assertion fails after a reprocess, re-derive the set
    # rather than editing it to match: the point is to catch the
    # classifier drifting, and a hand-tuned expectation cannot.
    expected = {70: True, 64: False, 56: False, 68: False}
    ok = True
    print(f"\n  {'idx':>4}  {'expected':>9}  {'actual':>7}  verdict")
    for idx, want in sorted(expected.items()):
        got = flags.get(idx, False)
        verdict = "PASS" if got == want else "FAIL"
        ok &= got == want
        print(f"  {idx:>4}  {str(want):>9}  {str(got):>7}  {verdict}")
    total = sum(1 for v in flags.values() if v)
    print(f"\n  {total} of {len(flags)} ISO 42001 chunks flagged: "
          f"{sorted(i for i, v in flags.items() if v)}")
    print(f"  => classification {'PASS' if ok else 'FAIL'}")

    top_k = settings.retrieval_top_k

    # ---------- 2. Case A ----------
    _print("2. Case A — questions about content, not about references")
    case_a_ok = True
    for query in CASE_A_QUERIES:
        print(f"\n  query: {query}")
        candidates = retrieve_candidates(db, query, flags)
        before = rerank_chunks(query, list(candidates), reference_penalty=1.0)
        after = rerank_chunks(query, list(candidates))

        ref_before = [c.chunk_index for c in before[:top_k] if c.is_reference_list]
        ref_after = [c.chunk_index for c in after[:top_k] if c.is_reference_list]
        # The slots the bibliography vacated must be filled by real content,
        # not left empty — check what moved up into the top-k.
        promoted = [
            c.chunk_index
            for c in after[:top_k]
            if c.chunk_index not in {b.chunk_index for b in before[:top_k]}
        ]
        if ref_before:
            verdict = "PASS" if not ref_after else "FAIL"
            case_a_ok &= not ref_after
        else:
            verdict = "n/a — bibliography never entered top-k here"
        print(f"    reference chunks in top-{top_k}: "
              f"before={ref_before or '[]'}  after={ref_after or '[]'}")
        print(f"    promoted into top-{top_k}: {promoted or '[]'}")
        print(f"    => {verdict}")

        if ref_before:
            show_ranking("BEFORE (penalty disabled)", before, top=6)
            show_ranking("AFTER  (penalty active)", after, top=6)

    print(f"\n  => Case A overall {'PASS' if case_a_ok else 'FAIL'}")

    # ---------- 3. Control ----------
    _print("3. Control — questions that ARE about the references")
    print("  The requirement is that the penalty lowers priority WITHOUT")
    print("  making the bibliography invisible.")
    print()
    print("  NOTE on how this is measured. Comparing rerank_chunks(...) against")
    print("  rerank_chunks(..., reference_penalty=1.0) would be TAUTOLOGICAL")
    print("  here: on a reference-seeking query the waiver already forces the")
    print("  multiplier to 1.0, so both sides compute the same thing and the")
    print("  ranks trivially match. To show the waiver is actually doing the")
    print("  work, the run below SUPPRESSES it and lets the 0.25 penalty land,")
    print("  which is what the bibliography's rank would be without the waiver.")
    control_ok = True
    for query in CONTROL_QUERIES:
        candidates = retrieve_candidates(db, query, flags)
        with_waiver = rerank_chunks(query, list(candidates))

        # Suppress the waiver so the penalty genuinely applies. Restored
        # immediately — this is a measurement, not a behaviour change.
        original = ranking._is_reference_seeking_query
        ranking._is_reference_seeking_query = lambda _q: False
        try:
            without_waiver = rerank_chunks(query, list(candidates))
        finally:
            ranking._is_reference_seeking_query = original

        pos_with = {c.chunk_index: i + 1 for i, c in enumerate(with_waiver)}
        pos_without = {c.chunk_index: i + 1 for i, c in enumerate(without_waiver)}
        available = sorted(c.chunk_index for c in candidates if c.is_reference_list)

        print(f"\n  query: {query}")
        if not available:
            print("    no reference chunks in the candidate pool — nothing to check")
            continue

        waiver_works = True
        for idx in available:
            rank_w, rank_wo = pos_with[idx], pos_without[idx]
            # The waiver must strictly improve the rank, otherwise it is
            # not actually protecting the bibliography.
            waiver_works &= rank_w < rank_wo
            marker = f"  <== in top-{top_k}" if rank_w <= top_k else ""
            print(f"    idx={idx}: rank {rank_w:>2} with waiver  vs  "
                  f"{rank_wo:>2} if the penalty applied{marker}")
        control_ok &= waiver_works
        print(f"    => {'PASS — waiver keeps the bibliography reachable' if waiver_works else 'FAIL — waiver had no effect'}")

    print(f"\n  => Control overall "
          f"{'PASS' if control_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
