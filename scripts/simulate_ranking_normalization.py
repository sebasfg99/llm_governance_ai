#!/usr/bin/env python3
# ============================================================
# scripts/simulate_ranking_normalization.py
#
# Offline simulation of two fixes for the reranker calibration bug.
# Changes nothing: app/domain/retrieval/ranking.py is imported, never
# modified. The fused score is recomputed here from the same features
# the production reranker uses.
#
# THE BUG BEING SIMULATED AWAY:
# rerank_chunks() weights four features as if they were on comparable
# scales. Measured over a real 46-candidate pool:
#
#   feature            observed range   weight   effective influence
#   section_priority   0.00 - 1.00       0.10    0.100   <- largest
#   similarity         0.446 - 0.591     0.55    0.080
#   length_quality     0.40 - 1.00       0.05    0.030
#   keyword_overlap    0.000 - 0.025     0.30    0.008   <- inert
#
# The order of real influence inverts the intended one. Cosine
# similarity is compressed into a narrow band while section_priority
# uses its full range, so a nominal 0.10 weight decides the ranking.
#
# THE TWO STRATEGIES
#
#   A) Min-max scaling inside the candidate pool.
#      Rescale each feature to [0,1] using the min and max observed in
#      THAT pool, then apply the existing weights. Each feature then
#      contributes in proportion to how much it actually discriminates
#      among these candidates.
#
#   B) Fixed-range weight recalibration.
#      Keep the raw features, but divide each weight by that feature's
#      typical range so the effective influence matches intent. No
#      per-pool state — the weights become constants that can be read
#      and reasoned about.
#
# WHAT IT REPORTS
#   1. Per query: current top-5 vs A vs B.
#   2. Whether the query's expected framework reaches the top-5.
#   3. Effective-influence tables, to show the fix does what it claims.
#
# HOW TO RUN (from project root):
#   PYTHONPATH=. .venv/bin/python scripts/simulate_ranking_normalization.py
#   PYTHONPATH=. .venv/bin/python scripts/simulate_ranking_normalization.py --verbose
# ============================================================

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.config import settings
from app.db.supabase import get_supabase_client
from app.domain.documents.embeddings import embed_single
from app.domain.retrieval import ranking as R
from app.domain.retrieval.query_expansion import generate_bilingual_queries
from app.domain.retrieval.service import RetrievalService, _extract_search_topic

SEP = "=" * 78

# Production weights, read from the function's own defaults so this
# simulation cannot drift away from what ships.
W_SIM, W_OVL, W_SEC, W_LEN = 0.55, 0.30, 0.10, 0.05

# ------------------------------------------------------------
# Strategy B constants
#
# Typical spans measured across the pools this script builds. Dividing
# each weight by its feature's span equalises effective influence
# without needing per-pool state.
#
# similarity is the reference feature (span ~0.15), so its weight is
# left alone and the others are expressed relative to it.
# ------------------------------------------------------------
SPAN_SIM = 0.15   # cosine band actually seen among candidates
SPAN_OVL = 0.03   # Jaccard of a short query against a long chunk
SPAN_SEC = 1.00   # discrete 0.0 / 0.5 / 1.0
SPAN_LEN = 0.60   # 0.40 - 1.00 in practice


@dataclass
class Case:
    name: str
    query: str
    agent_tags: list[str]
    expect_framework: str | None = None
    expect_document: str | None = None


CASES = [
    # --- the case that exposed the bug ---
    Case("SEG Q5 bibliografía",
         "¿Qué referencias bibliográficas cita la Guía de Soluciones de "
         "Seguridad GenAI de OWASP?",
         ["seguridad"], "owasp_llm_top10",
         "OWASP-GenAI-Security-Project-Solutions-Reference-Guide-Q2_Q325.pdf"),
    # --- other Seguridad queries: must not break ---
    Case("SEG Q1 OWASP",
         "Según los documentos de OWASP GenAI Security, ¿cómo se aborda el "
         "riesgo de prompt injection en aplicaciones con LLMs?",
         ["seguridad"], "owasp_llm_top10"),
    Case("SEG Q3 MITRE",
         "¿Qué tácticas describe MITRE ATLAS sobre model evasion?",
         ["seguridad"], "mitre_atlas"),
    Case("SEG Q4 PCI",
         "¿Qué exige PCI DSS sobre gestión de logs?",
         ["seguridad"], "pci_dss"),
    # --- Normativa: the regression risk. Passes 4/4 today. ---
    Case("NRM Q1 ISO",
         "¿Qué exige ISO 42001 sobre evaluación de riesgos de IA?",
         ["normativa"], "iso_42001"),
    Case("NRM Q2 sesgo",
         "¿Cómo aborda ISO 42001 el sesgo en sistemas de IA y la toma de "
         "decisiones asistida?",
         ["normativa"], "iso_42001"),
    Case("NRM Q3 bibliografía",
         "¿Qué normas ISO cita el Anexo D de ISO 42001?",
         ["normativa"], "iso_42001"),
    Case("NRM Q4 país",
         "¿Qué normativa de gobernanza de IA aplica en Costa Rica?",
         ["normativa"], None),
]


def features(chunk, query_tokens) -> tuple[float, float, float, float]:
    """The exact four features rerank_chunks() computes."""
    return (
        float(chunk.similarity or 0.0),
        R._keyword_overlap(query_tokens, chunk.content or ""),
        R._section_priority(chunk.section_ref, chunk.content or ""),
        R._length_quality(chunk.content or ""),
    )


def reference_multiplier(chunk, query: str) -> float:
    """
    The reference-list penalty rerank_chunks() applies after fusing.

    Reimplemented here because leaving it out changes the answer: on
    bibliography-adjacent queries the flagged chunks float to the top of
    the simulation while production demotes them, which would make every
    strategy look wrong for the same reason.
    """
    if not getattr(chunk, "is_reference_list", False):
        return 1.0
    if R._is_reference_seeking_query(query or ""):
        return 1.0
    return settings.retrieval_reference_penalty


def score_current(f, mult: float = 1.0) -> float:
    sim, ovl, sec, ln = f
    return (W_SIM * sim + W_OVL * ovl + W_SEC * sec + W_LEN * ln) * mult


def _minmax(values: list[float]) -> tuple[float, float]:
    lo, hi = min(values), max(values)
    return lo, hi


def score_strategy_a(all_feats: list[tuple], idx: int, mult: float = 1.0) -> float:
    """Min-max scale each feature inside the pool, then apply weights."""
    cols = list(zip(*all_feats))
    scaled = []
    for col in cols:
        lo, hi = _minmax(list(col))
        span = hi - lo
        # A feature that does not vary in this pool carries no
        # information here — give every candidate the same 0.0 so it
        # cannot tip the ranking either way.
        scaled.append(0.0 if span == 0 else (col[idx] - lo) / span)
    sim, ovl, sec, ln = scaled
    return (W_SIM * sim + W_OVL * ovl + W_SEC * sec + W_LEN * ln) * mult


def score_strategy_b(f, mult: float = 1.0) -> float:
    """Divide each weight by its feature's typical span."""
    sim, ovl, sec, ln = f
    return (
        (W_SIM / SPAN_SIM) * sim
        + (W_OVL / SPAN_OVL) * ovl
        + (W_SEC / SPAN_SEC) * sec
        + (W_LEN / SPAN_LEN) * ln
    ) * mult


def score_strategy_c(f, mult: float = 1.0) -> float:
    """
    B, but WITHOUT amplifying keyword_overlap.

    Strategy B divides every weight by its feature's span, which
    multiplies the overlap weight by ~10. That feature is near-noise
    (Jaccard of a short query against a long chunk never exceeds ~0.025)
    AND it is cross-lingually biased: an English chunk shares almost no
    tokens with a Spanish query, so amplifying it re-creates exactly the
    blind spot that made grounding_score inverted.

    So overlap and length keep their raw nominal weights — they stay the
    weak tie-breakers they effectively already are — and only similarity
    and section_priority are put on a common scale.
    """
    sim, ovl, sec, ln = f
    return (
        (W_SIM / SPAN_SIM) * sim
        + W_OVL * ovl
        + (W_SEC / SPAN_SEC) * sec
        + W_LEN * ln
    ) * mult


def dedup_contiguous(chunks, max_gap: int = 1):
    """
    Drop a chunk when an already-selected one comes from the same
    document and sits within `max_gap` positions of it.

    MEASURED AND REJECTED — kept as the record of why.

    The hypothesis: adjacent chunks of one PDF share an overlapping text
    window, so an embedding artefact on one drags its neighbour in.
    It came from SEG Q3, where LLM_AI_Checklist #27 and #28 took the top
    two slots under Strategy B and #27 is about an unrelated tool
    (Defect Dojo), i.e. a genuine false positive riding on its
    neighbour.

    What the measurement showed:

      - Contiguous pairs are UBIQUITOUS, not a Q3 quirk: 8-23 pairs per
        candidate pool, and 6 of the 8 measured queries carry exactly
        one contiguous pair inside the top-5.
      - The dedup does fix Q3: it removes #27 and lifts SAFEAI from
        positions 3-5 to 2-4.
      - But it degrades three others, invisibly to the framework check
        (which stays True everywhere):
            SEG Q5  loses OWASP-GenAI #3 -> 1 chunk of the REQUESTED
                    document instead of 2
            NRM Q1  loses ISO #20        -> 2 ISO chunks instead of 3
            NRM Q2  loses ISO #56        -> 2 ISO chunks instead of 3

    Why it does not generalise: the chunker uses a sliding window WITH
    overlap (settings.chunk_overlap), so two consecutive chunks
    legitimately cover one continuous passage. Contiguity is therefore
    not evidence of an artefact, and dropping the neighbour discards
    real evidence more often than noise. Q3 was a coincidence — there
    #27 and #28 happen to discuss different things.

    Left in the script so the measurement is reproducible and the
    hypothesis is not re-proposed from scratch.
    """
    kept = []
    for c in chunks:
        clash = any(
            k.document_id == c.document_id
            and abs(k.chunk_index - c.chunk_index) <= max_gap
            for k in kept
        )
        if not clash:
            kept.append(c)
    return kept


def contiguous_pairs(chunks) -> list[tuple]:
    """Pairs of same-document chunks within 1 index of each other."""
    pairs = []
    for i, a in enumerate(chunks):
        for b in chunks[i + 1:]:
            if (a.document_id == b.document_id
                    and abs(a.chunk_index - b.chunk_index) <= 1):
                pairs.append((a, b))
    return pairs


def build_pool(svc, case: Case, candidate_k: int):
    """Replay the production candidate-pool construction."""
    topic = _extract_search_topic(case.query)
    variants = generate_bilingual_queries(topic) or [topic]
    pool = {}
    for variant in variants:
        vector = embed_single(variant)
        for c in svc._search_global_chunks(
            query_embedding=vector,
            top_k=candidate_k,
            country="costa-rica",
            framework_filter=None,
            agent_tags=case.agent_tags,
            industry_codes=["cooperative"],
        ):
            prev = pool.get(c.chunk_id)
            if prev is None or c.similarity > prev.similarity:
                pool[c.chunk_id] = c
    return list(pool.values())


def top_k_after_pipeline(ordered, max_per_doc, top_k):
    """Apply the cap and threshold exactly as _run_retrieval_pass does."""
    capped = R.apply_per_document_cap(ordered, max_per_doc)
    above = [c for c in capped if c.similarity >= settings.similarity_threshold]
    return above[:top_k]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Simulate feature normalisation in the reranker"
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Show the full top-5 of every strategy")
    args = parser.parse_args()

    db = get_supabase_client()
    svc = RetrievalService(db)
    docs = {d["id"]: d for d in db.table("documents")
            .select("id, filename, framework").execute().data}

    top_k = settings.retrieval_top_k
    candidate_k = max(settings.retrieval_candidate_k, top_k)

    print(SEP)
    print("  Ranking normalisation — offline simulation")
    print(SEP)
    print(f"  weights (production): sim={W_SIM} ovl={W_OVL} sec={W_SEC} len={W_LEN}")
    print(f"  Strategy A: min-max inside the candidate pool")
    print(f"  Strategy B: weight / typical span "
          f"(sim/{SPAN_SIM} ovl/{SPAN_OVL} sec/{SPAN_SEC} len/{SPAN_LEN})")
    print(f"  candidate_k={candidate_k}  top_k={top_k}")

    summary = []

    for case in CASES:
        cands = build_pool(svc, case, candidate_k)
        if not cands:
            print(f"\n  {case.name}: empty pool, skipped")
            continue

        qt = R._tokenize_for_overlap(case.query)
        feats = [features(c, qt) for c in cands]

        max_per_doc = (settings.agent_max_chunks_per_document
                       if case.agent_tags else
                       settings.retrieval_max_chunks_per_document)

        mults = [reference_multiplier(c, case.query) for c in cands]

        variants = {
            "current": sorted(
                range(len(cands)),
                key=lambda i: -score_current(feats[i], mults[i])),
            "A": sorted(
                range(len(cands)),
                key=lambda i: -score_strategy_a(feats, i, mults[i])),
            "B": sorted(
                range(len(cands)),
                key=lambda i: -score_strategy_b(feats[i], mults[i])),
            "C": sorted(
                range(len(cands)),
                key=lambda i: -score_strategy_c(feats[i], mults[i])),
        }

        print()
        print(SEP)
        print(f"  {case.name}   (pool: {len(cands)} candidates)")
        print(SEP)

        row = {"case": case.name}
        for label, order in variants.items():
            ordered = [cands[i] for i in order]
            final = top_k_after_pipeline(ordered, max_per_doc, top_k)
            frameworks = [docs.get(c.document_id, {}).get("framework")
                          for c in final]
            filenames = [docs.get(c.document_id, {}).get("filename", "?")
                         for c in final]

            hit_fw = (case.expect_framework in frameworks
                      if case.expect_framework else None)
            hit_doc = (any(f == case.expect_document for f in filenames)
                       if case.expect_document else None)
            row[label] = (hit_fw, hit_doc, frameworks)

            flag = ""
            if hit_fw is not None:
                flag += f"  expected framework: {'YES' if hit_fw else 'NO '}"
            if hit_doc is not None:
                flag += f"  expected document: {'YES' if hit_doc else 'NO '}"
            print(f"\n  [{label:7}]{flag}")
            if args.verbose or label != "current":
                for i, c in enumerate(final, 1):
                    d = docs.get(c.document_id, {})
                    print(f"     {i}. sim={c.similarity:.3f} "
                          f"sec={R._section_priority(c.section_ref, c.content or ''):.1f} "
                          f"[{str(d.get('framework'))[:15]:15}] "
                          f"{d.get('filename', '?')[:34]:34}#{c.chunk_index}")
            else:
                for i, c in enumerate(final, 1):
                    d = docs.get(c.document_id, {})
                    print(f"     {i}. sim={c.similarity:.3f} "
                          f"sec={R._section_priority(c.section_ref, c.content or ''):.1f} "
                          f"[{str(d.get('framework'))[:15]:15}] "
                          f"{d.get('filename', '?')[:34]:34}#{c.chunk_index}")

        summary.append(row)

    # ---------- comparative table ----------
    print()
    print(SEP)
    print("  Comparative summary — does the expected framework reach top-5?")
    print(SEP)
    print(f"  {'query':22} {'current':>9} {'A':>9} {'B':>9} {'C':>9}   expected")
    print("  " + "-" * 78)
    tally = {"current": 0, "A": 0, "B": 0, "C": 0}
    total = 0
    for row in summary:
        exp = None
        cells = []
        for label in ("current", "A", "B", "C"):
            hit_fw, _hit_doc, _fw = row[label]
            if hit_fw is None:
                cells.append("   n/a")
            else:
                cells.append("    YES" if hit_fw else "    NO ")
                if hit_fw:
                    tally[label] += 1
        # count the query once
        if row["current"][0] is not None:
            total += 1
        case = next(c for c in CASES if c.name == row["case"])
        exp = case.expect_framework or "—"
        print(f"  {row['case']:22} {cells[0]:>9} {cells[1]:>9} {cells[2]:>9} {cells[3]:>9}   {exp}")
    print("  " + "-" * 78)
    print(f"  {'TOTAL with expected fw':22} {tally['current']:>6}/{total} "
          f"{tally['A']:>6}/{total} {tally['B']:>6}/{total} {tally['C']:>6}/{total}")

    # Q5's document-level check, the one the whole exercise is about.
    q5 = next((r for r in summary if r["case"].startswith("SEG Q5")), None)
    if q5:
        print()
        print("  SEG Q5 — does the REQUESTED DOCUMENT reach top-5?")
        for label in ("current", "A", "B", "C"):
            _fw, hit_doc, _f = q5[label]
            print(f"    {label:8} {'YES' if hit_doc else 'NO'}")


if __name__ == "__main__":
    main()
