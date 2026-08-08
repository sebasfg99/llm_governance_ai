#!/usr/bin/env python3
# ============================================================
# scripts/diagnose_grounding_crosslingual.py
#
# Investigation: grounding_score() cannot see evidence that is not in
# the same language as the answer.
#
# WHAT PROMPTED IT:
# On the Seguridad validation set every query whose retrieved context
# was mostly English landed under the 0.40 gate (0.299–0.397) while
# every all-Spanish one cleared it comfortably (0.467–0.785). The
# values barely moved between runs, so this was not judge variance.
#
# WHAT THIS SCRIPT ESTABLISHES
#
#   1. HARNESS FIDELITY — recomputing with the real grounding_score()
#      reproduces the stored value, so the rest of the numbers are
#      about the metric and not about a replay artefact.
#
#   2. THE STOP-WORD HYPOTHESIS IS WRONG. graders.py strips English
#      stop words only, and every answer is Spanish by prompt, so the
#      obvious guess is that Spanish function words inflate same-
#      language scores and deflate cross-language ones. Adding a
#      Spanish stop list moves BOTH groups down by about the same
#      amount (~-0.13), so it explains nothing. Kept in the output as
#      V1 because ruling it out is the point.
#
#   3. THE ACTUAL MECHANISM — containment computed against only the
#      Spanish chunks equals the full score, while containment against
#      only the English chunks is ~0. English chunks occupy top-k
#      slots and contribute nothing. The metric silently shrinks the
#      effective context to the same-language subset.
#
#   4. THE CONSEQUENCE THAT MATTERS — on English-sourced answers the
#      metric stops discriminating. A deliberately invented answer
#      scores about the same as the real grounded one (0.267 vs 0.299
#      on SEG Q1), so the gate is not measuring grounding there at all.
#      A semantic (embedding cosine) variant keeps that separation.
#
# It changes nothing: graders.py is untouched. This is measurement.
#
# HOW TO RUN (from project root):
#   PYTHONPATH=. .venv/bin/python scripts/diagnose_grounding_crosslingual.py
#   PYTHONPATH=. .venv/bin/python scripts/diagnose_grounding_crosslingual.py --semantic
#
# --semantic adds the embedding comparison, which costs one embedding
# call per response plus one per chunk.
# ============================================================

from __future__ import annotations

import argparse
import math
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.supabase import get_supabase_client
from app.domain.evals.graders import grounding_score

SEP = "=" * 78

# The exact list graders.py uses today.
EN_STOP = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall",
    "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "and", "or", "but", "not", "it", "this", "that", "which",
}

# The Spanish set graders.py does NOT have.
ES_STOP = {
    "el", "la", "los", "las", "un", "una", "unos", "unas", "lo",
    "de", "del", "al", "en", "por", "para", "con", "sin", "sobre",
    "y", "e", "o", "u", "que", "se", "su", "sus", "es", "son",
    "ser", "está", "estan", "están", "fue", "fueron", "ha", "han",
    "como", "más", "mas", "pero", "si", "no", "ya", "cuando",
    "donde", "cual", "cuales", "este", "esta", "estos", "estas",
    "ese", "esa", "esos", "esas", "debe", "deben", "puede", "pueden",
    "cada", "todo", "toda", "todos", "todas", "otro", "otra",
    "entre", "desde", "hasta", "también", "tambien", "ante", "tras",
}

_PUNCT_RE = re.compile(r"[^\w\s]", re.UNICODE)

# Answers invented for the discrimination test. Neither is supported by
# any document in the corpus; the first is deliberately plausible
# governance prose, the second is off-topic.
FABRICATED = (
    "La norma establece que toda organización debe designar un comité de ética "
    "algorítmica con al menos siete miembros independientes, renovables cada 18 "
    "meses, y presentar un informe trimestral de sesgo ante el regulador. El "
    "incumplimiento acarrea multas de hasta el 4% de la facturación anual y la "
    "suspensión del sistema de IA por 90 días."
)
OFF_TOPIC = (
    "La receta tradicional requiere harina, huevos y mantequilla. Se mezcla todo "
    "hasta obtener una masa homogénea y se hornea a 180 grados durante 25 minutos."
)

QUERIES = [
    ("SEG Q1 OWASP", "Según los documentos de OWASP GenAI Security, ¿cómo se aborda el riesgo de prompt injection en aplicaciones con LLMs?"),
    ("SEG Q2 Encript", "¿Qué controles de encriptación aplican para datos usados en entrenamiento de IA?"),
    ("SEG Q3 MITRE", "¿Qué tácticas describe MITRE ATLAS sobre model evasion?"),
    ("SEG Q4 PCI", "¿Qué exige PCI DSS sobre gestión de logs?"),
    ("SEG Q5 Biblio", "¿Qué referencias bibliográficas cita la Guía de Soluciones de Seguridad GenAI de OWASP?"),
    ("SEG Q6 País", "¿Qué normativa aplica en Costa Rica sobre ciberseguridad en entidades financieras?"),
    ("NRM Q1 ISO", "¿Qué exige ISO 42001 sobre evaluación de riesgos de IA?"),
    ("NRM Q2 Sesgo", "¿Cómo aborda ISO 42001 el sesgo en sistemas de IA y la toma de decisiones asistida?"),
    ("NRM Q3 Biblio", "¿Qué normas ISO cita el Anexo D de ISO 42001?"),
    ("NRM Q4 País", "¿Qué normativa de gobernanza de IA aplica en Costa Rica?"),
]


def _tokens(text: str, stop: set[str]) -> set[str]:
    return {w for w in _PUNCT_RE.sub(" ", text.lower()).split() if w} - stop


def containment(response: str, chunks: list[str], stop: set[str]) -> float:
    """grounding_score()'s formula with a swappable stop-word list."""
    resp = _tokens(response, stop)
    if not resp:
        return 0.0
    ctx: set[str] = set()
    for c in chunks:
        ctx |= _tokens(c, stop)
    if not ctx:
        return 0.0
    return round(len(resp & ctx) / len(resp), 4)


def is_english(text: str) -> bool:
    """Crude language probe: which function-word family dominates."""
    words = [w for w in _PUNCT_RE.sub(" ", text.lower()).split() if w]
    return sum(1 for w in words if w in EN_STOP) > sum(1 for w in words if w in ES_STOP)


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def load_trace(db, query: str) -> tuple[str, list[str], float | None] | None:
    """Most recent (response, chunk texts, stored grounding) for a query."""
    traces = (
        db.table("traces")
        .select("id, response_text, retrieved_chunk_ids")
        .eq("user_query", query)
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
    )
    if not traces:
        return None
    t = traces[0]
    ev = (
        db.table("evaluations")
        .select("grounding_score")
        .eq("trace_id", t["id"])
        .execute()
        .data
    )
    stored = ev[0]["grounding_score"] if ev else None
    ids = [
        x if isinstance(x, str) else x.get("chunk_id")
        for x in (t.get("retrieved_chunk_ids") or [])
    ]
    if not ids:
        return None
    chunks = [
        r["content"]
        for r in (
            db.table("document_chunks").select("content").in_("id", ids).execute().data
            or []
        )
    ]
    response = t.get("response_text") or ""
    if not chunks or not response:
        return None
    return response, chunks, stored


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose the cross-lingual behaviour of grounding_score()"
    )
    parser.add_argument(
        "--semantic",
        action="store_true",
        help="Also compute the embedding-based variant (costs embedding calls)",
    )
    args = parser.parse_args()

    db = get_supabase_client()
    loaded = []
    for label, query in QUERIES:
        row = load_trace(db, query)
        if row is None:
            print(f"  {label:16} (no usable trace)")
            continue
        response, chunks, stored = row
        loaded.append((label, response, chunks, stored))

    # ---------- 1 & 2: fidelity, and ruling out the stop-word theory ----------
    print(SEP)
    print("  1. Current metric, and the bilingual stop-word variant")
    print(SEP)
    print("  V0    = current implementation (must match the stored score)")
    print("  V1    = same formula, stop words for BOTH languages")
    print("  ES/EN = containment against only the Spanish / only the English chunks")
    print()
    print(f"  {'query':16} {'stored':>7} {'V0':>7} {'V1':>7} | {'ES':>6} {'EN':>6} | {'nEN/n':>7}")
    print("  " + "-" * 68)

    rows = []
    for label, response, chunks, stored in loaded:
        v0 = grounding_score(response, chunks)
        v1 = containment(response, chunks, EN_STOP | ES_STOP)
        es_chunks = [c for c in chunks if not is_english(c)]
        en_chunks = [c for c in chunks if is_english(c)]
        vs_es = grounding_score(response, es_chunks) if es_chunks else float("nan")
        vs_en = grounding_score(response, en_chunks) if en_chunks else float("nan")
        rows.append((label, v0, v1, vs_es, vs_en, len(en_chunks), len(chunks)))
        stored_s = f"{stored:.3f}" if stored is not None else "  n/a"
        print(
            f"  {label:16} {stored_s:>7} {v0:>7.3f} {v1:>7.3f} | "
            f"{vs_es:>6.3f} {vs_en:>6.3f} | {len(en_chunks):>3}/{len(chunks)}"
        )

    english_ctx = [r for r in rows if r[5] >= 3]
    spanish_ctx = [r for r in rows if r[5] == 0]

    print()
    print("  Adding Spanish stop words (V1 - V0), by context language:")
    for name, group in (("English-dominant", english_ctx), ("all-Spanish", spanish_ctx)):
        if group:
            delta = statistics.mean(r[2] - r[1] for r in group)
            print(f"    {name:18} n={len(group)}  Δ={delta:+.3f}")
    print("  Both groups move together => the stop-word list is NOT the cause.")

    # ---------- 3: the real mechanism ----------
    print()
    print(SEP)
    print("  2. Where the score actually comes from")
    print(SEP)
    en_only = [r for r in rows if not math.isnan(r[4])]
    if en_only:
        print(f"  Mean containment against English chunks : "
              f"{statistics.mean(r[4] for r in en_only):.3f}")
        print(f"  Mean containment against Spanish chunks : "
              f"{statistics.mean(r[3] for r in en_only if not math.isnan(r[3])):.3f}")
    print("  English chunks take top-k slots but add ~nothing to the score:")
    print("  the metric effectively evaluates only the same-language subset.")

    # ---------- 4: discrimination ----------
    if not args.semantic:
        print()
        print("  Re-run with --semantic for the discrimination test.")
        return

    from app.domain.documents.embeddings import embed_texts  # noqa: E402

    print()
    print(SEP)
    print("  3. Does the metric still tell grounded from invented?")
    print(SEP)
    print("  REAL = the answer the model actually produced")
    print("  FAKE = plausible governance prose supported by no document")
    print("  OFF  = off-topic text (a cooking recipe)")
    print()
    print(f"  {'query':16} | {'REAL lex':>8} {'FAKE lex':>8} {'gap':>6} | "
          f"{'REAL sem':>8} {'FAKE sem':>8} {'gap':>6} | {'OFF sem':>7}")
    print("  " + "-" * 82)

    lex_gaps, sem_gaps = [], []
    for label, response, chunks, _stored in loaded:
        embs = embed_texts([response, FABRICATED, OFF_TOPIC] + chunks)
        rv, fv, ov, cvs = embs[0], embs[1], embs[2], embs[3:]
        sem = lambda v: max(cosine(v, c) for c in cvs)  # noqa: E731
        real_lex = grounding_score(response, chunks)
        fake_lex = grounding_score(FABRICATED, chunks)
        real_sem, fake_sem, off_sem = sem(rv), sem(fv), sem(ov)
        lex_gaps.append(real_lex - fake_lex)
        sem_gaps.append(real_sem - fake_sem)
        print(
            f"  {label:16} | {real_lex:>8.3f} {fake_lex:>8.3f} "
            f"{real_lex - fake_lex:>6.3f} | {real_sem:>8.3f} {fake_sem:>8.3f} "
            f"{real_sem - fake_sem:>6.3f} | {off_sem:>7.3f}"
        )

    print()
    print(f"  Mean separation, lexical : {statistics.mean(lex_gaps):+.3f}")
    print(f"  Mean separation, semantic: {statistics.mean(sem_gaps):+.3f}")
    print()
    print("  A gap near zero means the metric cannot tell a grounded answer")
    print("  from an invented one for that query.")


if __name__ == "__main__":
    main()
