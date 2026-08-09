#!/usr/bin/env python3
# ============================================================
# scripts/diagnose_front_matter_llm.py
#
# Does an LLM separate front matter from body text where cheap
# heuristics could not?
#
# WHY THIS EXISTS:
# The reranker's section_priority feature carries a structural bias
# (documents without legal section markers are demoted regardless of
# relevance). Removing it — Strategy B, see
# scripts/simulate_ranking_normalization.py — fixes that bias but
# regresses Normativa from 4/4 to 2/4, because section_priority was
# ALSO acting as an accidental front-matter filter: covers, tables of
# contents, copyright notices and committee credit lists have no
# section markers either, and once the bias is gone they surface into
# the answer context.
#
# Two cheaper approaches were measured and rejected first:
#
#   1. A single text feature (relative position, regulatory-verb
#      density, dot-leaders, digit density). All four ranges OVERLAP
#      between front matter and legitimate low-index body text — see
#      the measurement in the task log. SAFEAI #1 is real content at
#      relative position 0.009; ISO42001 #4 is a credits list at
#      0.056.
#
#   2. Per-subtype rules (cover / TOC / legal boilerplate). 11 of 17
#      caught, but with 4 false positives on real content (PCI-DSS
#      requirements, SAFEAI's executive summary) AND missing exactly
#      the two chunks that caused the regression (ISO42001 #3 and #4).
#
# The remaining hypothesis: telling "list of committee members" from
# "normative requirement" is a judgement about the ROLE of the text,
# not a pattern. An LLM should do it trivially. This script tests that
# on the labelled set before any pipeline is built around it.
#
# GROUND TRUTH
# 35 chunks drawn from the 8 measured query pools and labelled by
# reading them: 17 front matter, 18 body text that sits at a low
# chunk_index (the population any position rule would wrongly catch).
#
# HOW TO RUN (from project root):
#   PYTHONPATH=. .venv/bin/python scripts/diagnose_front_matter_llm.py
#   PYTHONPATH=. .venv/bin/python scripts/diagnose_front_matter_llm.py --verbose
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import openai as openai_sdk

from app.core.config import settings
from app.db.supabase import get_supabase_client

SEP = "=" * 78
MODEL = "gpt-4o-mini"
BATCH = 6

# (filename, chunk_index) -> known label
FRONT_MATTER = {
    ("Agentic-AI-Threats-and-Mitigations-1.1.pdf", 0),
    ("Agentic-AI-Threats-and-Mitigations-1.1.pdf", 1),
    ("CONASSIF 4-16 (v07 2 de abril de 2024).pdf", 3),
    ("ISO42001_Replica.pdf", 0),
    ("ISO42001_Replica.pdf", 1),
    ("ISO42001_Replica.pdf", 2),
    ("ISO42001_Replica.pdf", 3),
    ("ISO42001_Replica.pdf", 4),
    ("LLM_AI_Security_and_Governance_Checklist-v1.1.pdf", 0),
    ("NIST.AI.100-1.pdf", 0),
    ("OWASP-GenAI-Security-Project-Solutions-Reference-Guide-Q2_Q325.pdf", 0),
    ("OWASP-GenAI-Security-Project-Solutions-Reference-Guide-Q2_Q325.pdf", 1),
    ("OWASP-GenAI-Security-Project-Solutions-Reference-Guide-Q2_Q325.pdf", 3),
    ("PCI-DSS-v4_0_1-LA.pdf", 2),
    ("PCI-DSS-v4_0_1-LA.pdf", 3),
    ("SAFEAI_Full_Report.pdf", 0),
    ("SAFEAI_Full_Report.pdf", 2),
}

BODY_LOW_INDEX = {
    ("AI_RMF_Playbook.pdf", 2),
    ("AI_RMF_Playbook.pdf", 4),
    ("AI_RMF_Playbook.pdf", 6),
    ("CONASSIF 4-16 (v07 2 de abril de 2024).pdf", 5),
    ("CONASSIF 4-16 (v07 2 de abril de 2024).pdf", 6),
    ("ISO42001_Replica.pdf", 5),
    ("ISO42001_Replica.pdf", 6),
    ("NIST.AI.100-1.pdf", 3),
    ("OWASP-GenAI-Security-Project-Solutions-Reference-Guide-Q2_Q325.pdf", 2),
    ("OWASP-GenAI-Security-Project-Solutions-Reference-Guide-Q2_Q325.pdf", 4),
    ("OWASP-GenAI-Security-Project-Solutions-Reference-Guide-Q2_Q325.pdf", 6),
    ("PCI-DSS-v4_0_1-LA.pdf", 4),
    ("PCI-DSS-v4_0_1-LA.pdf", 5),
    ("PCI-DSS-v4_0_1-LA.pdf", 6),
    ("SAFEAI_Full_Report.pdf", 1),
    ("SAFEAI_Full_Report.pdf", 3),
    ("SAFEAI_Full_Report.pdf", 4),
    ("SAFEAI_Full_Report.pdf", 5),
}

_SYSTEM_PROMPT = """Clasificas fragmentos de documentos normativos y técnicos \
(normas ISO, reglamentos financieros, guías de seguridad).

Para cada fragmento decide si es FRONT_MATTER o CONTENIDO.

FRONT_MATTER es material editorial que rodea al documento pero no dice nada \
sustantivo sobre su materia:
- portadas, títulos, códigos de norma, fechas de edición
- índices y tablas de contenido
- avisos de copyright, licencias, descargos de responsabilidad
- listas de autores, participantes o miembros de comité
- historiales de revisión, agradecimientos, patrocinadores
- datos de contacto de la organización emisora

CONTENIDO es texto que aporta materia del documento: requisitos, controles, \
definiciones, procedimientos, análisis, recomendaciones, explicaciones \
técnicas. Un resumen ejecutivo es CONTENIDO. Una introducción que explica el \
tema es CONTENIDO.

Criterio decisivo: ¿alguien que pregunta por el tema del documento querría \
que esto apareciera como respuesta? Si solo describe quién publicó el \
documento o cómo está organizado, es FRONT_MATTER.

Responde SOLO con un objeto JSON:
{"results": [{"id": <int>, "label": "FRONT_MATTER"|"CONTENIDO", \
"reason": "<8 palabras como máximo>"}]}"""


def classify(client, items: list[tuple[int, str]]) -> dict[int, tuple[str, str]]:
    """Classify a batch of (id, text). Returns {id: (label, reason)}."""
    blocks = "\n\n".join(
        f"--- Fragmento {i} ---\n{text[:1200]}" for i, text in items
    )
    completion = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": blocks},
        ],
        temperature=0.0,
        response_format={"type": "json_object"},
        max_tokens=800,
    )
    parsed = json.loads(completion.choices[0].message.content or "{}")
    out = {}
    for r in parsed.get("results", []):
        try:
            out[int(r["id"])] = (str(r.get("label", "")).upper(),
                                 str(r.get("reason", ""))[:60])
        except (KeyError, ValueError, TypeError):
            continue
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Test LLM front-matter classification on the labelled set"
    )
    parser.add_argument("--verbose", action="store_true",
                        help="Show every chunk with its verdict")
    args = parser.parse_args()

    db = get_supabase_client()
    client = openai_sdk.OpenAI(api_key=settings.openai_api_key)

    docs = {d["filename"]: d["id"]
            for d in db.table("documents").select("id, filename").execute().data}

    samples = []  # (id, filename, chunk_index, text, truth)
    for label_set, truth in ((FRONT_MATTER, "FRONT_MATTER"),
                             (BODY_LOW_INDEX, "CONTENIDO")):
        for filename, idx in sorted(label_set):
            doc_id = docs.get(filename)
            if not doc_id:
                continue
            rows = (
                db.table("document_chunks")
                .select("content")
                .eq("document_id", doc_id)
                .eq("chunk_index", idx)
                .execute()
                .data
            )
            if rows:
                samples.append((len(samples), filename, idx,
                                rows[0]["content"] or "", truth))

    print(SEP)
    print("  LLM front-matter classification — labelled set")
    print(SEP)
    print(f"  model    : {MODEL} (temperature 0)")
    print(f"  samples  : {len(samples)} "
          f"({sum(1 for s in samples if s[4] == 'FRONT_MATTER')} front matter, "
          f"{sum(1 for s in samples if s[4] == 'CONTENIDO')} body)")

    verdicts: dict[int, tuple[str, str]] = {}
    for start in range(0, len(samples), BATCH):
        batch = samples[start:start + BATCH]
        verdicts.update(classify(client, [(s[0], s[3]) for s in batch]))

    tp = fp = tn = fn = 0
    errors = []
    for sid, filename, idx, _text, truth in samples:
        got, reason = verdicts.get(sid, ("<sin respuesta>", ""))
        ok = (got == truth)
        if truth == "FRONT_MATTER":
            tp += ok
            fn += not ok
        else:
            tn += ok
            fp += not ok
        if not ok:
            errors.append((filename, idx, truth, got, reason))
        if args.verbose:
            print(f"  {'ok ' if ok else 'ERR'} {filename[:38]:38} #{idx:<4} "
                  f"real={truth:12} llm={got:12} {reason}")
    print()
    print(SEP)
    print("  Results")
    print(SEP)
    total = len(samples)
    print(f"  correct               : {tp + tn}/{total}")
    print(f"  front matter detected : {tp}/{tp + fn}")
    print(f"  body misclassified    : {fp}/{fp + tn}")
    if errors:
        print("\n  Disagreements:")
        for filename, idx, truth, got, reason in errors:
            print(f"    {filename[:42]:42} #{idx:<4} real={truth:12} "
                  f"llm={got:12} {reason}")
    else:
        print("\n  No disagreements.")


if __name__ == "__main__":
    main()
