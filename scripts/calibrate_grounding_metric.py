#!/usr/bin/env python3
# ============================================================
# scripts/calibrate_grounding_metric.py
#
# Calibration study for the grounding gate, on a 30-sample set plus an
# off-topic control floor.
#
# WHY THIS EXISTS:
# scripts/diagnose_grounding_crosslingual.py established, on 10 samples,
# that the lexical grounding_score() cannot separate a grounded answer
# from an invented one once the evidence is in English, and that an
# embedding-based variant can. Ten samples and one fabricated answer are
# not enough to move a production threshold, so this widens the set and
# reports the numbers a threshold decision actually needs.
#
# SAMPLE DESIGN
#   A) real / Spanish-dominant corpus   (n=10)
#   B) real / English-dominant corpus   (n=10)
#   C) fabricated but plausible         (n=10)
#   D) off-topic control                (n=5)   — the floor. Without it
#      there is no reference for what "clearly ungrounded" scores.
#
# Group C is written with real domain vocabulary (cláusulas, controles,
# trazabilidad, cifrado...) and paired with a genuine retrieved context.
# A fabrication that used obviously alien words would make the lexical
# metric look better than it is.
#
# Real answers come from the live API so they are what the system
# actually produces. Answers already present in `traces` are reused.
#
# WHAT IT REPORTS
#   1. The 4x2 matrix: sample type x metric.
#   2. A threshold sweep for both metrics, with the operating point that
#      maximises separation (accepted real - accepted fabricated).
#   3. Projected effect on both e2e validators at the chosen threshold.
#
# It changes nothing: graders.py is untouched.
#
# HOW TO RUN (from project root, API up):
#   API_BASE_URL=http://localhost:8001 PYTHONPATH=. \
#       .venv/bin/python scripts/calibrate_grounding_metric.py
# ============================================================

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv

load_dotenv()

from app.db.supabase import get_supabase_client  # noqa: E402
from app.domain.documents.embeddings import embed_texts  # noqa: E402
from app.domain.evals.graders import grounding_score  # noqa: E402
from app.domain.evals.service import GROUNDING_THRESHOLD  # noqa: E402

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8001").rstrip("/")
AI_CORE_SERVICE_KEY = os.getenv("AI_CORE_SERVICE_KEY", "")
TENANT_ID = os.getenv("E2E_TENANT_ID", "e2e-validation")
USER_ID = os.getenv("E2E_USER_ID", "00000000-0000-0000-0000-000000000001")

SEP = "=" * 78
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".grounding_cal_cache.json")

# ------------------------------------------------------------
# A) Real answers expected to sit on Spanish-dominant evidence
# ------------------------------------------------------------
SPANISH_QUERIES = [
    ("¿Qué exige ISO 42001 sobre evaluación de riesgos de IA?", ["normativa"]),
    ("¿Cómo aborda ISO 42001 el sesgo en sistemas de IA y la toma de decisiones asistida?", ["normativa"]),
    ("¿Qué normas ISO cita el Anexo D de ISO 42001?", ["normativa"]),
    ("¿Qué normativa de gobernanza de IA aplica en Costa Rica?", ["normativa"]),
    ("¿Qué exige PCI DSS sobre gestión de logs?", ["seguridad"]),
    ("¿Qué normativa aplica en Costa Rica sobre ciberseguridad en entidades financieras?", ["seguridad"]),
    ("¿Qué obligaciones establece la Estrategia Nacional de Ciberseguridad de Costa Rica?", ["seguridad"]),
    ("¿Qué requisitos de gobierno corporativo exige CONASSIF a las entidades financieras?", ["normativa"]),
    ("¿Qué controles exige PCI DSS para el cifrado de datos de titulares de tarjetas?", ["seguridad"]),
    ("¿Qué establece ISO 42001 sobre la política de inteligencia artificial de la organización?", ["normativa"]),
]

# ------------------------------------------------------------
# B) Real answers expected to sit on English-dominant evidence
#    (OWASP, MITRE ATLAS / SAFEAI, PCI DSS English edition)
# ------------------------------------------------------------
ENGLISH_QUERIES = [
    ("Según los documentos de OWASP GenAI Security, ¿cómo se aborda el riesgo de prompt injection en aplicaciones con LLMs?", ["seguridad"]),
    ("¿Qué controles de encriptación aplican para datos usados en entrenamiento de IA?", ["seguridad"]),
    ("¿Qué tácticas describe MITRE ATLAS sobre model evasion?", ["seguridad"]),
    ("¿Qué referencias bibliográficas cita la Guía de Soluciones de Seguridad GenAI de OWASP?", ["seguridad"]),
    ("¿Qué amenazas describe OWASP para los sistemas agénticos de IA?", ["seguridad"]),
    ("¿Cómo se mitiga el envenenamiento de datos de entrenamiento en sistemas de machine learning?", ["seguridad"]),
    ("¿Qué controles recomienda el framework para proteger los modelos de IA contra extracción?", ["seguridad"]),
    ("¿Qué prácticas de MLOps y monitoreo de modelos se recomiendan para sistemas de IA en producción?", ["seguridad"]),
    ("¿Qué riesgos de cadena de suministro afectan a los componentes de inteligencia artificial?", ["seguridad"]),
    ("¿Qué verificaciones de seguridad propone OWASP para validar aplicaciones basadas en LLM?", ["seguridad"]),
]

# ------------------------------------------------------------
# C) Fabricated answers — plausible, domain vocabulary, unsupported.
#    Each is paired with the query whose real context it is scored
#    against, so the comparison is like-for-like.
# ------------------------------------------------------------
FABRICATED = [
    ("¿Qué exige ISO 42001 sobre evaluación de riesgos de IA?",
     "La cláusula 6.1.7 de ISO 42001 exige que la organización ejecute la evaluación del "
     "riesgo de IA con periodicidad trimestral y conserve los registros por un mínimo de "
     "siete años. El proceso debe ser aprobado por un comité de riesgo algorítmico "
     "integrado por al menos cinco miembros independientes."),
    ("¿Qué normativa de gobernanza de IA aplica en Costa Rica?",
     "El Reglamento 51-2024 del CONASSIF obliga a las entidades supervisadas a registrar "
     "todo sistema de inteligencia artificial ante la Superintendencia en un plazo de 30 "
     "días hábiles, y a designar un oficial de gobernanza algorítmica con dedicación "
     "exclusiva y reporte directo a la junta directiva."),
    ("¿Qué exige PCI DSS sobre gestión de logs?",
     "PCI DSS establece en el requisito 10.9 que los registros de auditoría deben "
     "conservarse por 24 meses en almacenamiento inmutable, con revisión diaria "
     "obligatoria por parte de un analista certificado QSA y notificación al adquirente "
     "dentro de las 6 horas siguientes a cualquier anomalía."),
    ("¿Cómo aborda ISO 42001 el sesgo en sistemas de IA y la toma de decisiones asistida?",
     "ISO 42001 obliga a medir el sesgo con la métrica de paridad demográfica y a "
     "mantenerla por debajo del 5% de desviación entre grupos protegidos. Si se supera "
     "ese umbral, el sistema debe suspenderse automáticamente y notificarse al regulador "
     "en un plazo máximo de 48 horas."),
    ("¿Qué tácticas describe MITRE ATLAS sobre model evasion?",
     "MITRE ATLAS clasifica la evasión de modelos en catorce técnicas numeradas de AML.T0100 "
     "a AML.T0113, y recomienda un umbral de robustez adversarial del 92% medido con la "
     "métrica CLEVER antes de autorizar el despliegue en entornos productivos."),
    ("Según los documentos de OWASP GenAI Security, ¿cómo se aborda el riesgo de prompt injection en aplicaciones con LLMs?",
     "OWASP clasifica la inyección de prompts como riesgo LLM01 y exige implementar un "
     "filtro de doble capa con validación sintáctica y semántica, alcanzando una tasa de "
     "detección mínima del 98,5% verificada mediante el conjunto de pruebas certificado "
     "OWASP-PI-Bench v3."),
    ("¿Qué normativa aplica en Costa Rica sobre ciberseguridad en entidades financieras?",
     "La Norma Técnica 45061-H obliga a las entidades financieras costarricenses a "
     "realizar pruebas de penetración semestrales por un tercero acreditado, mantener un "
     "centro de operaciones de seguridad con cobertura 24/7 y reportar incidentes al "
     "SICVECA dentro de las 4 horas posteriores a su detección."),
    ("¿Qué controles de encriptación aplican para datos usados en entrenamiento de IA?",
     "El estándar exige cifrado AES-256 en reposo y TLS 1.3 en tránsito para todo conjunto "
     "de entrenamiento, con rotación de llaves cada 90 días gestionada por un HSM "
     "certificado FIPS 140-3 nivel 4, y segregación criptográfica por cada versión del "
     "modelo entrenado."),
    ("¿Qué normas ISO cita el Anexo D de ISO 42001?",
     "El Anexo D de ISO 42001 cita las normas ISO/IEC 27017:2015, ISO/IEC 20000-1:2018, "
     "ISO 9001:2015 e ISO/IEC 33001:2015 como referencias normativas complementarias para "
     "la gestión integrada de sistemas de inteligencia artificial."),
    ("¿Qué referencias bibliográficas cita la Guía de Soluciones de Seguridad GenAI de OWASP?",
     "La guía cita como fuentes principales el marco NIST SP 800-218A, la publicación "
     "ENISA Threat Landscape for AI 2024, el estándar IEEE 7009-2024 sobre fail-safe en "
     "sistemas autónomos y el informe Gartner Magic Quadrant for AI Security Platforms."),
]

# ------------------------------------------------------------
# D) Off-topic control — the floor.
# ------------------------------------------------------------
OFF_TOPIC = [
    ("¿Qué exige ISO 42001 sobre evaluación de riesgos de IA?",
     "La receta tradicional requiere harina, huevos y mantequilla. Se mezcla todo hasta "
     "obtener una masa homogénea y se hornea a 180 grados durante 25 minutos."),
    ("¿Qué exige PCI DSS sobre gestión de logs?",
     "El equipo visitante ganó el partido por dos goles a uno en un estadio lleno. El "
     "delantero marcó en el minuto ochenta tras una jugada colectiva."),
    ("¿Qué tácticas describe MITRE ATLAS sobre model evasion?",
     "Para cultivar tomates conviene un suelo bien drenado y riego moderado. La cosecha "
     "llega entre los 70 y 90 días después del trasplante."),
    ("¿Qué normativa de gobernanza de IA aplica en Costa Rica?",
     "El vuelo sale a las nueve de la mañana desde la terminal dos. Se recomienda llegar "
     "con dos horas de antelación para el control de equipaje."),
    ("Según los documentos de OWASP GenAI Security, ¿cómo se aborda el riesgo de prompt injection en aplicaciones con LLMs?",
     "La sinfonía consta de cuatro movimientos y dura aproximadamente cuarenta minutos. "
     "El segundo movimiento es un adagio en si bemol mayor."),
]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def load_cache() -> dict:
    if os.path.exists(CACHE):
        with open(CACHE) as fh:
            return json.load(fh)
    return {}


def save_cache(cache: dict) -> None:
    with open(CACHE, "w") as fh:
        json.dump(cache, fh)


def fetch_context(db, query: str) -> tuple[str, list[str]] | None:
    """Most recent (response, chunk texts) stored for a query."""
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
    return (t.get("response_text") or ""), chunks


def ask_api(query: str, agent_tags: list[str]) -> None:
    """Generate a real answer so a trace exists for this query."""
    payload = {
        "query": query,
        "use_case": "chatbot",
        "agent_tags": agent_tags,
        "country": "costa-rica",
        "industry": "cooperative",
        "top_k": 5,
        "include_global": True,
        "enable_general_fallback": True,
    }
    httpx.post(
        f"{API_BASE_URL}/ai/query",
        json=payload,
        headers={
            "Authorization": f"Bearer {AI_CORE_SERVICE_KEY}",
            "X-Tenant-Id": TENANT_ID,
            "X-User-Id": USER_ID,
        },
        timeout=180.0,
    ).raise_for_status()


def english_share(chunks: list[str]) -> float:
    en = {"the", "of", "and", "to", "in", "for", "is", "are", "that", "with", "on", "as", "be", "by"}
    es = {"de", "la", "el", "los", "las", "que", "en", "para", "con", "por", "se", "es", "un", "una"}
    import re
    n_en = 0
    for c in chunks:
        w = re.sub(r"[^\w\s]", " ", c.lower()).split()
        if sum(1 for x in w if x in en) > sum(1 for x in w if x in es):
            n_en += 1
    return n_en / len(chunks) if chunks else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate the grounding gate")
    parser.add_argument("--refresh", action="store_true",
                        help="Re-ask the API even for queries that already have a trace")
    args = parser.parse_args()

    db = get_supabase_client()

    # --- make sure every query has a real answer to score ---
    all_real = [(q, t, "ES") for q, t in SPANISH_QUERIES] + \
               [(q, t, "EN") for q, t in ENGLISH_QUERIES]
    print("Preparing real answers (reusing stored traces where possible)...")
    for query, tags, _grp in all_real:
        if args.refresh or fetch_context(db, query) is None:
            print(f"  asking API: {query[:64]}...")
            try:
                ask_api(query, tags)
            except Exception as e:  # noqa: BLE001
                print(f"    failed: {e}")

    samples: list[tuple[str, str, str, list[str]]] = []  # (group, query, answer, chunks)
    for query, _tags, grp in all_real:
        got = fetch_context(db, query)
        if not got:
            print(f"  [skip] no context for {query[:50]}")
            continue
        answer, chunks = got
        if not answer or not chunks:
            continue
        label = "A) real / ES corpus" if grp == "ES" else "B) real / EN corpus"
        samples.append((label, query, answer, chunks))

    for query, answer in FABRICATED:
        got = fetch_context(db, query)
        if got:
            samples.append(("C) fabricated", query, answer, got[1]))
    for query, answer in OFF_TOPIC:
        got = fetch_context(db, query)
        if got:
            samples.append(("D) off-topic", query, answer, got[1]))

    # --- score everything ---
    scored = []
    for group, query, answer, chunks in samples:
        embs = embed_texts([answer] + chunks)
        av, cvs = embs[0], embs[1:]
        lex = grounding_score(answer, chunks)
        sem = max(cosine(av, c) for c in cvs)
        scored.append((group, query, lex, sem, english_share(chunks)))

    # --- 1. the matrix ---
    print()
    print(SEP)
    print("  1. Sample type x metric")
    print(SEP)
    print(f"  {'group':22} {'n':>3} {'lexical':>18} {'semantic':>18} {'ctx EN':>7}")
    print(f"  {'':22} {'':>3} {'mean [min–max]':>18} {'mean [min–max]':>18}")
    print("  " + "-" * 72)
    groups = ["A) real / ES corpus", "B) real / EN corpus", "C) fabricated", "D) off-topic"]
    stats = {}
    for g in groups:
        rows = [s for s in scored if s[0] == g]
        if not rows:
            continue
        lex = [r[2] for r in rows]
        sem = [r[3] for r in rows]
        stats[g] = (lex, sem)
        print(f"  {g:22} {len(rows):>3} "
              f"{statistics.mean(lex):>6.3f} [{min(lex):.2f}–{max(lex):.2f}] "
              f"{statistics.mean(sem):>6.3f} [{min(sem):.2f}–{max(sem):.2f}] "
              f"{statistics.mean(r[4] for r in rows):>7.2f}")

    # --- 2. threshold sweep ---
    print()
    print(SEP)
    print("  2. Threshold sweep")
    print(SEP)
    real = [s for s in scored if s[0].startswith(("A", "B"))]
    bad = [s for s in scored if s[0].startswith(("C", "D"))]
    print(f"  real n={len(real)}   ungrounded n={len(bad)} (fabricated + off-topic)")
    print()
    for metric_name, idx in (("lexical", 2), ("semantic", 3)):
        print(f"  {metric_name}:")
        print(f"    {'thr':>5} {'real kept':>10} {'ungrounded kept':>16} {'separation':>11}")
        best = None
        for i in range(20, 90, 5):
            thr = i / 100
            tp = sum(1 for s in real if s[idx] >= thr) / len(real)
            fp = sum(1 for s in bad if s[idx] >= thr) / len(bad)
            sep = tp - fp
            if best is None or sep > best[1]:
                best = (thr, sep, tp, fp)
            print(f"    {thr:>5.2f} {tp:>9.0%} {fp:>15.0%} {sep:>+11.2f}")
        print(f"    -> best separation at {best[0]:.2f}: "
              f"keeps {best[2]:.0%} of real, admits {best[3]:.0%} of ungrounded")
        print()

    # --- 3. projected validator impact ---
    print(SEP)
    print("  3. Projected effect on the e2e validators")
    print(SEP)
    NRM_QUERIES = [q for q, _ in SPANISH_QUERIES[:4]]
    SEG_QUERIES = [
        "Según los documentos de OWASP GenAI Security, ¿cómo se aborda el riesgo de prompt injection en aplicaciones con LLMs?",
        "¿Qué controles de encriptación aplican para datos usados en entrenamiento de IA?",
        "¿Qué tácticas describe MITRE ATLAS sobre model evasion?",
        "¿Qué exige PCI DSS sobre gestión de logs?",
        "¿Qué referencias bibliográficas cita la Guía de Soluciones de Seguridad GenAI de OWASP?",
        "¿Qué normativa aplica en Costa Rica sobre ciberseguridad en entidades financieras?",
    ]
    # Real answers only — a validator scores what the system produced, not
    # the fabricated probes used for calibration.
    real_by_query = {s[1]: s for s in scored if s[0].startswith(("A", "B"))}

    for name, qlist in (("Normativa", NRM_QUERIES), ("Seguridad", SEG_QUERIES)):
        rows = [real_by_query[q] for q in qlist if q in real_by_query]
        if not rows:
            continue
        lex_pass = sum(1 for r in rows if r[2] >= GROUNDING_THRESHOLD)
        print(f"  {name} ({len(rows)} queries)")
        print(f"    lexical @ {GROUNDING_THRESHOLD:.2f} (current): {lex_pass}/{len(rows)} pass")
        for thr in (0.50, 0.55, 0.60, 0.62, 0.65):
            n = sum(1 for r in rows if r[3] >= thr)
            print(f"    semantic @ {thr:.2f}          : {n}/{len(rows)} pass")
        print()

    # Fine sweep around the optimum, to answer "is 0.62 still right?"
    print(SEP)
    print("  4. Fine sweep for the semantic operating point")
    print(SEP)
    print(f"    {'thr':>5} {'real kept':>10} {'ungrounded kept':>16} {'separation':>11}")
    best = None
    t = 0.50
    while t <= 0.75001:
        tp = sum(1 for s in real if s[3] >= t) / len(real)
        fp = sum(1 for s in bad if s[3] >= t) / len(bad)
        sep = tp - fp
        if best is None or sep > best[1] + 1e-9:
            best = (t, sep, tp, fp)
        print(f"    {t:>5.2f} {tp:>9.0%} {fp:>15.0%} {sep:>+11.2f}")
        t += 0.01
    print(f"\n    -> optimum {best[0]:.2f} (separation {best[1]:+.2f}): "
          f"keeps {best[2]:.0%} of real, admits {best[3]:.0%} of ungrounded")


if __name__ == "__main__":
    main()
