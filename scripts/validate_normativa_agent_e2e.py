#!/usr/bin/env python3
# ============================================================
# scripts/validate_normativa_agent_e2e.py
#
# End-to-end validation of the Normativa agent: drives the real
# POST /ai/query endpoint, then reads back what the pipeline
# actually stored in Supabase and checks it.
#
# WHY IT READS THE DATABASE AND NOT JUST THE RESPONSE:
# AIQueryResponse.retrieved_chunks (RetrievedChunkInfo) carries
# only chunk_id, document_id, similarity and content_preview. The
# things worth validating — which framework a chunk belongs to,
# whether it is a bibliography chunk, its section_ref — are not in
# the payload at all. So every chunk_id is joined back to
# document_chunks / documents to get the real attributes, instead
# of inferring them from a 200-character preview.
#
# CHECKS PER QUERY
#   1. frameworks      — the expected normative framework is present
#   2. diversity cap   — no document contributes more than
#                        settings.agent_max_chunks_per_document (3,
#                        the agent-scoped regime)
#   3. bibliography    — substantive queries must not surface
#                        reference-list chunks; the control query
#                        must
#   4. grounding       — evaluations.grounding_score >= 0.40
#                        (GROUNDING_THRESHOLD, app/domain/evals/service.py)
#   5. hallucination   — evaluations.hallucination is false
#
# HOW TO RUN (from project root):
#   PYTHONPATH=. .venv/bin/python scripts/validate_normativa_agent_e2e.py
#   PYTHONPATH=. .venv/bin/python scripts/validate_normativa_agent_e2e.py --verbose
#   API_BASE_URL=https://<railway-host> PYTHONPATH=. .venv/bin/python \
#       scripts/validate_normativa_agent_e2e.py
#
# Needs the API running and reachable at API_BASE_URL
# (default http://localhost:8000):
#   .venv/bin/uvicorn app.main:app --port 8000
# ============================================================

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import httpx
from dotenv import load_dotenv
from supabase import acreate_client

load_dotenv()

# ------------------------------------------------------------
# Thresholds, read from the code that actually enforces them so
# this script cannot drift away from production behaviour.
#
# NOTE: GROUNDING_THRESHOLD lives in app/domain/evals/service.py,
# not in graders.py — graders.py still carries stale comments
# quoting an older value of 0.15.
# ------------------------------------------------------------
from app.core.config import settings  # noqa: E402
from app.domain.evals.service import GROUNDING_THRESHOLD  # noqa: E402

MAX_CHUNKS_PER_DOC = settings.agent_max_chunks_per_document

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000").rstrip("/")
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY", "")
AI_CORE_SERVICE_KEY = os.getenv("AI_CORE_SERVICE_KEY", "")

TENANT_ID = os.getenv("E2E_TENANT_ID", "e2e-validation")
USER_ID = os.getenv("E2E_USER_ID", "00000000-0000-0000-0000-000000000001")

REQUEST_TIMEOUT = 180.0

SEP = "=" * 78


# ------------------------------------------------------------
# Test cases
# ------------------------------------------------------------
@dataclass
class TestCase:
    name: str
    query: str
    #  Frameworks of which at least one must appear among the
    #  retrieved chunks. Empty means "no framework expectation".
    expect_frameworks: list[str] = field(default_factory=list)
    #  True  -> this question is ABOUT the references, so bibliography
    #           chunks SHOULD surface (the rerank penalty is waived).
    #  False -> substantive question; bibliography must stay out.
    expect_bibliography: bool = False
    country: str | None = "costa-rica"
    industry: str | None = "cooperative"


TEST_CASES = [
    TestCase(
        name="Sustantiva ISO",
        query="¿Qué exige ISO 42001 sobre evaluación de riesgos de IA?",
        expect_frameworks=["iso_42001"],
        expect_bibliography=False,
    ),
    TestCase(
        name="Sustantiva sesgo",
        query=(
            "¿Cómo aborda ISO 42001 el sesgo en sistemas de IA y la toma de "
            "decisiones asistida?"
        ),
        expect_frameworks=["iso_42001"],
        expect_bibliography=False,
    ),
    TestCase(
        name="Control bibliografía",
        query="¿Qué normas ISO cita el Anexo D de ISO 42001?",
        expect_frameworks=["iso_42001"],
        expect_bibliography=True,
    ),
    TestCase(
        name="Contexto país",
        query="¿Qué normativa de gobernanza de IA aplica en Costa Rica?",
        # Deliberately no framework expectation: this question is not
        # about one framework, and the corpus holds country-specific
        # normativa only as corporate_governance.
        expect_frameworks=[],
        expect_bibliography=False,
        country="costa-rica",
    ),
]


# ------------------------------------------------------------
# Check result plumbing
# ------------------------------------------------------------
@dataclass
class Check:
    label: str
    status: str  # "pass" | "fail" | "skip"
    detail: str

    @property
    def mark(self) -> str:
        return {"pass": "✓", "fail": "✗", "skip": "—"}[self.status]


def _passed(checks: list[Check]) -> bool:
    return all(c.status != "fail" for c in checks)


# ------------------------------------------------------------
# HTTP
# ------------------------------------------------------------
async def call_ai_query(client: httpx.AsyncClient, case: TestCase) -> dict:
    payload = {
        "query": case.query,
        "use_case": "chatbot",
        "agent_tags": ["normativa"],
        "top_k": 5,
        "include_global": True,
        "enable_general_fallback": True,
    }
    if case.country:
        payload["country"] = case.country
    if case.industry:
        payload["industry"] = case.industry

    response = await client.post(
        f"{API_BASE_URL}/ai/query",
        json=payload,
        headers={
            "Authorization": f"Bearer {AI_CORE_SERVICE_KEY}",
            "X-Tenant-Id": TENANT_ID,
            "X-User-Id": USER_ID,
        },
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


# ------------------------------------------------------------
# Supabase reads
# ------------------------------------------------------------
async def fetch_chunk_attributes(db, chunk_ids: list[str]) -> dict[str, dict]:
    """
    Join retrieved chunk ids back to their real attributes.

    Needed because the API response does not carry framework,
    section_ref or is_reference_list — see the module header.
    """
    if not chunk_ids:
        return {}

    rows = (
        await db.table("document_chunks")
        .select("id, document_id, chunk_index, framework, section_ref, is_reference_list")
        .in_("id", chunk_ids)
        .execute()
    ).data or []

    doc_ids = sorted({r["document_id"] for r in rows})
    docs = {}
    if doc_ids:
        doc_rows = (
            await db.table("documents")
            .select("id, filename, framework")
            .in_("id", doc_ids)
            .execute()
        ).data or []
        docs = {d["id"]: d for d in doc_rows}

    out = {}
    for row in rows:
        doc = docs.get(row["document_id"], {})
        out[str(row["id"])] = {
            **row,
            # document_chunks.framework is denormalised and can be NULL on
            # older rows; fall back to the parent document's value.
            "framework": row.get("framework") or doc.get("framework"),
            "filename": doc.get("filename", "?"),
        }
    return out


async def fetch_evaluation(db, trace_id: str, attempts: int = 5) -> dict | None:
    """
    Read the evaluation row for a trace.

    evaluate_trace() runs inline before /ai/query returns, so the row
    is normally present immediately; the retry only covers the gap
    between the insert and its visibility.
    """
    for attempt in range(attempts):
        rows = (
            await db.table("evaluations")
            .select("*")
            .eq("trace_id", trace_id)
            .execute()
        ).data or []
        if rows:
            return rows[0]
        await asyncio.sleep(0.6 * (attempt + 1))
    return None


async def count_flagged_chunks(db) -> int:
    """How many reference-list chunks exist in the corpus at all."""
    result = (
        await db.table("document_chunks")
        .select("id", count="exact")
        .eq("is_reference_list", True)
        .execute()
    )
    return int(getattr(result, "count", 0) or 0)


# ------------------------------------------------------------
# Checks
# ------------------------------------------------------------
def check_frameworks(case: TestCase, attrs: dict[str, dict]) -> Check:
    found = sorted({a["framework"] for a in attrs.values() if a.get("framework")})
    if not case.expect_frameworks:
        return Check("frameworks", "pass", f"retrieved: {found or 'none'}")
    hit = [f for f in case.expect_frameworks if f in found]
    if hit:
        sims = [a for a in attrs.values() if a.get("framework") in hit]
        return Check(
            "frameworks",
            "pass",
            f"expected {hit} present ({len(sims)} chunk(s)); all retrieved: {found}",
        )
    return Check(
        "frameworks",
        "fail",
        f"expected one of {case.expect_frameworks}, retrieved: {found or 'none'}",
    )


def check_diversity_cap(attrs: dict[str, dict]) -> Check:
    per_doc: dict[str, int] = {}
    for a in attrs.values():
        per_doc[a["filename"]] = per_doc.get(a["filename"], 0) + 1
    if not per_doc:
        return Check("diversity_cap", "skip", "no chunks retrieved")
    worst = max(per_doc.values())
    summary = ", ".join(f"{n}×{fn[:28]}" for fn, n in sorted(per_doc.items()))
    if worst <= MAX_CHUNKS_PER_DOC:
        return Check(
            "diversity_cap",
            "pass",
            f"max {worst}/doc (limit {MAX_CHUNKS_PER_DOC}) — {summary}",
        )
    return Check(
        "diversity_cap",
        "fail",
        f"max {worst}/doc exceeds limit {MAX_CHUNKS_PER_DOC} — {summary}",
    )


def check_bibliography(
    case: TestCase, attrs: dict[str, dict], corpus_flagged: int
) -> Check:
    refs = [
        f"{a['filename'][:24]}#{a['chunk_index']}"
        for a in attrs.values()
        if a.get("is_reference_list")
    ]

    # Guard against a vacuous result. With nothing flagged anywhere in
    # the corpus, "no bibliography surfaced" is true by construction and
    # proves nothing — and the control query cannot succeed at all.
    if corpus_flagged == 0:
        return Check(
            "bibliography",
            "skip",
            "INCONCLUSIVE: 0 chunks in the corpus carry is_reference_list. "
            "Run scripts/backfill_reference_flags.py — until then the rerank "
            "penalty is inert and this check cannot mean anything.",
        )

    if case.expect_bibliography:
        if refs:
            return Check(
                "bibliography",
                "pass",
                f"control query surfaced reference chunk(s): {refs}",
            )
        return Check(
            "bibliography",
            "fail",
            "control query surfaced NO reference chunks — the penalty waiver "
            "may not be firing for this phrasing",
        )

    if refs:
        return Check(
            "bibliography",
            "fail",
            f"substantive query contaminated by reference chunk(s): {refs}",
        )
    return Check("bibliography", "pass", "no reference-list chunks in top-k")


def check_grounding(evaluation: dict | None) -> Check:
    if evaluation is None:
        return Check("grounding", "fail", "no evaluations row found for trace")
    score = evaluation.get("grounding_score")
    if score is None:
        return Check("grounding", "fail", "evaluations.grounding_score is NULL")
    # grounding_score is the heuristic gated by GROUNDING_THRESHOLD.
    # groundedness_score is the separate LLM-judge metric — reported
    # alongside because the two are easy to confuse.
    judged = evaluation.get("groundedness_score")
    extra = f", judge groundedness={judged:.3f}" if isinstance(judged, (int, float)) else ""
    status = "pass" if score >= GROUNDING_THRESHOLD else "fail"
    return Check(
        "grounding",
        status,
        f"grounding_score={score:.3f} (threshold {GROUNDING_THRESHOLD}){extra}",
    )


def check_hallucination(evaluation: dict | None) -> Check:
    if evaluation is None:
        return Check("hallucination", "fail", "no evaluations row found for trace")
    flag = evaluation.get("hallucination")
    supported = evaluation.get("claims_supported")
    total = evaluation.get("claims_total")
    claims = f" ({supported}/{total} claims supported)" if total else ""
    if flag:
        return Check("hallucination", "fail", f"hallucination=True{claims}")
    return Check("hallucination", "pass", f"hallucination=False{claims}")


# ------------------------------------------------------------
# Runner
# ------------------------------------------------------------
async def run_case(
    client: httpx.AsyncClient,
    db,
    case: TestCase,
    corpus_flagged: int,
    verbose: bool,
) -> bool:
    print(f'\nQuery: "{case.query}"')

    try:
        body = await call_ai_query(client, case)
    except httpx.HTTPStatusError as e:
        print(f"  ✗ HTTP {e.response.status_code}: {e.response.text[:300]}")
        print("  FAIL")
        return False
    except (httpx.ConnectError, httpx.ReadTimeout) as e:
        print(f"  ✗ could not reach {API_BASE_URL}: {e}")
        print("  FAIL")
        return False

    trace_id = body.get("trace_id")
    chunks = body.get("retrieved_chunks") or []
    chunk_ids = [c["chunk_id"] for c in chunks]
    sims = [c.get("similarity") for c in chunks if c.get("similarity") is not None]

    attrs = await fetch_chunk_attributes(db, chunk_ids)
    evaluation = await fetch_evaluation(db, trace_id) if trace_id else None

    checks = [
        check_frameworks(case, attrs),
        check_diversity_cap(attrs),
        check_bibliography(case, attrs, corpus_flagged),
        check_grounding(evaluation),
        check_hallucination(evaluation),
    ]

    span = f"{max(sims):.2f}–{min(sims):.2f}" if sims else "n/a"
    print(f"  trace_id={trace_id}  chunks={len(chunks)}  similarity={span}")
    for check in checks:
        print(f"  {check.mark} {check.label}: {check.detail}")

    if verbose:
        print("\n  --- retrieved chunks ---")
        for c in chunks:
            a = attrs.get(c["chunk_id"], {})
            print(
                f"    sim={c.get('similarity', 0):.3f} "
                f"[{a.get('framework')}|{a.get('section_ref')}] "
                f"ref={a.get('is_reference_list')} "
                f"{a.get('filename', '?')[:32]}#{a.get('chunk_index')}"
            )
        print("\n  --- response ---")
        print(f"    {(body.get('response_text') or '')[:700]}")
        if evaluation:
            print("\n  --- evaluation row ---")
            for key in (
                "passed", "grounding_score", "groundedness_score",
                "context_adherence_score", "retrieval_precision",
                "hallucination", "should_have_said_idk",
                "claims_supported", "claims_total", "evaluator_version",
            ):
                print(f"    {key:26}: {evaluation.get(key)}")

    ok = _passed(checks)
    skipped = sum(1 for c in checks if c.status == "skip")
    verdict = "PASS" if ok else "FAIL"
    if ok and skipped:
        verdict += f" ({skipped} check(s) inconclusive)"
    print(f"  {verdict}")
    return ok


async def main_async(verbose: bool) -> int:
    missing = [
        name
        for name, value in (
            ("SUPABASE_URL", SUPABASE_URL),
            ("SUPABASE_SERVICE_KEY", SUPABASE_SERVICE_KEY),
            ("AI_CORE_SERVICE_KEY", AI_CORE_SERVICE_KEY),
        )
        if not value
    ]
    if missing:
        print(f"Missing required environment variable(s): {', '.join(missing)}")
        return 2

    print(SEP)
    print("  Normativa Agent E2E Validation")
    print(SEP)
    print(f"  API            : {API_BASE_URL}")
    print(f"  Supabase       : {SUPABASE_URL}")
    print(f"  agent_tags     : ['normativa']")
    print(f"  diversity cap  : {MAX_CHUNKS_PER_DOC} chunks/doc (agent-scoped)")
    print(f"  grounding gate : {GROUNDING_THRESHOLD}")
    print(f"  reference pen. : x{settings.retrieval_reference_penalty}")

    db = await acreate_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    corpus_flagged = await count_flagged_chunks(db)
    print(f"  corpus flagged : {corpus_flagged} reference-list chunk(s)")
    if corpus_flagged == 0:
        print(
            "\n  WARNING: nothing in the corpus is flagged as a reference list,\n"
            "  so the bibliography checks are reported as inconclusive rather\n"
            "  than passing on a technicality."
        )

    results: list[tuple[str, bool]] = []
    async with httpx.AsyncClient() as client:
        for index, case in enumerate(TEST_CASES, start=1):
            print()
            print(SEP)
            print(f"  Query {index}/{len(TEST_CASES)} — {case.name}")
            print(SEP)
            ok = await run_case(client, db, case, corpus_flagged, verbose)
            results.append((case.name, ok))

    passed = sum(1 for _n, ok in results if ok)
    print()
    print(SEP)
    print(f"  Summary: {passed}/{len(results)} queries PASSED")
    print(SEP)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")

    return 0 if passed == len(results) else 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end validation of the Normativa agent via /ai/query"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show full responses, retrieved chunk attributes and evaluation rows",
    )
    args = parser.parse_args()
    raise SystemExit(asyncio.run(main_async(verbose=args.verbose)))


if __name__ == "__main__":
    main()
