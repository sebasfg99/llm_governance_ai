#!/usr/bin/env python3
# ============================================================
# scripts/validate_privacidad_agent_e2e.py
#
# End-to-end validation of the Privacidad agent: drives the real
# POST /ai/query endpoint with agent_tags=["privacidad"], then reads
# back what the pipeline actually stored in Supabase and checks it.
#
# Third of the three agent validators, and the one with a different
# purpose. Normativa and Seguridad drove fixes to SHARED
# infrastructure — the semantic grounding metric, the reranker's
# span normalisation, the front-matter penalty, the IDK guard. This
# corpus took part in NONE of that calibration, so the question here
# is whether those fixes generalise, and whether this agent has
# problems of its own.
#
# CORPUS (verified 2026-08-14, not taken from the spec):
#   7 documents, 477 chunks
#     gdpr                   189   CELEX_32018R1725 (shared with
#                                  normativa + proveedores)
#     local_data_protection  288   six national laws
#
# THE STRUCTURAL DIFFERENCE FROM THE OTHER TWO AGENTS:
# this corpus is mostly COUNTRY-SCOPED. Five of its six local laws
# belong to countries other than Costa Rica, and the country filter
# returns "this country + global (country IS NULL)". So with
# country="costa-rica" — the value every validator uses — only
#
#     217 of 477 chunks (45%) are reachable
#
# against 83% for Normativa. The unreachable 260 are Nicaragua (125),
# Dominican Republic (57), El Salvador (40) and Panama (38). That is
# not a defect: a question about Costa Rican law should not be
# answered with Nicaraguan law. But it means this agent runs on an
# effective corpus of 217 chunks, and query expectations are written
# against that, not against the headline 477.
#
# CHECKS PER QUERY
#   1. frameworks      — at least ONE expected framework is present.
#                        Not exclusivity.
#   2. diversity cap   — see check_diversity_cap for why exceeding it
#                        is not always a failure.
#   3. bibliography    — substantive queries must not surface
#                        reference-list chunks; the control query must.
#                        NOTE: this corpus has ZERO flagged chunks, so
#                        the check reports INCONCLUSIVE rather than
#                        passing vacuously. See the Q5 comment.
#   4. grounding       — evaluations.grounding_score >= 0.62
#                        (GROUNDING_THRESHOLD, semantic scale since
#                        evaluator_version v3.semantic)
#   5. hallucination   — evaluations.hallucination is false
#
# HOW TO READ THE OUTPUT
#   ✓ pass   ✗ fail   — check skipped / inconclusive, with a reason
# A query PASSES when no check failed. Skipped checks never turn a
# run green on a technicality — they are reported as inconclusive.
#
# The LLM judge is stochastic, so checks 4 and 5 vary between runs on
# identical input. Run the script 2-3 times before concluding that a
# single failure is a real regression. Checks 1-3 are deterministic:
# a failure there is a genuine retrieval or corpus problem.
#
# HOW TO RUN (from project root):
#   PYTHONPATH=. .venv/bin/python scripts/validate_privacidad_agent_e2e.py
#   PYTHONPATH=. .venv/bin/python scripts/validate_privacidad_agent_e2e.py --verbose
#   API_BASE_URL=http://localhost:8001 PYTHONPATH=. .venv/bin/python \
#       scripts/validate_privacidad_agent_e2e.py
#
# Needs the API running and reachable at API_BASE_URL
# (default http://localhost:8000):
#   .venv/bin/uvicorn app.main:app --port 8001
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
# Thresholds, read from the code that actually enforces them so this
# script cannot drift away from production behaviour.
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

AGENT_TAGS = ["privacidad"]

SEP = "=" * 78


# ------------------------------------------------------------
# Test cases
# ------------------------------------------------------------
@dataclass
class TestCase:
    name: str
    query: str
    # Frameworks of which at least ONE must appear among the retrieved
    # chunks. Empty means "no framework expectation".
    expect_frameworks: list[str] = field(default_factory=list)
    # True  -> the question is ABOUT the references, so bibliography
    #          chunks SHOULD surface (the rerank penalty is waived).
    # False -> substantive question; bibliography must stay out.
    expect_bibliography: bool = False
    country: str | None = "costa-rica"
    industry: str | None = "cooperative"
    # Shown in the report when a check fails, to separate a real
    # regression from a known corpus limitation.
    note: str = ""


TEST_CASES = [
    # --- Substantive (4) ---
    TestCase(
        # GDPR core. The densest document in the corpus (189 chunks)
        # and the only one shared with normativa and proveedores.
        name="Sustantiva GDPR consentimiento",
        query=(
            "¿Qué requiere el GDPR sobre el consentimiento para el tratamiento "
            "de datos personales?"
        ),
        expect_frameworks=["gdpr"],
        expect_bibliography=False,
    ),
    TestCase(
        # The Costa Rican law, and the tightest case in the set: Ley
        # 8968 contributes only 28 chunks. If the country filter or the
        # ranking drowned it under the 189-chunk GDPR, this is where it
        # would show.
        name="Sustantiva Ley 8968",
        query=(
            "¿Qué exige la Ley 8968 de Costa Rica sobre protección de datos "
            "personales?"
        ),
        expect_frameworks=["local_data_protection"],
        expect_bibliography=False,
        note="Ley 8968 has 28 chunks against GDPR's 189 in the same pool.",
    ),
    TestCase(
        # Discrimination WITHIN one large document, rather than between
        # documents as in Seguridad. GDPR spans consent, transfers,
        # automated decisions, supervisory authorities... so this asks
        # whether the normalised ranking still isolates one topic inside
        # 189 topically varied chunks.
        name="Sustantiva GDPR perfilado",
        query=(
            "¿Qué dice el GDPR sobre decisiones automatizadas y perfilado "
            "mediante IA?"
        ),
        expect_frameworks=["gdpr"],
        expect_bibliography=False,
        note="Tests intra-document discrimination, not inter-document.",
    ),
    TestCase(
        # A topic both frameworks cover, so either is a correct answer.
        name="Sustantiva transferencias",
        query=(
            "¿Qué requisitos aplican para transferencias internacionales de "
            "datos personales?"
        ),
        expect_frameworks=["gdpr", "local_data_protection"],
        expect_bibliography=False,
    ),
    # --- Bibliography control (1) ---
    TestCase(
        # Control case. It CANNOT pass on this corpus: zero Privacidad
        # chunks carry is_reference_list, so check_bibliography reports
        # INCONCLUSIVE rather than inventing a verdict either way.
        #
        # That is itself the finding. GDPR's "considerandos" cite other
        # instruments heavily, but the classifier looks for a standalone
        # bibliography HEADING and a EUR-Lex regulation has none — its
        # citations are woven into recital prose. The detector was built
        # for standards documents (ISO, OWASP) with an explicit
        # BIBLIOGRAFÍA section, and this is a document shape it does not
        # cover. Documented, not fixed: the task forbids touching the
        # corpus, and fixing it means a new detector.
        name="Control bibliografía",
        query=(
            "¿Qué normas o instrumentos legales cita el GDPR en sus "
            "considerandos?"
        ),
        expect_frameworks=["gdpr"],
        expect_bibliography=True,
        note=(
            "0 flagged chunks in this corpus — EUR-Lex recitals cite inline, "
            "with no bibliography heading for the classifier to anchor on."
        ),
    ),
    # --- Country context (1) ---
    TestCase(
        # Costa Rica + financial sector. Verifies the country filter
        # surfaces Ley 8968 rather than falling back to GDPR alone.
        name="Contexto país",
        query=(
            "¿Qué normativa de protección de datos aplica a instituciones "
            "financieras en Costa Rica?"
        ),
        expect_frameworks=["local_data_protection"],
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
        "agent_tags": AGENT_TAGS,
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
        .select(
            "id, document_id, chunk_index, framework, section_ref, is_reference_list"
        )
        .in_("id", chunk_ids)
        .execute()
    ).data or []

    doc_ids = sorted({r["document_id"] for r in rows})
    docs = {}
    if doc_ids:
        doc_rows = (
            await db.table("documents")
            .select("id, filename, framework, country")
            .in_("id", doc_ids)
            .execute()
        ).data or []
        docs = {d["id"]: d for d in doc_rows}

    out = {}
    for row in rows:
        doc = docs.get(row["document_id"], {})
        out[str(row["id"])] = {
            **row,
            # document_chunks.framework is denormalised and can be NULL
            # on older rows; fall back to the parent document's value.
            "framework": row.get("framework") or doc.get("framework"),
            "filename": doc.get("filename", "?"),
            "country": doc.get("country"),
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
    """
    How many reference-list chunks the Privacidad corpus holds at all.

    Used to keep the bibliography check honest: with nothing flagged,
    "no bibliography surfaced" is true by construction and proves
    nothing.
    """
    result = (
        await db.table("document_chunks")
        .select("id", count="exact")
        .eq("is_reference_list", True)
        .overlaps("agent_tags", AGENT_TAGS)
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
        n = sum(1 for a in attrs.values() if a.get("framework") in hit)
        return Check(
            "frameworks",
            "pass",
            f"expected {hit} present ({n} chunk(s)); all retrieved: {found}",
        )
    return Check(
        "frameworks",
        "fail",
        f"expected one of {case.expect_frameworks}, retrieved: {found or 'none'}",
    )


def check_diversity_cap(attrs: dict[str, dict], top_k: int) -> Check:
    """
    The cap is a PREFERENCE, not a hard limit.

    apply_per_document_cap() returns `kept + overflow` — it reorders and
    drops nothing, so a document's chunks beyond the cap move to the back
    of the queue but stay eligible. When the candidate pool does not hold
    enough distinct documents to fill top-k, those overflow chunks are
    selected and a single document legitimately exceeds the cap.

    So exceeding it is only a real failure when there WERE other
    documents available to take the slot.
    """
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

    # Could the cap have been honoured at all?
    #
    # Capacity is sum(min(contributed, cap)), NOT len(docs) * cap. A
    # document that contributed fewer chunks than the cap had nothing
    # more to give: apply_per_document_cap fills `kept` with up to cap
    # per document BEFORE any overflow, so an over-cap document can
    # only reach the top-k once every other document's supply is
    # exhausted. If that sum falls short of top_k, overflow was forced.
    #
    # The naive len(docs) * cap version reported a FALSE FAILURE on
    # this corpus: GDPR contributes 29-36 chunks above threshold while
    # Ley 8968 contributes 1, so a top-5 of 4 GDPR + 1 Ley 8968 has
    # capacity 3 + 1 = 4 < 5 — unavoidable — yet 2 docs × 3 = 6 looked
    # like room to spare.
    #
    # NOTE: scripts/validate_seguridad_agent_e2e.py still carries the
    # naive version. It produced no false failures there (its
    # over-cap case had a single document, which both formulas handle),
    # but the fix should be ported.
    capacity = sum(min(n, MAX_CHUNKS_PER_DOC) for n in per_doc.values())
    if capacity < top_k:
        return Check(
            "diversity_cap",
            "skip",
            f"max {worst}/doc over the limit of {MAX_CHUNKS_PER_DOC}, but the "
            f"{len(per_doc)} matching document(s) could only supply {capacity} "
            f"chunk(s) within it (< top_k={top_k}), so overflow was unavoidable. "
            f"Expected behaviour of apply_per_document_cap (reorder, not "
            f"filter) — {summary}",
        )

    return Check(
        "diversity_cap",
        "fail",
        f"max {worst}/doc exceeds limit {MAX_CHUNKS_PER_DOC} with "
        f"{len(per_doc)} documents available — {summary}",
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
    # the corpus, "no bibliography surfaced" is true by construction.
    if corpus_flagged == 0:
        return Check(
            "bibliography",
            "skip",
            "INCONCLUSIVE: 0 Privacidad chunks carry is_reference_list, so "
            "the rerank penalty is inert here and this check cannot mean "
            "anything. Unlike the other agents, running "
            "scripts/backfill_reference_flags.py would NOT help: the "
            "classifier anchors on a standalone bibliography heading, and "
            "this corpus has none — EUR-Lex recitals cite other instruments "
            "inline, and the national laws cite nothing.",
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
            "control query surfaced NO reference chunks — either the penalty "
            "waiver did not fire for this phrasing, or the flagged chunk is "
            "not retrievable for it",
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
    extra = (
        f", judge groundedness={judged:.3f}"
        if isinstance(judged, (int, float))
        else ""
    )
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
        check_diversity_cap(attrs, top_k=len(chunks) or settings.retrieval_top_k),
        check_bibliography(case, attrs, corpus_flagged),
        check_grounding(evaluation),
        check_hallucination(evaluation),
    ]

    span = f"{max(sims):.2f}–{min(sims):.2f}" if sims else "n/a"
    print(f"  trace_id={trace_id}  chunks={len(chunks)}  similarity={span}")
    for check in checks:
        print(f"  {check.mark} {check.label}: {check.detail}")

    ok = _passed(checks)
    if not ok and case.note:
        print(f"  note: {case.note}")

    if verbose:
        print("\n  --- retrieved chunks ---")
        for c in chunks:
            a = attrs.get(c["chunk_id"], {})
            print(
                f"    sim={c.get('similarity', 0):.3f} "
                f"[{a.get('framework')}|{a.get('section_ref')}] "
                f"ref={a.get('is_reference_list')} "
                f"country={a.get('country')} "
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
    print("  Privacidad Agent E2E Validation")
    print(SEP)
    print(f"  API            : {API_BASE_URL}")
    print(f"  Supabase       : {SUPABASE_URL}")
    print(f"  agent_tags     : {AGENT_TAGS}")
    print(f"  diversity cap  : {MAX_CHUNKS_PER_DOC} chunks/doc (agent-scoped)")
    print(f"  grounding gate : {GROUNDING_THRESHOLD} (on grounding_score)")
    print(f"  reference pen. : x{settings.retrieval_reference_penalty}")

    db = await acreate_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)

    corpus_flagged = await count_flagged_chunks(db)
    print(f"  corpus flagged : {corpus_flagged} reference-list chunk(s) in Privacidad")
    if corpus_flagged == 0:
        print(
            "\n  WARNING: nothing in the Privacidad corpus is flagged as a\n"
            "  reference list, so the bibliography checks are reported as\n"
            "  inconclusive rather than passing on a technicality."
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
        description="End-to-end validation of the Privacidad agent via /ai/query"
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
