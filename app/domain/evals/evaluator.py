# ============================================================
# app/domain/evals/evaluator.py — Public evaluation facade
#
# WHY THIS FILE EXISTS:
# The spec (Task 1) requires a single function:
#
#     evaluate_response(query, chunks, response, citations, instructions)
#
# returning the eight governance metrics. This module is that
# facade. It composes:
#
#   - The cheap heuristic graders in graders.py
#   - The LLM-as-judge in llm_judge.py (claim-level analysis)
#
# DESIGN NOTES:
#   - The function tolerates None/empty inputs and never raises.
#   - The judge is BEST-EFFORT: if it errors, the heuristic
#     graders fill in groundedness / context_adherence so the
#     pipeline still produces a usable evaluation.
#   - All scores are rounded to 4 decimals.
# ============================================================

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger

from app.domain.evals import graders
from app.domain.evals.llm_judge import judge_response

# Grounding level below which we flag hallucination WITHOUT a judge verdict.
#
# This is a last-resort proxy, used only when the LLM judge failed outright.
# RECALIBRATED for the containment formula: the previous value of 0.05 was
# tuned for Jaccard, where scores were compressed toward zero by context
# size. Under containment 0.05 means "almost no word of the response appears
# in the context at all", which essentially never happens for Spanish prose
# over a Spanish corpus — the check had become dead code.
#
# 0.30 flags the clearly-ungrounded cases from the calibration set (off-topic
# 0.2105, invented regulation 0.2667) while staying below a legitimate
# abstention (0.3750), because wrongly flagging hallucination is expensive:
# the response guardrails rewrite the user-visible answer when it fires.
# Deliberately more conservative than GROUNDING_THRESHOLD (0.40) — a coarse
# proxy should under-flag, not over-flag.
_HALLUCINATION_FALLBACK_GROUNDING = 0.30

logger = get_logger(__name__)


def evaluate_response(
    query: str,
    chunks: list,
    response: str,
    citations: list | None = None,
    instructions: str | None = None,
    *,
    use_llm_judge: bool = True,
    response_mode: str = "text",
) -> dict[str, Any]:
    """
    Run all RAG evaluation metrics on one response.

    Args:
        query:         The original user query.
        chunks:        The retrieved document chunks. Each entry can
                       be either a ChunkResult-like object (with
                       `.content`, `.similarity`, `.chunk_id`) or a
                       plain string. Mixed lists are tolerated.
        response:      The LLM's answer.
        citations:     Optional list of citation dicts/objects.
        instructions:  Optional system instructions the model was
                       supposed to follow.
        use_llm_judge: If False, skip the LLM judge. Useful for unit
                       tests and high-volume re-evaluation.
        response_mode: "text" or "structured" (passed to
                       instruction_adherence_score for JSON checks).

    Returns:
        A dict with the keys required by the spec:
          {
            "retrieval_precision":          float in [0,1],
            "groundedness_score":           float in [0,1],
            "context_adherence_score":      float in [0,1],
            "citation_score":               float in [0,1],
            "completeness_score":           float in [0,1],
            "instruction_adherence_score":  float in [0,1],
            "hallucination":                bool,
            "should_have_said_idk":         bool,
          }
        Plus extra keys for audit:
          "claims_total", "claims_supported", "judge_ok", "judge_raw"
    """
    chunk_texts = _extract_chunk_texts(chunks)

    # ---- Heuristic baselines (always computed) ----
    grounding = graders.grounding_score(response, chunk_texts)
    precision = graders.retrieval_precision(chunks)
    cit_score = graders.citation_score(response, citations, chunks)
    completeness = graders.completeness_score(query, response, chunks)
    instr_adherence = graders.instruction_adherence_score(
        response, instructions, response_mode=response_mode
    )

    # ---- LLM judge (best effort) ----
    if use_llm_judge and response and response.strip():
        judge_result = judge_response(
            query=query,
            response=response,
            chunks=chunk_texts,
            instructions=instructions,
        )
    else:
        judge_result = {
            "claims": [],
            "claims_total": 0,
            "claims_supported": 0,
            "hallucination": False,
            "should_have_said_idk": False,
            "raw": {"skipped": True},
            "ok": False,
        }

    claims_total = int(judge_result.get("claims_total", 0))
    claims_supported = int(judge_result.get("claims_supported", 0))

    # context_adherence = supported claims / total claims.
    # When the judge returns no claims (small talk, IDK) we cannot
    # compute the ratio, so we fall back to the heuristic grounding
    # score which is a reasonable proxy.
    if claims_total > 0:
        context_adherence = claims_supported / claims_total
    elif judge_result.get("ok") is False:
        context_adherence = grounding
    else:
        context_adherence = 1.0 if response else 0.0

    # Hallucination is the judge's call. If the judge failed, we
    # fall back to "very low grounding" as a coarse proxy so the
    # downstream guardrails / risk scorer still have a signal.
    hallucination = bool(judge_result.get("hallucination", False))
    if (
        not judge_result.get("ok")
        and grounding < _HALLUCINATION_FALLBACK_GROUNDING
        and chunk_texts
    ):
        hallucination = True

    should_have_said_idk = bool(judge_result.get("should_have_said_idk", False))

    return {
        "retrieval_precision":         round(precision, 4),
        "groundedness_score":          round(grounding, 4),
        "context_adherence_score":     round(context_adherence, 4),
        "citation_score":              round(cit_score, 4),
        "completeness_score":          round(completeness, 4),
        "instruction_adherence_score": round(instr_adherence, 4),
        "hallucination":               hallucination,
        "should_have_said_idk":        should_have_said_idk,
        # Audit fields (not in the spec output but written to DB)
        "claims_total":                claims_total,
        "claims_supported":            claims_supported,
        "judge_ok":                    bool(judge_result.get("ok")),
        "judge_raw":                   judge_result.get("raw"),
        "judge_reasoning":             judge_result.get("reasoning", ""),
    }


# ============================================================
# Helpers
# ============================================================
def _extract_chunk_texts(chunks: list | None) -> list[str]:
    """
    Normalise the chunk list into a list of strings the judge can
    consume. Accepts ChunkResult, dicts, and raw strings.
    """
    if not chunks:
        return []
    out: list[str] = []
    for c in chunks:
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, dict):
            text = c.get("content") or c.get("text") or ""
            out.append(str(text))
        else:
            text = getattr(c, "content", None) or getattr(c, "text", None) or ""
            out.append(str(text))
    return out
