# ============================================================
# app/domain/response_risk/scorer.py — compute_risk()
#
# WHY THIS FILE EXISTS:
# After every AI response we need a single risk_level
# (low | medium | high) plus a confidence score and a list of
# human-readable reasons. The risk feeds:
#
#   - The response payload (so the UI can warn users)
#   - The traces table (for filtering / dashboards)
#   - The /metrics/rag endpoint
#   - The fine-tuning export filter (only low risk is exported)
#
# RULES (from the spec, Task 3):
#
# HIGH risk:
#   - hallucination = true
#   - context_adherence_score < 0.7
#   - groundedness low
#   - no citations
#   - sensitive data exposure
#   - legal output without disclaimer
#
# MEDIUM risk:
#   - partial grounding
#   - context adherence between 0.7 - 0.9
#   - incomplete answer
#
# LOW risk:
#   - strong grounding
#   - context adherence close to 1.0
#   - correct citations
#   - no violations
# ============================================================

from __future__ import annotations

from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)


# ---- Score thresholds ----
# Tunable, in one place so dashboards can document them.
HIGH_CONTEXT_ADHERENCE_FLOOR = 0.7
LOW_GROUNDEDNESS_THRESHOLD = 0.10
MEDIUM_CONTEXT_ADHERENCE_FLOOR = 0.9
COMPLETENESS_PARTIAL_FLOOR = 0.5
CITATION_LOW_FLOOR = 0.2


def compute_risk(
    evaluation: dict[str, Any] | None,
    guardrails: dict[str, Any] | None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Compute the per-response risk level and supporting reasons.

    Args:
        evaluation: Output of evaluate_response() — must contain
                    the 8 governance metrics. None is tolerated and
                    treated as "everything unknown" → medium risk.
        guardrails: Output of validate_response_guardrails() — must
                    contain `violations`, `requires_disclaimer`,
                    `pii_findings`. None is tolerated.
        metadata:   Optional context about the request. Recognised
                    keys:
                      - "use_case":    str
                      - "chunks":      list (used to compute
                                       retrieval_confidence)
                      - "has_disclaimer": bool (override)
                      - "client_id":   str
                      - "model_provider": str
                      - "model_name":  str
                      - "latency_ms":  int

    Returns:
        {
          "risk_level":           "low" | "medium" | "high",
          "confidence_score":     float in [0,1],
          "risk_reasons":         list[str],
          "retrieval_confidence": float in [0,1],
        }
    """
    evaluation = evaluation or {}
    guardrails = guardrails or {}
    metadata = metadata or {}

    reasons: list[str] = []

    # ---- Pull out signals (with safe defaults) ----
    hallucination = bool(evaluation.get("hallucination", False))
    context_adherence = _safe_float(evaluation.get("context_adherence_score"), 0.0)
    groundedness = _safe_float(evaluation.get("groundedness_score"), 0.0)
    citation = _safe_float(evaluation.get("citation_score"), 0.0)
    completeness = _safe_float(evaluation.get("completeness_score"), 0.5)
    instruction_adherence = _safe_float(
        evaluation.get("instruction_adherence_score"), 1.0
    )
    should_have_said_idk = bool(evaluation.get("should_have_said_idk", False))

    violations = guardrails.get("violations") or []
    pii_findings = guardrails.get("pii_findings") or []
    requires_disclaimer = bool(guardrails.get("requires_disclaimer", False))
    safe_response = guardrails.get("safe_response") or ""
    has_disclaimer = bool(metadata.get("has_disclaimer")) or (
        # The default disclaimer adds the sentinel emoji; if it's
        # present we know the disclaimer was appended.
        "Disclaimer legal" in safe_response or "\u26a0\ufe0f" in safe_response
    )

    # Retrieval confidence is the average chunk similarity, capped at 1.
    chunks = metadata.get("chunks") or []
    retrieval_confidence = _retrieval_confidence(chunks)

    # ---- HIGH risk ----
    if hallucination:
        reasons.append("hallucination_detected")

    if context_adherence < HIGH_CONTEXT_ADHERENCE_FLOOR:
        reasons.append(
            f"low_context_adherence ({context_adherence:.2f} < {HIGH_CONTEXT_ADHERENCE_FLOOR})"
        )

    if groundedness < LOW_GROUNDEDNESS_THRESHOLD and chunks:
        reasons.append(
            f"low_groundedness ({groundedness:.2f} < {LOW_GROUNDEDNESS_THRESHOLD})"
        )

    if citation <= 0.0 and chunks:
        reasons.append("no_citations")

    if pii_findings:
        reasons.append(
            f"pii_exposure ({len(pii_findings)} hits: "
            f"{','.join(sorted({p.get('type', '?') for p in pii_findings if isinstance(p, dict)}))})"
        )

    if requires_disclaimer and not has_disclaimer:
        reasons.append("legal_output_missing_disclaimer")

    # Any guardrail violation flagged as 'high' contributes to HIGH risk.
    high_severity_codes = {
        v.get("code")
        for v in violations
        if isinstance(v, dict) and v.get("severity") == "high"
    }
    if high_severity_codes:
        reasons.append(
            "guardrail_high_severity:" + ",".join(sorted(c for c in high_severity_codes if c))
        )

    is_high = bool(reasons)

    # ---- MEDIUM risk (only evaluated if NOT already high) ----
    medium_reasons: list[str] = []
    if not is_high:
        if HIGH_CONTEXT_ADHERENCE_FLOOR <= context_adherence < MEDIUM_CONTEXT_ADHERENCE_FLOOR:
            medium_reasons.append(
                f"partial_context_adherence ({context_adherence:.2f})"
            )

        if completeness < COMPLETENESS_PARTIAL_FLOOR:
            medium_reasons.append(f"incomplete_answer ({completeness:.2f})")

        if 0.0 < citation < CITATION_LOW_FLOOR and chunks:
            medium_reasons.append(f"weak_citations ({citation:.2f})")

        if should_have_said_idk:
            medium_reasons.append("should_have_abstained")

        if instruction_adherence < 0.7:
            medium_reasons.append(
                f"low_instruction_adherence ({instruction_adherence:.2f})"
            )

    is_medium = bool(medium_reasons) and not is_high

    # ---- Final level ----
    if is_high:
        level = "high"
    elif is_medium:
        level = "medium"
    else:
        level = "low"

    final_reasons = reasons if is_high else medium_reasons
    if not final_reasons:
        final_reasons = ["all_signals_clean"]

    confidence_score = _confidence_score(
        level=level,
        context_adherence=context_adherence,
        groundedness=groundedness,
        instruction_adherence=instruction_adherence,
        retrieval_confidence=retrieval_confidence,
    )

    logger.info(
        f"Risk computed: level={level} confidence={confidence_score:.3f} "
        f"reasons={final_reasons}"
    )

    return {
        "risk_level": level,
        "confidence_score": round(confidence_score, 4),
        "risk_reasons": final_reasons,
        "retrieval_confidence": round(retrieval_confidence, 4),
    }


# ============================================================
# Helpers
# ============================================================
def _safe_float(value: Any, default: float) -> float:
    """Coerce arbitrary input to float, falling back to default."""
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _retrieval_confidence(chunks: list) -> float:
    """
    Confidence in the retrieval set:
      mean(similarity) * (1 - exp(-n/3))

    The exponential dampener penalises tiny retrieval sets so a
    single high-similarity chunk does not produce a misleading
    confidence of 1.0.
    """
    if not chunks:
        return 0.0
    sims: list[float] = []
    for c in chunks:
        if isinstance(c, (int, float)):
            sims.append(float(c))
        elif isinstance(c, dict) and "similarity" in c:
            sims.append(float(c["similarity"]))
        else:
            s = getattr(c, "similarity", None)
            if s is not None:
                sims.append(float(s))
    if not sims:
        return 0.0
    avg = sum(sims) / len(sims)

    # Dampener: 1 chunk → 0.28; 3 chunks → 0.63; 5 chunks → 0.81; 10 → 0.96
    import math
    dampener = 1 - math.exp(-len(sims) / 3.0)
    return max(0.0, min(1.0, avg * dampener))


def _confidence_score(
    *,
    level: str,
    context_adherence: float,
    groundedness: float,
    instruction_adherence: float,
    retrieval_confidence: float,
) -> float:
    """
    Aggregate confidence number in [0, 1]. Higher = more confident
    that the response is safe to deliver.

    The score is a weighted average of the strongest indicators,
    further scaled by the level itself so a HIGH risk response
    has confidence well below 0.5.
    """
    base = (
        0.45 * context_adherence
        + 0.20 * groundedness
        + 0.20 * instruction_adherence
        + 0.15 * retrieval_confidence
    )
    if level == "high":
        return base * 0.4
    if level == "medium":
        return base * 0.7
    return base
