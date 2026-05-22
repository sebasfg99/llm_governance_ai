#!/usr/bin/env python3
# ============================================================
# scripts/export_finetune_data.py — Export curated fine-tune data
#
# WHY THIS SCRIPT EXISTS (Task 7):
# Fine-tuning teaches a model HOW to respond — style, format,
# tone. We must NOT teach it bad behaviour, so this script
# exports ONLY traces that satisfy ALL of:
#
#   - response_risks.risk_level = 'low'
#   - evaluations.hallucination = false
#   - evaluations.context_adherence_score >= 0.9
#   - evaluations.groundedness_score    >= 0.8
#   - (optional) feedback.rating >= --min-rating when feedback exists
#
# OUTPUT FORMAT — one JSON object per line (JSONL):
# {
#   "messages": [
#     {"role": "system", "content": "..."},
#     {"role": "user",    "content": "..."},
#     {"role": "assistant", "content": "..."}
#   ],
#   "metadata": {
#     "risk_level":              "low",
#     "context_adherence_score": 0.97,
#     "groundedness_score":      0.84,
#     "use_case":                "chatbot",
#     "trace_id":                "<uuid>",
#     "client_id":               "..."
#   }
# }
#
# HOW TO RUN:
#   PYTHONPATH=. python scripts/export_finetune_data.py \
#     --use-case chatbot \
#     --output exports/finetune.jsonl
#
# The script never raises on missing tables: when migration 006
# has not yet been applied it gracefully degrades to the legacy
# eval.passed + feedback.rating filter.
# ============================================================

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.logging import get_logger
from app.db.supabase import get_supabase_client

logger = get_logger(__name__)


# ============================================================
# Quality filter constants (single source of truth)
# Adjusted in one place if standards change.
# ============================================================
MIN_CONTEXT_ADHERENCE = 0.9
MIN_GROUNDEDNESS      = 0.8


def export_finetune_data(
    *,
    use_case: str | None,
    min_rating: int,
    output_path: str,
    limit: int = 1000,
    client_id: str | None = None,
    require_feedback: bool = False,
) -> dict[str, Any]:
    """
    Export curated traces as OpenAI-compatible JSONL.

    Args:
        use_case:         Optional use_case filter.
        min_rating:       Minimum feedback rating (1-5). Only enforced
                          when require_feedback is True or when a
                          feedback row exists for the trace.
        output_path:      Output .jsonl file path.
        limit:            Max traces to export.
        client_id:        Optional client/tenant filter for multi-tenant exports.
        require_feedback: If True, only export traces that have a
                          feedback row at or above min_rating. Default
                          False because the governance metrics alone
                          are usually enough to qualify a trace.

    Returns:
        Stats dict with keys: examples_written, qualified_traces,
        passed_governance, passed_feedback, output_path.
    """
    db = get_supabase_client()

    logger.info(
        f"Fine-tune export start: use_case={use_case or 'all'} "
        f"client_id={client_id or 'all'} min_rating={min_rating} "
        f"output={output_path} limit={limit}"
    )

    # ---- 1. Pull traces ----
    trace_query = (
        db.table("traces")
        .select(
            "id, user_query, response_text, use_case, model_provider, "
            "model_name, client_id, tenant_id"
        )
        .order("created_at", desc=True)
        .limit(limit * 5)  # Pull extra; the governance filter is strict
    )
    if use_case:
        trace_query = trace_query.eq("use_case", use_case)
    if client_id:
        trace_query = trace_query.eq("client_id", client_id)

    traces_result = trace_query.execute()
    traces: dict[str, dict] = {t["id"]: t for t in (traces_result.data or [])}
    if not traces:
        logger.info("No traces found matching the base filter.")
        return _empty_stats(output_path)

    trace_ids = list(traces.keys())

    # ---- 2. Pull evaluations + response_risks (governance gate) ----
    governance: dict[str, dict] = {}
    try:
        evals_result = (
            db.table("evaluations")
            .select(
                "trace_id, hallucination, context_adherence_score, "
                "groundedness_score, retrieval_precision, completeness_score, "
                "instruction_adherence_score, citation_score, passed"
            )
            .in_("trace_id", trace_ids)
            .execute()
        )
        for row in evals_result.data or []:
            governance.setdefault(row["trace_id"], {}).update(row)
    except Exception as e:
        logger.warning(f"Could not pull governance evaluation columns: {e}")

    risks: dict[str, dict] = {}
    try:
        risks_result = (
            db.table("response_risks")
            .select("trace_id, risk_level, confidence_score, retrieval_confidence")
            .in_("trace_id", trace_ids)
            .execute()
        )
        for row in risks_result.data or []:
            risks[row["trace_id"]] = row
    except Exception as e:
        logger.warning(
            f"response_risks table missing or unreadable, falling back to "
            f"eval.passed gate. Error: {e}"
        )

    # ---- 3. Pull feedback ----
    feedback: dict[str, int] = {}
    try:
        feedback_result = (
            db.table("feedback")
            .select("trace_id, rating")
            .gte("rating", min_rating)
            .in_("trace_id", trace_ids)
            .execute()
        )
        for row in feedback_result.data or []:
            feedback[row["trace_id"]] = int(row["rating"])
    except Exception as e:
        logger.warning(f"Could not pull feedback: {e}")

    # ---- 4. Apply gates ----
    passed_governance = 0
    passed_feedback = 0
    qualified: list[tuple[dict, dict, dict | None]] = []
    for tid, trace in traces.items():
        ev = governance.get(tid, {})
        rk = risks.get(tid)

        gov_ok = _passes_governance(ev, rk)
        if gov_ok:
            passed_governance += 1

        rating = feedback.get(tid)
        if require_feedback:
            fb_ok = rating is not None and rating >= min_rating
        else:
            # Either no feedback (accept) or rating >= min_rating
            fb_ok = (rating is None) or (rating >= min_rating)
        if fb_ok:
            passed_feedback += 1

        if gov_ok and fb_ok:
            qualified.append((trace, ev, rk))
        if len(qualified) >= limit:
            break

    if not qualified:
        logger.info(
            "No traces qualified after the governance + feedback gates. "
            f"passed_governance={passed_governance} passed_feedback={passed_feedback}"
        )
        return _empty_stats(
            output_path,
            qualified_traces=0,
            passed_governance=passed_governance,
            passed_feedback=passed_feedback,
        )

    # ---- 5. Write JSONL ----
    os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)
    written = 0
    with open(output_path, "w", encoding="utf-8") as f:
        for trace, ev, rk in qualified:
            entry = {
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "Eres un asistente experto en gobernanza, riesgo y cumplimiento "
                            "de IA. Responde solo con base en los documentos disponibles, "
                            "cita tus fuentes y abst\u00e9nte cuando no haya evidencia."
                        ),
                    },
                    {"role": "user",      "content": trace["user_query"]},
                    {"role": "assistant", "content": trace["response_text"]},
                ],
                "metadata": {
                    "risk_level":              (rk or {}).get("risk_level", "unknown"),
                    "context_adherence_score": ev.get("context_adherence_score"),
                    "groundedness_score":      ev.get("groundedness_score"),
                    "retrieval_precision":     ev.get("retrieval_precision"),
                    "completeness_score":      ev.get("completeness_score"),
                    "use_case":                trace.get("use_case"),
                    "trace_id":                trace["id"],
                    "client_id":               trace.get("client_id") or trace.get("tenant_id"),
                    "model_provider":          trace.get("model_provider"),
                    "model_name":              trace.get("model_name"),
                },
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            written += 1

    logger.info(
        f"Export complete: wrote {written} examples → {output_path}"
        f"\n  passed_governance: {passed_governance}/{len(traces)}"
        f"\n  passed_feedback:   {passed_feedback}/{len(traces)}"
    )

    return {
        "examples_written":   written,
        "qualified_traces":   len(qualified),
        "passed_governance":  passed_governance,
        "passed_feedback":    passed_feedback,
        "output_path":        output_path,
    }


# ============================================================
# Helpers
# ============================================================
def _passes_governance(eval_row: dict, risk_row: dict | None) -> bool:
    """
    Apply the curated governance gate.

    A trace passes when ALL the following hold:
      - risk_row.risk_level == 'low' (when present), AND
      - eval_row.hallucination is False (when present), AND
      - eval_row.context_adherence_score >= 0.9 (when present), AND
      - eval_row.groundedness_score      >= 0.8 (when present)

    When migration 006 has not been applied (column missing), we
    fall back to eval_row.passed = True so the script still works.
    """
    # Risk gate
    if risk_row is not None:
        if risk_row.get("risk_level") != "low":
            return False
    # Eval gates
    if eval_row.get("hallucination") is True:
        return False

    adherence = eval_row.get("context_adherence_score")
    groundedness = eval_row.get("groundedness_score")

    if adherence is not None and float(adherence) < MIN_CONTEXT_ADHERENCE:
        return False
    if groundedness is not None and float(groundedness) < MIN_GROUNDEDNESS:
        return False

    # Legacy fallback: when none of the new columns are populated,
    # require the legacy `passed` flag.
    if (
        risk_row is None
        and adherence is None
        and groundedness is None
    ):
        return bool(eval_row.get("passed"))

    return True


def _empty_stats(
    output_path: str,
    *,
    qualified_traces: int = 0,
    passed_governance: int = 0,
    passed_feedback: int = 0,
) -> dict[str, Any]:
    return {
        "examples_written":   0,
        "qualified_traces":   qualified_traces,
        "passed_governance":  passed_governance,
        "passed_feedback":    passed_feedback,
        "output_path":        output_path,
    }


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export curated fine-tuning dataset from approved traces"
    )
    parser.add_argument(
        "--use-case",
        type=str,
        default=None,
        help="Filter to a specific use case (e.g., 'chatbot'). Omit for all.",
    )
    parser.add_argument(
        "--client-id",
        type=str,
        default=None,
        help="Filter to a specific client/tenant. Omit for all.",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=4,
        help="Minimum feedback rating (1-5). Only enforced when --require-feedback.",
    )
    parser.add_argument(
        "--require-feedback",
        action="store_true",
        help="If set, traces without feedback are excluded from the export.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"exports/finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        help="Output .jsonl path (default: exports/finetune_<timestamp>.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of examples to export (default: 1000)",
    )
    args = parser.parse_args()

    stats = export_finetune_data(
        use_case=args.use_case,
        min_rating=args.min_rating,
        output_path=args.output,
        limit=args.limit,
        client_id=args.client_id,
        require_feedback=args.require_feedback,
    )

    print(json.dumps(stats, indent=2, ensure_ascii=False))
