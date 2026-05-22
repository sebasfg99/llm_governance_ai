# ============================================================
# app/domain/observability/logger.py — Unified call logger
#
# WHY THIS FILE EXISTS:
# The spec (Task 4) asks for a `log_llm_call(...)` that records:
#   user_query, retrieved_chunks, response, model_used,
#   cost_estimate, latency, evaluation, risk, errors.
#
# Rather than create a brand-new `llm_logs` table that
# duplicates `traces` + `evaluations` + `response_risks`, we
# reuse the three tables (now extended in migration 006) and
# expose a single composable entrypoint that writes to all of
# them in the right order.
#
# HOW TO CALL:
#   logger = ObservabilityLogger(db)
#   logger.log_llm_call(
#       client_id=..., user_id=..., query=..., chunks=...,
#       response=..., model_provider=..., model_name=...,
#       prompt_tokens=..., completion_tokens=..., latency_ms=...,
#       evaluation=..., guardrails=..., risk=..., use_case=...,
#   )
#
# OR call the module-level `log_llm_call(db, ...)` for one-off use.
# ============================================================

from __future__ import annotations

from typing import Any

from supabase import Client

from app.core.logging import get_logger
from app.domain.evals.service import EvalService
from app.domain.observability.cost import estimate_cost
from app.domain.response_risk.service import ResponseRiskService
from app.domain.traces.models import CreateTraceRequest
from app.domain.traces.service import TraceService

logger = get_logger(__name__)


class ObservabilityLogger:
    """
    One-stop logger for an entire LLM call.

    Responsibilities (in order):
      1. Compute cost from tokens + model.
      2. Insert a `traces` row (legacy + new fields).
      3. Insert an `evaluations` row (legacy + 8 governance metrics).
      4. Insert a `response_risks` row.

    Each step is wrapped in try/except so an upstream failure in
    one of them does not bring down the chain. The trace always
    runs first because every other table references it via FK.
    """

    def __init__(self, db: Client) -> None:
        self.db = db
        self.trace_service = TraceService(db)
        self.eval_service = EvalService(db)
        self.risk_service = ResponseRiskService(db)

    def log_llm_call(
        self,
        *,
        client_id: str,
        tenant_id: str | None = None,
        user_id: str,
        query: str,
        prompt_text: str,
        use_case: str,
        chunks: list,
        response_text: str,
        model_provider: str,
        model_name: str,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
        latency_ms: int = 0,
        response_mode: str = "text",
        project_id: str | None = None,
        thread_id: str | None = None,
        evaluation: dict[str, Any] | None = None,
        guardrails: dict[str, Any] | None = None,
        risk: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        """
        Persist a complete audit record of one LLM call.

        Returns a dict with the keys:
          - "trace_id": str | None
          - "evaluation_id": str | None
          - "response_risk_id": str | None
          - "cost_estimate_usd": float
          - "errors": list[str]   (ones we caught while writing)

        The function NEVER raises. Persistence failures are logged
        and surfaced through the "errors" list so the caller can
        decide whether to alert / retry, but the user response is
        delivered regardless.
        """
        errors: list[str] = []
        cost = estimate_cost(
            provider=model_provider,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        # --- 1. trace ---
        trace_id: str | None = None
        try:
            chunk_ids = [
                getattr(c, "chunk_id", None)
                or (c.get("chunk_id") if isinstance(c, dict) else None)
                for c in (chunks or [])
            ]
            chunk_ids = [str(c) for c in chunk_ids if c]
            retrieved_content = "\n\n".join(
                getattr(c, "content", None)
                or (c.get("content", "") if isinstance(c, dict) else str(c))
                for c in (chunks or [])
            )
            risk_level = (risk or {}).get("risk_level")
            requires_disclaimer = bool((guardrails or {}).get("requires_disclaimer"))

            trace = self.trace_service.create_trace(
                CreateTraceRequest(
                    tenant_id=tenant_id or client_id,
                    user_id=user_id,
                    project_id=project_id,
                    thread_id=thread_id,
                    user_query=query,
                    prompt_text=prompt_text,
                    use_case=use_case,
                    model_provider=model_provider,
                    model_name=model_name,
                    retrieved_chunk_ids=chunk_ids,
                    retrieved_content=retrieved_content,
                    response_text=response_text,
                    response_mode=response_mode,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_estimate_usd=cost,
                    risk_level=risk_level or "medium",
                    requires_disclaimer=requires_disclaimer,
                    client_id=client_id,
                )
            )
            trace_id = str(trace.id)
        except Exception as e:
            errors.append(f"trace_persist_error: {e}")
            logger.error(f"Failed to persist trace: {e}", exc_info=True)

        # --- 2. evaluation ---
        evaluation_id: str | None = None
        if trace_id and response_text:
            try:
                ev = self.eval_service.evaluate_trace(
                    trace_id=trace_id,
                    response_text=response_text,
                    chunk_texts=[
                        (
                            getattr(c, "content", None)
                            or (c.get("content", "") if isinstance(c, dict) else str(c))
                        )
                        for c in (chunks or [])
                    ],
                    response_mode=response_mode,
                    query=query,
                    chunks=chunks,
                    instructions=None,
                    use_llm_judge=evaluation is None,
                )
                evaluation_id = ev.id
            except Exception as e:
                errors.append(f"evaluation_persist_error: {e}")
                logger.warning(f"Evaluation persistence failed: {e}")

        # --- 3. response risk ---
        risk_id: str | None = None
        if trace_id and risk:
            try:
                row = self.risk_service.persist(
                    trace_id=trace_id,
                    tenant_id=tenant_id or client_id,
                    client_id=client_id,
                    risk=risk,
                )
                if row:
                    risk_id = str(row.get("id"))
            except Exception as e:
                errors.append(f"risk_persist_error: {e}")
                logger.warning(f"Risk persistence failed: {e}")

        if error:
            errors.append(f"upstream_error: {error}")

        return {
            "trace_id": trace_id,
            "evaluation_id": evaluation_id,
            "response_risk_id": risk_id,
            "cost_estimate_usd": cost,
            "errors": errors,
        }


def log_llm_call(db: Client, **kwargs: Any) -> dict[str, Any]:
    """
    Module-level convenience wrapper around ObservabilityLogger.

    Useful for scripts and tests that want a one-liner without
    instantiating the class.
    """
    return ObservabilityLogger(db).log_llm_call(**kwargs)
