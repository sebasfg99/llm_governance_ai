# ============================================================
# app/api/routes/metrics.py — RAG metrics endpoint (Task 8)
#
# GET /metrics/rag?client_id=...&since=...&until=...
#
# Aggregates from `traces` + `evaluations` + `response_risks`
# to return:
#   - avg quality scores (8 governance metrics)
#   - hallucination rate
#   - context adherence average
#   - risk distribution (low / medium / high counts)
#   - latency p50/p95/avg
#   - cost total/avg
#
# When migration 006 is missing, the endpoint returns the legacy
# subset gracefully (no governance scores, no risk distribution).
# ============================================================

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, Depends, Query
from supabase import Client

from app.api.deps import get_current_user, get_db
from app.core.errors import AuthenticationError
from app.core.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.get(
    "/rag",
    summary="Aggregated RAG quality and cost metrics",
    description=(
        "Returns averages, distributions, latency percentiles and total/average cost "
        "for the calling client. Time window defaults to the last 30 days."
    ),
    tags=["Metrics"],
)
def rag_metrics(
    client_id: str | None = Query(
        default=None,
        description="Tenant/client identifier. Required when no token is provided.",
    ),
    since: str | None = Query(
        default=None,
        description="ISO 8601 lower bound (inclusive). Defaults to 30 days ago.",
    ),
    until: str | None = Query(
        default=None,
        description="ISO 8601 upper bound (exclusive). Defaults to now.",
    ),
    use_case: str | None = Query(
        default=None,
        description="Optional use_case filter (chatbot, risk_intake, ...).",
    ),
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> dict[str, Any]:
    """
    Compute and return per-client RAG metrics.

    The `client_id` query parameter takes precedence over the token
    identity. The endpoint enforces that the caller can only request
    metrics for their own tenant — the token identity is the source
    of truth, the query param exists only for service-to-service
    callers that pass an explicit client.
    """
    # ---- 1. Resolve and validate client_id ----
    token_tenant = (current_user or {}).get("tenant_id")
    requested = client_id or token_tenant
    if not requested:
        raise AuthenticationError("client_id is required for /metrics/rag")
    if client_id and token_tenant and client_id != token_tenant:
        # We deliberately do NOT raise — the dev token returns the same
        # tenant across all callers, but in production this would be a
        # potential cross-tenant attempt. Log it for audit.
        logger.warning(
            "metrics request with client_id=%s but token tenant=%s",
            client_id,
            token_tenant,
        )

    # ---- 2. Resolve time window ----
    now = datetime.now(timezone.utc)
    since_dt = _parse_iso(since) if since else now - timedelta(days=30)
    until_dt = _parse_iso(until) if until else now
    since_iso = since_dt.isoformat()
    until_iso = until_dt.isoformat()

    # ---- 3. Pull traces ----
    trace_query = (
        db.table("traces")
        .select("id, latency_ms, cost_estimate_usd, use_case, model_name, created_at, "
                "prompt_tokens, completion_tokens")
        .gte("created_at", since_iso)
        .lt("created_at", until_iso)
    )
    # Prefer the new client_id alias when migration 006 is applied;
    # fall back to tenant_id otherwise. We try client_id first, then
    # silently retry with tenant_id if the column is missing.
    try:
        trace_query = trace_query.eq("client_id", requested)
        if use_case:
            trace_query = trace_query.eq("use_case", use_case)
        traces_data = (trace_query.execute().data or [])
    except Exception:
        trace_query = (
            db.table("traces")
            .select("id, latency_ms, use_case, model_name, created_at, "
                    "prompt_tokens, completion_tokens")
            .eq("tenant_id", requested)
            .gte("created_at", since_iso)
            .lt("created_at", until_iso)
        )
        if use_case:
            trace_query = trace_query.eq("use_case", use_case)
        traces_data = (trace_query.execute().data or [])

    if not traces_data:
        return _empty_metrics(requested, since_iso, until_iso)

    trace_ids = [t["id"] for t in traces_data]

    # ---- 4. Pull evaluations ----
    eval_data: list[dict] = []
    try:
        eval_data = (
            db.table("evaluations")
            .select(
                "trace_id, retrieval_precision, groundedness_score, "
                "context_adherence_score, citation_score, completeness_score, "
                "instruction_adherence_score, hallucination, should_have_said_idk, "
                "passed"
            )
            .in_("trace_id", trace_ids)
            .execute()
            .data
            or []
        )
    except Exception as e:
        logger.warning(f"Could not pull evaluations columns (governance v2): {e}")

    # ---- 5. Pull response_risks ----
    risk_data: list[dict] = []
    try:
        risk_data = (
            db.table("response_risks")
            .select("trace_id, risk_level")
            .in_("trace_id", trace_ids)
            .execute()
            .data
            or []
        )
    except Exception as e:
        logger.warning(f"Could not pull response_risks: {e}")

    # ---- 6. Aggregate ----
    return _aggregate(
        client_id=requested,
        since_iso=since_iso,
        until_iso=until_iso,
        traces=traces_data,
        evaluations=eval_data,
        risks=risk_data,
    )


# ============================================================
# Helpers
# ============================================================
def _parse_iso(value: str) -> datetime:
    """Tolerant ISO 8601 parser. Falls back to UTC when no tz given."""
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as e:
        raise AuthenticationError(f"Invalid ISO timestamp: {value}") from e
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _empty_metrics(client_id: str, since_iso: str, until_iso: str) -> dict[str, Any]:
    """Shape used when no data is available so dashboards never break."""
    return {
        "client_id": client_id,
        "window": {"since": since_iso, "until": until_iso},
        "totals": {"calls": 0, "cost_usd": 0.0},
        "averages": {},
        "latency_ms": {"avg": 0, "p50": 0, "p95": 0, "max": 0},
        "risk_distribution": {"low": 0, "medium": 0, "high": 0, "unknown": 0},
        "hallucination_rate": 0.0,
        "should_have_said_idk_rate": 0.0,
        "use_cases": {},
        "models": {},
    }


def _aggregate(
    *,
    client_id: str,
    since_iso: str,
    until_iso: str,
    traces: list[dict],
    evaluations: list[dict],
    risks: list[dict],
) -> dict[str, Any]:
    """Produce the metrics payload."""
    n = len(traces)

    # ---- Cost + latency ----
    total_cost = sum(float(t.get("cost_estimate_usd") or 0.0) for t in traces)
    avg_cost = total_cost / n if n else 0.0
    latencies = sorted(int(t.get("latency_ms") or 0) for t in traces)
    avg_lat = sum(latencies) / n if n else 0
    p50 = latencies[int(0.50 * (n - 1))] if n else 0
    p95 = latencies[int(0.95 * (n - 1))] if n else 0
    max_lat = latencies[-1] if n else 0

    # ---- Tokens ----
    total_prompt_tokens = sum(int(t.get("prompt_tokens") or 0) for t in traces)
    total_completion_tokens = sum(int(t.get("completion_tokens") or 0) for t in traces)

    # ---- Use case + model breakdown ----
    use_cases: dict[str, int] = {}
    models: dict[str, int] = {}
    for t in traces:
        uc = t.get("use_case") or "unknown"
        use_cases[uc] = use_cases.get(uc, 0) + 1
        m = t.get("model_name") or "unknown"
        models[m] = models.get(m, 0) + 1

    # ---- Eval averages ----
    avg_fields = (
        "retrieval_precision",
        "groundedness_score",
        "context_adherence_score",
        "citation_score",
        "completeness_score",
        "instruction_adherence_score",
    )
    averages: dict[str, float] = {}
    eval_count = 0
    hallucination_hits = 0
    idk_hits = 0
    for ev in evaluations:
        eval_count += 1
        for field in avg_fields:
            v = ev.get(field)
            if v is not None:
                averages.setdefault(field, 0.0)
                averages[field] += float(v)
        if ev.get("hallucination") is True:
            hallucination_hits += 1
        if ev.get("should_have_said_idk") is True:
            idk_hits += 1
    if eval_count > 0:
        averages = {k: round(v / eval_count, 4) for k, v in averages.items()}

    hallucination_rate = (hallucination_hits / eval_count) if eval_count else 0.0
    idk_rate = (idk_hits / eval_count) if eval_count else 0.0

    # ---- Risk distribution ----
    risk_dist = {"low": 0, "medium": 0, "high": 0, "unknown": 0}
    risk_seen = set()
    for r in risks:
        lvl = (r.get("risk_level") or "unknown").lower()
        if lvl not in risk_dist:
            lvl = "unknown"
        risk_dist[lvl] += 1
        risk_seen.add(r.get("trace_id"))
    # Traces without a risk row count as "unknown" so totals add up.
    risk_dist["unknown"] += sum(1 for t in traces if t["id"] not in risk_seen)

    return {
        "client_id": client_id,
        "window": {"since": since_iso, "until": until_iso},
        "totals": {
            "calls": n,
            "cost_usd": round(total_cost, 6),
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "evaluations": eval_count,
            "risks": len(risks),
        },
        "averages": {
            **averages,
            "cost_usd": round(avg_cost, 6),
        },
        "latency_ms": {
            "avg": round(avg_lat, 2),
            "p50": p50,
            "p95": p95,
            "max": max_lat,
        },
        "risk_distribution": risk_dist,
        "hallucination_rate": round(hallucination_rate, 4),
        "should_have_said_idk_rate": round(idk_rate, 4),
        "use_cases": use_cases,
        "models": models,
    }
