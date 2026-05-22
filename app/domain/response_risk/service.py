# ============================================================
# app/domain/response_risk/service.py — Persist response risks
#
# WHY THIS FILE EXISTS:
# compute_risk() is a pure function. This service is the thin
# wrapper that writes the result into the response_risks table
# created in migration 006. It is called by the observability
# logger after the trace is created.
# ============================================================

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from supabase import Client

from app.core.logging import get_logger
from app.domain.response_risk.models import ResponseRisk

logger = get_logger(__name__)


class ResponseRiskService:
    """Persist and read per-response risk records."""

    def __init__(self, db: Client) -> None:
        self.db = db

    def persist(
        self,
        *,
        trace_id: str,
        tenant_id: str,
        client_id: str | None,
        risk: dict[str, Any],
    ) -> dict | None:
        """
        Write a response_risks row.

        Args:
            trace_id:  The trace this risk row belongs to.
            tenant_id: The owning tenant.
            client_id: The client identifier (defaults to tenant_id).
            risk:      Output of compute_risk().

        Returns:
            The persisted row dict, or None if the write failed.
            Failures are logged but never raised — the API response
            is delivered regardless.
        """
        try:
            payload = {
                "trace_id":             trace_id,
                "tenant_id":            tenant_id,
                "client_id":            client_id or tenant_id,
                "risk_level":           risk["risk_level"],
                "confidence_score":     float(risk["confidence_score"]),
                "retrieval_confidence": float(risk.get("retrieval_confidence") or 0.0),
                "risk_reasons":         risk.get("risk_reasons") or [],
                "created_at":           datetime.now(timezone.utc).isoformat(),
            }
            result = self.db.table("response_risks").insert(payload).execute()
            row = result.data[0] if result.data else None
            if row:
                logger.debug(
                    f"Response risk persisted: trace={trace_id} level={risk['risk_level']}"
                )
            return row
        except Exception as e:
            logger.error(f"Failed to persist response_risk for trace {trace_id}: {e}")
            return None

    def get_for_trace(self, trace_id: str) -> ResponseRisk | None:
        """Return the response_risk linked to a trace, if any."""
        try:
            result = (
                self.db.table("response_risks")
                .select("*")
                .eq("trace_id", trace_id)
                .limit(1)
                .execute()
            )
            if result.data:
                return ResponseRisk(**result.data[0])
            return None
        except Exception as e:
            logger.warning(f"Failed to load response_risk for trace {trace_id}: {e}")
            return None
