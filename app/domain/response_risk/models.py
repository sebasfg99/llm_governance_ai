# ============================================================
# app/domain/response_risk/models.py — Per-response risk schemas
# ============================================================

from __future__ import annotations

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


RiskLevel = Literal["low", "medium", "high"]


class ResponseRisk(BaseModel):
    """
    Persisted shape of a per-response risk record.

    Mirrors the `response_risks` table created in migration
    006_rag_governance.sql. Returned by ResponseRiskService.persist
    and consumed by the metrics endpoint.
    """
    id: UUID
    trace_id: UUID
    tenant_id: str
    client_id: str

    risk_level: RiskLevel
    confidence_score: float = Field(ge=0.0, le=1.0)
    retrieval_confidence: float | None = None
    risk_reasons: list[str] = Field(default_factory=list)

    created_at: datetime

    class Config:
        from_attributes = True


class ComputeRiskResult(BaseModel):
    """
    In-memory shape returned by compute_risk(). The service
    persists this into the response_risks table verbatim.
    """
    risk_level: RiskLevel
    confidence_score: float
    retrieval_confidence: float
    risk_reasons: list[str]
