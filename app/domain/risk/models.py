# ============================================================
# app/domain/risk/models.py — Risk assessment domain models
#
# These schemas mirror the JSON contract defined in the
# risk_analysis system prompt (scripts/seed_prompts.py). Keeping
# a pydantic model here lets the service parse the LLM output
# and raise clear errors if the LLM returns garbage, instead of
# propagating a raw JSON dict everywhere.
# ============================================================

from typing import Literal

from pydantic import BaseModel, Field

from app.domain.projects.models import ProjectProfile
from app.schemas.ai import Citation


EuAiActTier = Literal["prohibido", "alto", "limitado", "mínimo", "minimo"]
NistFunction = Literal["GOVERN", "MAP", "MEASURE", "MANAGE"]


class Risk(BaseModel):
    id: str
    category: str
    source_framework: str | None = None
    source_reference: str | None = None
    description: str
    likelihood: int = Field(ge=1, le=5)
    impact: int = Field(ge=1, le=5)
    inherent_score: int
    suggested_controls: list[str] = Field(default_factory=list)
    nist_function: NistFunction | None = None
    citations: list[Citation] = Field(default_factory=list)


class RiskRegister(BaseModel):
    eu_ai_act_tier: EuAiActTier | None = None
    tier_rationale: str | None = None
    requires_dpia: bool = False
    applicable_frameworks: list[str] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    executive_summary: str | None = None


# ------------------------------------------------------------
# Request / response schemas for the /risk endpoints
# ------------------------------------------------------------
class RunRiskAnalysisRequest(BaseModel):
    """
    POST /risk/{project_id}/analyze

    Triggers the orchestrator. Optional overrides let the caller
    pass a profile even before it is persisted (useful for
    dry-run mode in the UI).
    """

    profile_override: ProjectProfile | None = None

    # Country slug for RAG country filtering (country-specific laws).
    country: str | None = None


class RunRiskAnalysisResponse(BaseModel):
    project_id: str
    risk_register: RiskRegister
    trace_id: str | None = None


class RiskIntakeRequest(BaseModel):
    """
    POST /risk/{project_id}/intake

    Used by the UI to drive the B2 conversational interview. The
    caller sends the next user turn; the service runs the chatbot
    pipeline with use_case='risk_intake' and thread scoping.
    """

    message: str
    thread_id: str | None = None


class RiskIntakeResponse(BaseModel):
    project_id: str
    thread_id: str
    assistant_message: str
    is_complete: bool
    trace_id: str | None = None
