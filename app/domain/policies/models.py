# ============================================================
# app/domain/policies/models.py — Policy + controls schemas
#
# Shape contracts for the policy_generation (C1) and
# controls_generation (C2) orchestrators. They match the JSON
# output declared in their respective system prompts.
# ============================================================

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.ai import Citation


# ============================================================
# C1 — Policy
# ============================================================
class PolicySection(BaseModel):
    id: str
    title: str
    content_markdown: str


class Policy(BaseModel):
    title: str
    version: str = "1.0"
    sections: list[PolicySection] = Field(default_factory=list)
    iso_42001_coverage: list[str] = Field(default_factory=list)
    nist_functions_covered: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    review_cycle_months: int = 12


class GeneratePolicyRequest(BaseModel):
    """POST /policies/{project_id}/generate or POST /policies/generate"""
    country: str | None = None

    # When true, if the project already has a risk_register we inject
    # a short summary so the policy mentions the project's specific
    # risks. Default: true — usually more useful.
    include_project_context: bool = True


class GeneratePolicyResponse(BaseModel):
    project_id: str | None
    policy: Policy
    trace_id: str | None = None


# ============================================================
# C2 — Controls
# ============================================================
NistFunction = Literal["GOVERN", "MAP", "MEASURE", "MANAGE"]
Priority = Literal["alta", "media", "baja"]


class Control(BaseModel):
    id: str
    title: str
    description: str
    addresses_risks: list[str] = Field(default_factory=list)
    iso_42001_clause: str | None = None
    nist_function: NistFunction | None = None
    implementation_guidance: str | None = None
    owner_role: str | None = None
    evidence_required: list[str] = Field(default_factory=list)
    priority: Priority | None = None


class CoverageSummary(BaseModel):
    risks_addressed: list[str] = Field(default_factory=list)
    risks_not_addressed: list[str] = Field(default_factory=list)
    rationale_gap: str | None = None


class ControlsBundle(BaseModel):
    controls: list[Control] = Field(default_factory=list)
    coverage_summary: CoverageSummary = Field(default_factory=CoverageSummary)


class GenerateControlsRequest(BaseModel):
    """POST /policies/{project_id}/controls"""
    # The risk_register snapshot to use. When None, the service reads
    # the current one from projects.risk_register — that's the common
    # path. Passing one is useful for what-if analyses.
    risk_register_override: dict | None = None


class GenerateControlsResponse(BaseModel):
    project_id: str
    controls: ControlsBundle
    trace_id: str | None = None
