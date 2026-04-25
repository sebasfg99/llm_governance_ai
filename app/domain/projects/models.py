# ============================================================
# app/domain/projects/models.py — Project expediente models
#
# The "project" is the central L2 memory unit of the platform.
# Everything the org does around a given AI initiative (intake,
# risk analysis, policy drafting, controls, chat threads, traces)
# binds to one project row.
#
# The three JSON payloads (project_profile, risk_register, policy,
# controls) are intentionally unstructured at the DB layer: their
# shape is enforced by the pydantic models below instead of by
# columns. That lets the prompts evolve without a migration every
# time we add an attribute.
# ============================================================

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ============================================================
# project_profile — filled during B1 (form) and B2 (interview)
# ============================================================

AIType = Literal[
    "llm",
    "computer_vision",
    "classification",
    "recommendation",
    "generative_other",
    "other",
]

AutonomyLevel = Literal["advisor", "human_in_loop", "autonomous"]
Deployment = Literal["internal", "external_clients", "public_open"]

# Data sensitivity tags. Multi-select in the UI.
DataTag = Literal[
    "pii",
    "phi",
    "pci",
    "ip_confidential",
    "public",
]


class ProjectProfile(BaseModel):
    """
    Structured description of an AI project. Stored as JSON in
    projects.project_profile. Every field is optional because the
    profile is filled incrementally (form then interview).
    """

    project_name: Optional[str] = None
    project_description: Optional[str] = None
    ai_type: Optional[AIType] = None
    data_used: list[DataTag] = Field(default_factory=list)
    industry: Optional[str] = None
    affected_audience: Optional[str] = None
    autonomy_level: Optional[AutonomyLevel] = None
    deployment: Optional[Deployment] = None
    country: Optional[str] = None
    existing_vendor: Optional[str] = None

    # Free-form notes the user can add during the interview.
    notes: Optional[str] = None


# ============================================================
# Project row models
# ============================================================

ProjectStatus = Literal[
    "draft",
    "intake_complete",
    "risks_analyzed",
    "policy_generated",
    "archived",
]


class Project(BaseModel):
    """Projects row as read from the DB."""

    id: UUID
    tenant_id: str
    name: str
    description: str | None = None
    project_profile: ProjectProfile = Field(default_factory=ProjectProfile)
    risk_register: dict | None = None
    policy: dict | None = None
    controls: dict | None = None
    status: ProjectStatus = "draft"
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class CreateProjectRequest(BaseModel):
    """Input to POST /projects."""

    name: str = Field(..., min_length=1, max_length=200)
    description: str | None = None
    project_profile: ProjectProfile = Field(default_factory=ProjectProfile)


class UpdateProjectRequest(BaseModel):
    """Input to PATCH /projects/{id}. Every field optional."""

    name: str | None = None
    description: str | None = None
    project_profile: ProjectProfile | None = None
    risk_register: dict | None = None
    policy: dict | None = None
    controls: dict | None = None
    status: ProjectStatus | None = None
