from __future__ import annotations

# ============================================================
# app/domain/projects/service.py — Project expediente CRUD
#
# The service is intentionally boring: create / get / list / update.
# The interesting logic (filling a profile from an interview,
# generating a risk register, drafting a policy) lives in the
# specialised services (RiskService, PoliciesService) which read
# from and write to this one.
# ============================================================

from datetime import datetime, timezone
from typing import Optional

from supabase import Client

from app.core.logging import get_logger
from app.core.errors import NotFoundError
from app.domain.projects.models import (
    Project,
    ProjectProfile,
    ProjectStatus,
    CreateProjectRequest,
    UpdateProjectRequest,
)

logger = get_logger(__name__)


def _row_to_project(row: dict) -> Project:
    """
    Convert a Supabase row into a Project.

    project_profile comes back as a dict from JSONB; we coerce it
    into a ProjectProfile pydantic model to keep the service layer
    typed. Unknown fields inside the JSON are discarded by pydantic
    (model_config extra='ignore' in BaseModel defaults).
    """
    profile_raw = row.get("project_profile") or {}
    row = {**row, "project_profile": ProjectProfile(**profile_raw)}
    return Project(**row)


class ProjectService:
    """
    CRUD over the `projects` table.

    Tenant isolation: every method requires tenant_id and scopes all
    queries to it. There is no cross-tenant access path.
    """

    def __init__(self, db: Client) -> None:
        self.db = db

    # ------------------------------------------------------------
    # Read
    # ------------------------------------------------------------
    def get(self, project_id: str, tenant_id: str) -> Project:
        """
        Fetch a single project by id, enforcing tenant isolation.

        Raises NotFoundError if the project does not exist OR if it
        exists but belongs to another tenant. Both cases look the
        same to the caller so we don't leak existence.
        """
        result = (
            self.db.table("projects")
            .select("*")
            .eq("id", project_id)
            .eq("tenant_id", tenant_id)
            .limit(1)
            .execute()
        )

        if not result.data:
            raise NotFoundError(
                f"Project not found: {project_id}",
                resource_id=project_id,
            )

        return _row_to_project(result.data[0])

    def get_optional(
        self, project_id: str | None, tenant_id: str
    ) -> Optional[Project]:
        """Same as get() but returns None instead of raising when missing.

        Used by AIService when the caller may or may not have provided
        a project_id — we don't want a not-found to explode the AI call.
        """
        if not project_id:
            return None
        try:
            return self.get(project_id, tenant_id)
        except NotFoundError:
            logger.warning(
                f"AIService referenced non-existent project_id={project_id} "
                f"for tenant={tenant_id}; continuing without project context."
            )
            return None

    def list(self, tenant_id: str, limit: int = 50, offset: int = 0) -> list[Project]:
        """Return projects for a tenant, newest updated first."""
        result = (
            self.db.table("projects")
            .select("*")
            .eq("tenant_id", tenant_id)
            .order("updated_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return [_row_to_project(row) for row in (result.data or [])]

    # ------------------------------------------------------------
    # Write
    # ------------------------------------------------------------
    def create(self, tenant_id: str, request: CreateProjectRequest) -> Project:
        """Insert a new project in draft status."""
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "tenant_id": tenant_id,
            "name": request.name,
            "description": request.description,
            "project_profile": request.project_profile.model_dump(),
            "status": "draft",
            "created_at": now,
            "updated_at": now,
        }
        result = self.db.table("projects").insert(payload).execute()

        if not result.data:
            raise RuntimeError("Failed to create project")

        project = _row_to_project(result.data[0])
        logger.info(f"Project created: {project.id} (tenant={tenant_id})")
        return project

    def update(
        self,
        project_id: str,
        tenant_id: str,
        request: UpdateProjectRequest,
    ) -> Project:
        """
        Partial update. Touches only the fields that are not None in
        the request. Always bumps updated_at so the list view sorts
        correctly.
        """
        # Confirm existence + tenant match before updating.
        self.get(project_id, tenant_id)

        payload: dict = {"updated_at": datetime.now(timezone.utc).isoformat()}
        if request.name is not None:
            payload["name"] = request.name
        if request.description is not None:
            payload["description"] = request.description
        if request.project_profile is not None:
            payload["project_profile"] = request.project_profile.model_dump()
        if request.risk_register is not None:
            payload["risk_register"] = request.risk_register
        if request.policy is not None:
            payload["policy"] = request.policy
        if request.controls is not None:
            payload["controls"] = request.controls
        if request.status is not None:
            payload["status"] = request.status

        result = (
            self.db.table("projects")
            .update(payload)
            .eq("id", project_id)
            .eq("tenant_id", tenant_id)
            .execute()
        )

        if not result.data:
            raise RuntimeError(f"Failed to update project {project_id}")

        return _row_to_project(result.data[0])

    # Convenience mutators used by the orchestrators.
    def set_profile(
        self, project_id: str, tenant_id: str, profile: ProjectProfile
    ) -> Project:
        """Used by RiskService after the intake interview completes."""
        return self.update(
            project_id,
            tenant_id,
            UpdateProjectRequest(project_profile=profile, status="intake_complete"),
        )

    def set_risk_register(
        self, project_id: str, tenant_id: str, register: dict
    ) -> Project:
        """Used by RiskService once the orchestrator emits the register."""
        return self.update(
            project_id,
            tenant_id,
            UpdateProjectRequest(risk_register=register, status="risks_analyzed"),
        )

    def set_policy(self, project_id: str, tenant_id: str, policy: dict) -> Project:
        return self.update(
            project_id,
            tenant_id,
            UpdateProjectRequest(policy=policy, status="policy_generated"),
        )

    def set_controls(self, project_id: str, tenant_id: str, controls: dict) -> Project:
        return self.update(
            project_id, tenant_id, UpdateProjectRequest(controls=controls)
        )
