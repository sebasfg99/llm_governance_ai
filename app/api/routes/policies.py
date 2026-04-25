# ============================================================
# app/api/routes/policies.py — Policy + controls endpoints (C)
#
# Three entry points:
#
#   POST /policies/generate                    — draft a policy with no
#                                                 project (discover mode)
#   POST /policies/{project_id}/generate       — draft a policy for an
#                                                 existing project
#   POST /policies/{project_id}/controls       — derive controls from
#                                                 the project's
#                                                 risk_register
#
# Policy generation takes ~30s on average (long LLM call). The UI is
# expected to show a spinner and poll the project row once done.
# ============================================================

from fastapi import APIRouter, Depends
from supabase import Client

from app.api.deps import get_current_user, get_db
from app.domain.policies.models import (
    GenerateControlsRequest,
    GenerateControlsResponse,
    GeneratePolicyRequest,
    GeneratePolicyResponse,
)
from app.domain.policies.service import PoliciesService

router = APIRouter()


@router.post(
    "/generate",
    response_model=GeneratePolicyResponse,
    summary="Generate a policy without a project (discover mode)",
    tags=["Policies"],
)
def generate_policy_standalone(
    request: GeneratePolicyRequest,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> GeneratePolicyResponse:
    service = PoliciesService(db)
    return service.generate_policy(
        tenant_id=current_user["tenant_id"],
        user_id=current_user["user_id"],
        project_id=None,
        request=request,
    )


@router.post(
    "/{project_id}/generate",
    response_model=GeneratePolicyResponse,
    summary="Generate a policy bound to an existing project",
    tags=["Policies"],
)
def generate_policy_for_project(
    project_id: str,
    request: GeneratePolicyRequest,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> GeneratePolicyResponse:
    service = PoliciesService(db)
    return service.generate_policy(
        tenant_id=current_user["tenant_id"],
        user_id=current_user["user_id"],
        project_id=project_id,
        request=request,
    )


@router.post(
    "/{project_id}/controls",
    response_model=GenerateControlsResponse,
    summary="Generate controls from the project's risk register",
    tags=["Policies"],
)
def generate_controls_for_project(
    project_id: str,
    request: GenerateControlsRequest,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> GenerateControlsResponse:
    service = PoliciesService(db)
    return service.generate_controls(
        tenant_id=current_user["tenant_id"],
        user_id=current_user["user_id"],
        project_id=project_id,
        request=request,
    )
