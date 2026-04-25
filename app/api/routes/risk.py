# ============================================================
# app/api/routes/risk.py — Risk intake + risk analysis endpoints
#
# Two endpoints, one per step of the workflow:
#   POST /risk/{project_id}/intake   → next interview turn (B2)
#   POST /risk/{project_id}/analyze  → run the risk orchestrator (B3)
#
# Both live under /risk (not /projects/{id}/risk) to keep the URL
# short and the OpenAPI section self-contained. They do, however,
# all read/write through ProjectService so the project row stays
# the single source of truth for the risk_register.
# ============================================================

from fastapi import APIRouter, Depends
from supabase import Client

from app.api.deps import get_current_user, get_db
from app.domain.risk.models import (
    RiskIntakeRequest,
    RiskIntakeResponse,
    RunRiskAnalysisRequest,
    RunRiskAnalysisResponse,
)
from app.domain.risk.service import RiskService

router = APIRouter()


@router.post(
    "/{project_id}/intake",
    response_model=RiskIntakeResponse,
    summary="Run the next turn of the risk intake interview",
    tags=["Risk"],
)
def risk_intake(
    project_id: str,
    request: RiskIntakeRequest,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> RiskIntakeResponse:
    service = RiskService(db)
    return service.run_intake_turn(
        project_id=project_id,
        tenant_id=current_user["tenant_id"],
        user_id=current_user["user_id"],
        request=request,
    )


@router.post(
    "/{project_id}/analyze",
    response_model=RunRiskAnalysisResponse,
    summary="Run the multi-framework risk analysis orchestrator",
    tags=["Risk"],
)
def run_risk_analysis(
    project_id: str,
    request: RunRiskAnalysisRequest,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> RunRiskAnalysisResponse:
    service = RiskService(db)
    return service.run_risk_analysis(
        project_id=project_id,
        tenant_id=current_user["tenant_id"],
        user_id=current_user["user_id"],
        request=request,
    )
