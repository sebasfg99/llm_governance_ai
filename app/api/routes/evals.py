# ============================================================
# app/api/routes/evals.py — Evaluation endpoints
# ============================================================

from fastapi import APIRouter, Depends
from supabase import Client

from app.api.deps import get_current_user, get_db
from app.domain.evals.service import EvalService

router = APIRouter()


@router.get(
    "/{trace_id}",
    summary="Get evaluation for a trace",
    description="Return the automatic quality evaluation results for a specific AI response.",
    tags=["Evaluations"],
)
def get_evaluation(
    trace_id: str,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> dict:
    """Return the evaluation result for an AI response trace."""
    service = EvalService(db)
    eval_data = service.get_evaluation(trace_id)

    return {
        "trace_id": trace_id,
        "grounding_score": eval_data.get("grounding_score"),
        "length_ok": eval_data.get("length_ok"),
        "has_citations": eval_data.get("has_citations"),
        "passed": eval_data.get("passed"),
        "grader_details": eval_data.get("grader_details"),
        "created_at": eval_data.get("created_at"),
    }
