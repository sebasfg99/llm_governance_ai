# ============================================================
# app/api/routes/feedback.py — Feedback endpoints
# ============================================================

from fastapi import APIRouter, Depends
from supabase import Client

from app.api.deps import get_current_user, get_db
from app.domain.feedback.service import FeedbackService
from app.schemas.feedback import FeedbackSubmitRequest, FeedbackResponse
from app.domain.feedback.models import CreateFeedbackRequest

router = APIRouter()


@router.post(
    "",
    response_model=FeedbackResponse,
    summary="Submit feedback",
    description="Submit a 1-5 star rating and optional comment for an AI response.",
    tags=["Feedback"],
)
def submit_feedback(
    request: FeedbackSubmitRequest,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> FeedbackResponse:
    """Submit user feedback for an AI response trace."""
    service = FeedbackService(db)
    feedback = service.submit_feedback(
        request=CreateFeedbackRequest(
            trace_id=request.trace_id,
            rating=request.rating,
            comment=request.comment,
        ),
        tenant_id=current_user["tenant_id"],
        user_id=current_user["user_id"],
    )

    return FeedbackResponse(
        id=feedback.id,
        trace_id=feedback.trace_id,
        rating=feedback.rating,
        comment=feedback.comment,
        created_at=feedback.created_at,
    )
