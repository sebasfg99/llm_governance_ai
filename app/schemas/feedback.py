# ============================================================
# app/schemas/feedback.py — Feedback API schemas
# ============================================================

from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID


class FeedbackSubmitRequest(BaseModel):
    """Request body for POST /feedback."""
    trace_id: str = Field(..., description="The trace ID of the response being rated")
    rating: int = Field(..., ge=1, le=5, description="Rating: 1 (bad) to 5 (excellent)")
    comment: str | None = Field(None, max_length=2000, description="Optional comment")


class FeedbackResponse(BaseModel):
    """Response after successfully submitting feedback."""
    id: UUID
    trace_id: UUID
    rating: int
    comment: str | None
    created_at: datetime
    message: str = "Thank you for your feedback!"
