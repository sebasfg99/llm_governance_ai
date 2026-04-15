# ============================================================
# app/domain/feedback/models.py — Feedback domain models
# ============================================================

from datetime import datetime
from pydantic import BaseModel, Field
from uuid import UUID


class Feedback(BaseModel):
    """
    A user's rating and optional comment on an AI response.
    Maps to the `feedback` table.
    """

    id: UUID
    trace_id: UUID      # The response being rated
    tenant_id: str
    user_id: str
    rating: int         # 1 (terrible) to 5 (excellent)
    comment: str | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class CreateFeedbackRequest(BaseModel):
    """Input to submit feedback."""
    trace_id: str
    rating: int = Field(ge=1, le=5, description="Rating from 1 (bad) to 5 (excellent)")
    comment: str | None = None
