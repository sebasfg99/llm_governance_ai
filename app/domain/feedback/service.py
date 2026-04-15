# ============================================================
# app/domain/feedback/service.py — Feedback collection service
# ============================================================

from datetime import datetime, timezone

from supabase import Client

from app.core.logging import get_logger
from app.core.errors import NotFoundError, ValidationError
from app.domain.feedback.models import Feedback, CreateFeedbackRequest

logger = get_logger(__name__)


class FeedbackService:
    """
    Manages user feedback on AI responses.

    Feedback links to a trace record (the specific AI response
    the user is rating). Ratings are used to:
    - Identify poor responses for review
    - Build a quality-filtered fine-tuning dataset
    - Track AI quality trends over time

    Usage:
        service = FeedbackService(db)
        feedback = service.submit_feedback(
            request=CreateFeedbackRequest(trace_id="...", rating=4),
            tenant_id="acme",
            user_id="u_123",
        )
    """

    def __init__(self, db: Client) -> None:
        self.db = db

    def submit_feedback(
        self,
        request: CreateFeedbackRequest,
        tenant_id: str,
        user_id: str,
    ) -> Feedback:
        """
        Submit a rating and optional comment for an AI response.

        Validates that the trace exists before inserting.
        The DB also enforces uniqueness: one feedback per (trace, user).

        Args:
            request:   The feedback data (trace_id, rating, comment).
            tenant_id: The tenant of the user submitting feedback.
            user_id:   The user submitting the feedback.

        Returns:
            The persisted Feedback record.

        Raises:
            NotFoundError:   If the trace_id doesn't exist.
            ValidationError: If the user already submitted feedback for this trace.
        """
        # Verify the trace exists
        trace_check = (
            self.db.table("traces")
            .select("id, tenant_id")
            .eq("id", request.trace_id)
            .execute()
        )
        if not trace_check.data:
            raise NotFoundError(
                f"Trace not found: {request.trace_id}",
                resource_id=request.trace_id,
            )

        # Insert feedback record
        try:
            result = self.db.table("feedback").insert({
                "trace_id": request.trace_id,
                "tenant_id": tenant_id,
                "user_id": user_id,
                "rating": request.rating,
                "comment": request.comment,
                "created_at": datetime.now(timezone.utc).isoformat(),
            }).execute()

        except Exception as e:
            # The DB has a UNIQUE constraint on (trace_id, user_id)
            if "unique" in str(e).lower() or "duplicate" in str(e).lower():
                raise ValidationError(
                    "You have already submitted feedback for this response."
                )
            raise

        if not result.data:
            raise RuntimeError("Failed to save feedback")

        logger.info(
            f"Feedback saved: trace={request.trace_id}, "
            f"rating={request.rating}, user={user_id}"
        )

        return Feedback(**result.data[0])

    def list_feedback(
        self,
        tenant_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Feedback]:
        """List all feedback for a tenant, newest first."""
        result = (
            self.db.table("feedback")
            .select("*")
            .eq("tenant_id", tenant_id)
            .order("created_at", desc=True)
            .range(offset, offset + limit - 1)
            .execute()
        )
        return [Feedback(**row) for row in result.data]
