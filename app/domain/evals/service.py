# ============================================================
# app/domain/evals/service.py — Evaluation service
#
# Runs all heuristic graders after each AI response and
# persists the evaluation result to the `evaluations` table.
# ============================================================

from datetime import datetime, timezone

from supabase import Client

from app.core.logging import get_logger
from app.core.errors import NotFoundError
from app.domain.evals.graders import grounding_score, json_valid, length_ok, has_citations

logger = get_logger(__name__)

# The minimum grounding score required to "pass" evaluation
# 0.15 is stricter than the original 0.10 — aligns with improved citation guidance
GROUNDING_THRESHOLD = 0.15


class Evaluation:
    """Represents one evaluation result for a trace."""
    def __init__(
        self,
        id: str,
        trace_id: str,
        grounding_score_val: float,
        length_ok_val: bool,
        has_citations_val: bool,
        passed: bool,
        grader_details: dict,
        created_at: str,
    ):
        self.id = id
        self.trace_id = trace_id
        self.grounding_score = grounding_score_val
        self.length_ok = length_ok_val
        self.has_citations = has_citations_val
        self.passed = passed
        self.grader_details = grader_details
        self.created_at = created_at

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "trace_id": self.trace_id,
            "grounding_score": self.grounding_score,
            "length_ok": self.length_ok,
            "has_citations": self.has_citations,
            "passed": self.passed,
            "grader_details": self.grader_details,
            "created_at": self.created_at,
        }


class EvalService:
    """
    Runs quality graders on AI responses and stores results.

    Called synchronously after TraceService.create_trace().
    Grader failures are logged but never surfaced to API callers.

    Usage:
        service = EvalService(db)
        eval_result = service.evaluate_trace(
            trace_id="...",
            response_text="The contract states...",
            chunk_texts=["Section 8: Termination..."],
        )
        print(f"Passed: {eval_result.passed}")
    """

    def __init__(self, db: Client) -> None:
        self.db = db

    def evaluate_trace(
        self,
        trace_id: str,
        response_text: str,
        chunk_texts: list[str],
        response_mode: str = "text",
    ) -> Evaluation:
        """
        Run all heuristic graders and persist the evaluation.

        Args:
            trace_id:      The trace this evaluation belongs to.
            response_text: The LLM's response to evaluate.
            chunk_texts:   The retrieved context chunks (for grounding check).

        Returns:
            The Evaluation result (also persisted to the DB).
        """
        logger.debug(f"Evaluating trace: {trace_id}")

        # --- Run all graders ---
        g_score = grounding_score(response_text, chunk_texts)
        l_ok = length_ok(response_text)
        citations = has_citations(response_text)
        j_valid = json_valid(response_text) if response_mode == "structured" else None

        # --- Overall pass/fail decision ---
        # A response passes if it:
        #   1. Has sufficient grounding in the context
        #   2. Is within an acceptable length
        #   3. Produces valid JSON when structured mode was requested
        # Citations are NOT required to pass (informational only)
        json_check = (j_valid is True) if j_valid is not None else True
        passed = g_score >= GROUNDING_THRESHOLD and l_ok and json_check

        # Build a details dict for debugging and analysis
        grader_details = {
            "grounding_score": g_score,
            "grounding_threshold": GROUNDING_THRESHOLD,
            "word_count": len(response_text.split()),
            "length_ok": l_ok,
            "has_citations": citations,
            "json_valid": j_valid,
        }

        logger.info(
            f"Evaluation complete for trace {trace_id}: "
            f"grounding={g_score:.3f}, length_ok={l_ok}, "
            f"citations={citations}, passed={passed}"
        )

        # --- Persist to database ---
        now = datetime.now(timezone.utc).isoformat()
        try:
            result = self.db.table("evaluations").insert({
                "trace_id": trace_id,
                "grounding_score": g_score,
                "length_ok": l_ok,
                "has_citations": citations,
                "passed": passed,
                "grader_details": grader_details,
                "created_at": now,
            }).execute()

            row = result.data[0] if result.data else {}
            eval_id = row.get("id", "unknown")

        except Exception as e:
            # Eval persistence failure should NOT fail the AI response
            logger.error(f"Failed to persist evaluation for trace {trace_id}: {e}")
            eval_id = "error"

        return Evaluation(
            id=str(eval_id),
            trace_id=trace_id,
            grounding_score_val=g_score,
            length_ok_val=l_ok,
            has_citations_val=citations,
            passed=passed,
            grader_details=grader_details,
            created_at=now,
        )

    def get_evaluation(self, trace_id: str) -> dict:
        """
        Retrieve the evaluation record for a given trace.

        Args:
            trace_id: UUID of the trace.

        Returns:
            The evaluation as a dict.

        Raises:
            NotFoundError: If no evaluation exists for this trace.
        """
        result = (
            self.db.table("evaluations")
            .select("*")
            .eq("trace_id", trace_id)
            .execute()
        )

        if not result.data:
            raise NotFoundError(
                f"No evaluation found for trace: {trace_id}",
                resource_id=trace_id,
            )

        return result.data[0]
