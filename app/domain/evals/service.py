# ============================================================
# app/domain/evals/service.py — Evaluation service
#
# Runs all heuristic graders + the LLM-as-judge after each AI
# response and persists the result to the `evaluations` table.
#
# As of migration 006_rag_governance.sql the table carries the
# eight governance metrics in addition to the legacy columns,
# so this service writes both old and new fields in a single
# insert.
# ============================================================

from datetime import datetime, timezone
from typing import Any

from supabase import Client

from app.core.logging import get_logger
from app.core.errors import NotFoundError
from app.domain.evals.graders import grounding_score, json_valid, length_ok, has_citations
from app.domain.evals.evaluator import evaluate_response

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
        *,
        query: str | None = None,
        chunks: list | None = None,
        citations: list | None = None,
        instructions: str | None = None,
        use_llm_judge: bool = True,
    ) -> Evaluation:
        """
        Run heuristic + LLM-judge graders and persist the evaluation.

        Backward compatible: callers that pass only the legacy args
        (`response_text`, `chunk_texts`, `response_mode`) still work.
        New callers should pass `query`, `chunks` (with similarity +
        chunk_id), `citations`, and `instructions` so the governance
        metrics in migration 006 are populated.

        Args:
            trace_id:      The trace this evaluation belongs to.
            response_text: The LLM's response to evaluate.
            chunk_texts:   Plain texts of retrieved chunks (legacy).
            response_mode: "text" or "structured".
            query:         The user query (needed for completeness).
            chunks:        Full ChunkResult objects (preferred). When
                           provided, retrieval_precision and
                           citation_score have richer signal. Falls
                           back to chunk_texts when omitted.
            citations:     Optional citations parsed from the response.
            instructions:  Optional system prompt fragment for adherence.
            use_llm_judge: If False, skip the LLM judge call.

        Returns:
            The Evaluation result (also persisted to the DB).
        """
        logger.debug(f"Evaluating trace: {trace_id}")

        # --- Legacy heuristics (kept for the legacy DB columns) ---
        g_score = grounding_score(response_text, chunk_texts)
        l_ok = length_ok(response_text)
        citations_present = has_citations(response_text)
        j_valid = json_valid(response_text) if response_mode == "structured" else None

        # --- Governance metrics (migration 006) ---
        eval_chunks = chunks if chunks is not None else chunk_texts
        governance: dict[str, Any] = evaluate_response(
            query=query or "",
            chunks=eval_chunks,
            response=response_text,
            citations=citations,
            instructions=instructions,
            use_llm_judge=use_llm_judge,
            response_mode=response_mode,
        )

        # --- Overall pass/fail decision ---
        # Legacy contract: pass if grounding ok + length ok + JSON ok.
        # We add governance gates: pass requires NO hallucination AND
        # context_adherence >= 0.7 (the strict-context threshold).
        json_check = (j_valid is True) if j_valid is not None else True
        passed = (
            g_score >= GROUNDING_THRESHOLD
            and l_ok
            and json_check
            and governance["context_adherence_score"] >= 0.7
            and not governance["hallucination"]
        )

        grader_details = {
            "grounding_score": g_score,
            "grounding_threshold": GROUNDING_THRESHOLD,
            "word_count": len(response_text.split()),
            "length_ok": l_ok,
            "has_citations": citations_present,
            "json_valid": j_valid,
            "governance": {
                "retrieval_precision": governance["retrieval_precision"],
                "groundedness_score": governance["groundedness_score"],
                "context_adherence_score": governance["context_adherence_score"],
                "citation_score": governance["citation_score"],
                "completeness_score": governance["completeness_score"],
                "instruction_adherence_score": governance["instruction_adherence_score"],
                "hallucination": governance["hallucination"],
                "should_have_said_idk": governance["should_have_said_idk"],
                "claims_total": governance["claims_total"],
                "claims_supported": governance["claims_supported"],
                "judge_ok": governance["judge_ok"],
                "judge_reasoning": governance.get("judge_reasoning", ""),
            },
        }

        logger.info(
            f"Evaluation complete for trace {trace_id}: "
            f"grounding={g_score:.3f}, length_ok={l_ok}, "
            f"context_adh={governance['context_adherence_score']:.3f}, "
            f"hallucination={governance['hallucination']}, "
            f"passed={passed}"
        )

        # --- Persist to database ---
        now = datetime.now(timezone.utc).isoformat()
        try:
            result = self.db.table("evaluations").insert({
                "trace_id": trace_id,
                # Legacy columns
                "grounding_score": g_score,
                "length_ok": l_ok,
                "has_citations": citations_present,
                "passed": passed,
                "grader_details": grader_details,
                "created_at": now,
                # Governance columns (migration 006)
                "retrieval_precision":         governance["retrieval_precision"],
                "groundedness_score":          governance["groundedness_score"],
                "context_adherence_score":     governance["context_adherence_score"],
                "citation_score":              governance["citation_score"],
                "completeness_score":          governance["completeness_score"],
                "instruction_adherence_score": governance["instruction_adherence_score"],
                "hallucination":               governance["hallucination"],
                "should_have_said_idk":        governance["should_have_said_idk"],
                "claims_total":                governance["claims_total"],
                "claims_supported":            governance["claims_supported"],
                "judge_raw_json":              governance.get("judge_raw"),
                "evaluator_version":           "v2.governance",
            }).execute()

            row = result.data[0] if result.data else {}
            eval_id = row.get("id", "unknown")

        except Exception as e:
            # Eval persistence failure should NOT fail the AI response
            logger.error(f"Failed to persist evaluation for trace {trace_id}: {e}")
            eval_id = "error"

        evaluation = Evaluation(
            id=str(eval_id),
            trace_id=trace_id,
            grounding_score_val=g_score,
            length_ok_val=l_ok,
            has_citations_val=citations_present,
            passed=passed,
            grader_details=grader_details,
            created_at=now,
        )
        # Attach the governance dict so callers that need to feed the
        # risk scorer don't have to re-run evaluate_response.
        evaluation.governance = governance  # type: ignore[attr-defined]
        return evaluation

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
