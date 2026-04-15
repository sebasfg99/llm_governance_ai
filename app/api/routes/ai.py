# ============================================================
# app/api/routes/ai.py — Central AI query endpoint
#
# This is the main endpoint that consuming applications call.
# It's intentionally thin — all logic lives in AIService.
# ============================================================

from fastapi import APIRouter, Depends
from supabase import Client

from app.api.deps import get_current_user, get_db
from app.domain.ai_service import AIService
from app.schemas.ai import AIQueryRequest, AIQueryResponse

router = APIRouter()


@router.post(
    "/query",
    response_model=AIQueryResponse,
    summary="Run an AI query",
    description=(
        "The central AI endpoint. Retrieves relevant document chunks, "
        "applies guardrails, generates a response via the configured LLM, "
        "and returns the response along with trace and eval metadata."
    ),
    tags=["AI"],
)
def ai_query(
    request: AIQueryRequest,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> AIQueryResponse:
    """
    Process an AI query through the full RAG pipeline.

    Steps (all in AIService.query()):
      1. Load the system prompt for the use_case
      2. Retrieve relevant document chunks
      3. Apply guardrails (injection detection)
      4. Assemble the final prompt
      5. Call the LLM (OpenAI or Gemini)
      6. Save the trace
      7. Run heuristic evaluation
      8. Return the response

    Request body example:
        {
            "query": "What are the termination clauses in this contract?",
            "use_case": "default_rag",
            "top_k": 5,
            "include_global": true,
            "response_mode": "text"
        }
    """
    service = AIService(db)

    return service.query(
        request=request,
        tenant_id=current_user["tenant_id"],
        user_id=current_user["user_id"],
    )
