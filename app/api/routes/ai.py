# ============================================================
# app/api/routes/ai.py — Central AI query endpoint
#
# This is the main endpoint that consuming applications call.
# It's intentionally thin — all logic lives in AIService.
#
# Two response modes are supported:
#   - JSON (default): single response body with the AI text plus
#     metadata (trace_id, latency, retrieved chunks, ...).
#   - SSE streaming (request.stream=true): tokens are pushed
#     as they are generated; the stream closes with a final
#     `done` event carrying trace_id and metadata.
# ============================================================

from typing import Union

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from supabase import Client

from app.api.deps import get_current_user, get_db
from app.domain.ai_service import AIService
from app.schemas.ai import AIQueryRequest, AIQueryResponse

router = APIRouter()


@router.post(
    "/query",
    response_model=None,  # Response shape varies (JSON vs SSE)
    summary="Run an AI query",
    description=(
        "The central AI endpoint. Retrieves relevant document chunks, "
        "applies guardrails, generates a response via the configured LLM, "
        "and returns the response along with trace and eval metadata. "
        "Set stream=true to receive the response as Server-Sent Events "
        "(only valid with response_mode='text')."
    ),
    tags=["AI"],
)
def ai_query(
    request: AIQueryRequest,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> Union[AIQueryResponse, StreamingResponse]:
    """
    Process an AI query through the full RAG pipeline.

    Steps (all in AIService.query() / AIService.query_stream()):
      1. Load the system prompt for the use_case
      2. Resolve / create the conversation thread (L1)
      3. Load the project expediente if any (L2)
      4. Retrieve relevant document chunks
      5. Detect "why?" queries and load reasoning context (L3)
      6. Apply guardrails (injection detection)
      7. Assemble the final prompt
      8. Call the LLM (blocking or streaming)
      9. Persist the trace + thread messages
     10. Run heuristic evaluation
     11. Return the response (JSON or SSE)

    Request body example:
        {
            "query": "What are the termination clauses?",
            "use_case": "chatbot",
            "top_k": 5,
            "include_global": true,
            "response_mode": "text",
            "stream": false
        }
    """
    service = AIService(db)

    if request.stream:
        # StreamingResponse lazily consumes the generator; tokens are
        # flushed to the client as they arrive. media_type=text/event-stream
        # is what tells the browser / client to treat it as SSE.
        # Cache-Control + X-Accel-Buffering disable proxy buffering so
        # the stream is not held back by Nginx / Cloud Run.
        return StreamingResponse(
            service.query_stream(
                request=request,
                tenant_id=current_user["tenant_id"],
                user_id=current_user["user_id"],
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )

    return service.query(
        request=request,
        tenant_id=current_user["tenant_id"],
        user_id=current_user["user_id"],
    )
