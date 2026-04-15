# ============================================================
# app/domain/traces/models.py — Trace domain models
# ============================================================

from datetime import datetime
from pydantic import BaseModel
from uuid import UUID


class Trace(BaseModel):
    """
    A complete record of one AI query-response cycle.

    Maps to the `traces` table in the database.
    Every call to POST /ai/query creates one Trace.

    The trace captures everything needed to:
    - Replay or debug any AI response
    - Build a fine-tuning dataset
    - Audit what the AI said and in what context
    - Compute usage metrics per tenant
    """

    id: UUID
    tenant_id: str
    user_id: str

    # The raw user question before prompt assembly
    user_query: str

    # The full assembled prompt sent to the LLM
    # (system prompt + context chunks + user query)
    prompt_text: str

    use_case: str
    model_provider: str     # "openai" or "gemini"
    model_name: str         # e.g. "gpt-4o-mini"

    # IDs of the document chunks retrieved for this query
    # Stored as JSONB (list of UUID strings) in the DB
    retrieved_chunk_ids: list[str] | None = None

    # The concatenated text of the retrieved chunks
    # Stored for auditing — shows exactly what context the LLM saw
    retrieved_content: str | None = None

    response_text: str
    response_mode: str = "text"   # "text" or "structured"

    latency_ms: int | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class CreateTraceRequest(BaseModel):
    """
    Input to create a new trace record.
    Used internally by TraceService.create_trace().
    """
    tenant_id: str
    user_id: str
    user_query: str
    prompt_text: str
    use_case: str
    model_provider: str
    model_name: str
    retrieved_chunk_ids: list[str]
    retrieved_content: str
    response_text: str
    response_mode: str = "text"
    latency_ms: int | None = None
