# ============================================================
# app/schemas/ai.py — AI query API request/response schemas
# ============================================================

from typing import Literal

from pydantic import BaseModel, Field


class AIQueryRequest(BaseModel):
    """
    Request body for POST /ai/query — the central AI endpoint.

    This is the main API surface that consuming applications
    (chatbots, web apps) will call.
    """

    # The user's question or instruction
    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's question or instruction",
        example="What are the main risks in this contract?",
    )

    # Which system prompt / behavior profile to use
    # Maps to the `use_case` column in the prompts table
    use_case: Literal["chatbot", "executive_summary", "policy_generation"] = Field(
        default="chatbot",
        description="Use case key to select the system prompt",
        example="executive_summary",
    )

    # How many document chunks to retrieve for context
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve",
    )

    # Whether to include the global shared knowledge library
    include_global: bool = Field(
        default=True,
        description="Whether to search the global shared knowledge library",
    )

    # Output format: "text" or "structured"
    response_mode: str = Field(
        default="text",
        description="Response format: 'text' for prose, 'structured' for JSON-like output",
    )


class RetrievedChunkInfo(BaseModel):
    """Summary of a retrieved chunk included in the AI response metadata."""
    chunk_id: str
    document_id: str
    similarity: float
    content_preview: str  # First 200 chars of the chunk


class AIQueryResponse(BaseModel):
    """
    Response from POST /ai/query.

    Contains the AI's response plus metadata about how it was generated.
    All metadata is useful for debugging, evaluation, and display.
    """

    # The AI's generated response text
    response_text: str

    # The trace ID — use this to look up the full trace or submit feedback
    trace_id: str

    # Which LLM provider and model generated the response
    model_provider: str
    model_name: str

    # How long the LLM call took (milliseconds)
    latency_ms: int

    # The use case that was active for this query
    use_case: str

    # Number of chunks that were retrieved and used as context
    retrieved_chunk_count: int

    # Summary of the retrieved chunks (for display in the UI)
    # TODO: Add a "requested_citations" flag to the request to control this
    retrieved_chunks: list[RetrievedChunkInfo] = []
