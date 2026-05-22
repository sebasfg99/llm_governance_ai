# ============================================================
# app/schemas/ai.py — AI query API request/response schemas
#
# These models define the public HTTP contract of POST /ai/query.
# They are intentionally wider than the legacy contract because
# the endpoint now serves three capabilities:
#
#   - chatbot            → Q&A with citations (A)
#   - executive_summary  → one-off summarisation
#   - risk_intake        → conversational interview that fills a
#                          project_profile (B2)
#   - risk_analysis      → orchestrated risk register (B3, delegated
#                          to RiskService, exposed through /risk)
#   - controls_generation→ controls linked to risks (C2, delegated
#                          to PoliciesService, exposed through /policies)
#   - policy_generation  → org-level policy drafting (C1)
#
# Any of these may run with optional project_id (hydrates L2
# expediente) and thread_id (hydrates L1 session memory).
# ============================================================

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field, model_validator


# ============================================================
# Allowed use cases
#
# Centralised so the route, the service and the seed script stay
# in sync. The Literal is used on the request model; services
# read this as a string to avoid coupling to Pydantic.
# ============================================================
UseCaseLiteral = Literal[
    "chatbot",
    "executive_summary",
    "policy_generation",
    "risk_intake",
    "risk_analysis",
    "controls_generation",
]


class AIQueryRequest(BaseModel):
    """
    Request body for POST /ai/query — the central AI endpoint.

    The three new fields relative to the original MVP are:
      - project_id: binds the call to an expediente (L2 memory)
      - thread_id:  binds the call to a chat thread (L1 memory)
      - framework_filter: scopes RAG to specific normative frameworks
    """

    query: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="The user's question or instruction",
        example="¿Qué dice ISO 42001 sobre el comité de gobernanza?",
    )

    use_case: UseCaseLiteral = Field(
        default="chatbot",
        description="Use case key to select the system prompt",
        example="chatbot",
    )

    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of document chunks to retrieve",
    )

    include_global: bool = Field(
        default=True,
        description="Whether to search the global shared knowledge library",
    )

    response_mode: Literal["text", "structured"] = Field(
        default="text",
        description="Response format: 'text' for prose, 'structured' for JSON-like output",
    )

    # Country filter for RAG retrieval (normalized slug, e.g. "costa-rica", "panama")
    # If provided: returns chunks from this country + global chunks (country IS NULL)
    # If omitted: returns only global chunks (country IS NULL)
    country: str | None = Field(
        default=None,
        description="Country filter (e.g. 'costa-rica', 'panama'). If omitted, only global documents are searched.",
        example="costa-rica",
    )

    # L2 memory link — when present, AIService loads the project's
    # profile + risk register and injects a compact summary into the prompt.
    project_id: UUID | None = Field(
        default=None,
        description="Optional project UUID. When provided, the AI answer is grounded on the project expediente.",
    )

    # L1 memory link — when present, AIService loads the last N turns
    # of the thread and includes them as conversation history.
    # When omitted and use_case is conversational (chatbot / risk_intake),
    # AIService creates a new thread and returns its id in the response.
    thread_id: UUID | None = Field(
        default=None,
        description="Optional thread UUID. When provided, the call continues an existing conversation.",
    )

    # Normative framework filter. Each entry is a slug matching the
    # `framework` column on documents (see seed_documents.py mapping
    # from docs/admin/<Framework>/ folder names).
    # Examples: ["iso_42001"], ["nist_airmf", "owasp"].
    # Empty list / None means no framework restriction.
    framework_filter: list[str] | None = Field(
        default=None,
        description="Optional list of framework slugs to scope RAG retrieval to.",
        example=["iso_42001", "nist_airmf"],
    )

    # When True the response is delivered as Server-Sent Events
    # (text/event-stream). Each token arrives as `event: token`
    # with `data: {"text": "..."}`, and the stream closes with
    # `event: done` carrying trace_id, latency_ms and chunk count.
    #
    # Streaming is incompatible with response_mode="structured"
    # because partial JSON chunks are not parseable. The validator
    # below rejects that combination with a 422.
    stream: bool = Field(
        default=False,
        description=(
            "If true, the response is streamed token-by-token as "
            "Server-Sent Events (text/event-stream). Cannot be used "
            "with response_mode='structured'."
        ),
    )

    @model_validator(mode="after")
    def _stream_requires_text_mode(self) -> "AIQueryRequest":
        # Streaming JSON would arrive in non-parseable fragments and
        # provides zero UX benefit (the client must wait for the full
        # payload anyway to call json.loads). We reject the combination
        # explicitly so callers see a clear 422 instead of getting a
        # broken stream silently.
        if self.stream and self.response_mode == "structured":
            raise ValueError(
                "stream=true is not supported with response_mode='structured'. "
                "Use stream=true with response_mode='text', or use "
                "response_mode='structured' without streaming."
            )
        return self


class RetrievedChunkInfo(BaseModel):
    """Summary of a retrieved chunk included in the AI response metadata."""
    chunk_id: str
    document_id: str
    similarity: float
    content_preview: str  # First 200 chars of the chunk


class Citation(BaseModel):
    """
    A single citation pointing at a chunk that supported the answer.

    The UI uses this to render clickable anchors. We keep it narrow
    on purpose — adding fields (page number, URL, etc.) later is a
    non-breaking change, removing fields is not.
    """
    chunk_id: str
    document_name: str | None = None
    section: str | None = None
    quote: str | None = None


class StructuredAnswer(BaseModel):
    """
    Shape returned when response_mode='structured' and use_case='chatbot'.

    Returned as a JSON string in `response_text` (keeping the legacy
    field name to avoid a breaking change for existing clients). The
    UI parses this when response_mode == 'structured'.
    """
    answer: str
    citations: list[Citation] = []
    confidence: Literal["high", "medium", "low"] = "medium"
    suggested_followups: list[str] = []


class AIQueryResponse(BaseModel):
    """
    Response from POST /ai/query.

    Contains the AI's response plus metadata about how it was generated.
    The response is always exposed as `response_text` (string) — in
    structured mode it is a serialized JSON string that the UI parses.
    Richer capabilities (risk analysis, policy generation) return
    dedicated typed schemas through /risk and /policies endpoints.
    """

    # The AI's generated response. In 'text' mode this is prose;
    # in 'structured' mode it is a serialized JSON string matching
    # the schema implied by the use_case.
    response_text: str

    # The trace ID — use this to look up the full trace or submit feedback.
    # Nullable because tracing is non-blocking; if the trace write fails the
    # response is still returned.
    trace_id: str | None = None

    # The thread ID associated with this call. Always returned so the UI
    # can persist it locally and continue the same conversation.
    thread_id: str | None = None

    # The project ID if the call was bound to one (echoed back from request).
    project_id: str | None = None

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
    retrieved_chunks: list[RetrievedChunkInfo] = []

    # Signals that the AIService decided this was a "why did you X?"
    # query and hydrated the prompt with past traces from the project.
    # Useful for the UI to render a "Based on 3 past analyses" badge.
    used_reasoning_memory: bool = False
