# ============================================================
# app/domain/llm/models.py — LLM domain data models
#
# Defines the input/output shapes for the LLM routing layer.
# These are internal models used between services.
# ============================================================

from pydantic import BaseModel


class LLMRequest(BaseModel):
    """
    Input to the LLM router.

    Contains everything needed to generate a response:
    the fully assembled prompt and metadata about the request.

    NOTE: The prompt here is already assembled (system + context + query).
    Prompt assembly happens in AIService BEFORE calling the router.
    The router's job is only to dispatch + time the call.
    """

    # The fully assembled prompt to send to the LLM
    # Format: system_instructions + "\n\nContext:\n" + chunks + "\n\nUser: " + query
    prompt: str

    # The use case that triggered this request
    # Used for logging and future model-selection logic
    use_case: str = "default_rag"


class LLMResponse(BaseModel):
    """
    Output from the LLM router.

    Contains the generated text plus metadata about which
    model was used and how long it took.
    """

    # The LLM's generated response text
    text: str

    # Which provider handled this request: "openai" or "gemini"
    model_provider: str

    # The specific model name, e.g. "gpt-4o-mini" or "gemini-1.5-flash"
    model_name: str

    # End-to-end latency in milliseconds (from API call start to response received)
    latency_ms: int

    # Number of tokens in the prompt (if returned by the API)
    # OpenAI always returns this; Gemini may not
    prompt_tokens: int | None = None

    # Number of tokens in the response (if returned by the API)
    completion_tokens: int | None = None
