# ============================================================
# app/domain/llm/router.py — LLM provider router
#
# WHY THIS FILE EXISTS:
# The router is the single point of dispatch for all LLM calls.
# It reads the configured provider from settings, calls the
# correct client, and returns a normalized LLMResponse.
#
# CURRENT ROUTING LOGIC (MVP):
# Config-based only — reads LLM_DEFAULT_PROVIDER from .env.
# No per-request overrides in the MVP.
#
# FUTURE ROUTING STRATEGIES:
# - Per-request: caller specifies the provider in the API request
# - Use-case-based: "summarization" → Gemini, "Q&A" → GPT
# - Fallback: if OpenAI fails, retry with Gemini
# - Cost-based: route cheap requests to cheaper models
#
# TODO: Add retry logic with fallback provider
# TODO: Add circuit breaker pattern for provider outages
# ============================================================

from app.core.config import settings
from app.core.logging import get_logger
from app.core.errors import UpstreamError

from app.domain.llm.models import LLMRequest, LLMResponse
from app.domain.llm.openai_client import OpenAIClient
from app.domain.llm.gemini_client import GeminiClient

logger = get_logger(__name__)


class LLMRouter:
    """
    Routes LLM generation requests to the configured provider.

    Instantiates provider clients lazily (only when needed).
    The same router instance can be reused across requests.

    Usage:
        router = LLMRouter()
        response = router.route(LLMRequest(
            prompt="Your full assembled prompt here...",
            use_case="default_rag",
        ))
        print(response.text)
        print(f"Used: {response.model_provider}/{response.model_name}")
    """

    def __init__(self) -> None:
        """
        Initialize the router.

        Clients are created lazily — the OpenAI or Gemini SDK
        is only imported/configured when first needed.
        """
        # These are initialized to None and created on first use
        # This avoids startup errors if only one provider is configured
        self._openai_client: OpenAIClient | None = None
        self._gemini_client: GeminiClient | None = None

        logger.debug(
            f"LLM Router initialized. "
            f"Default provider: {settings.llm_default_provider}, "
            f"Default model: {settings.llm_default_model}"
        )

    def route(self, request: LLMRequest) -> LLMResponse:
        """
        Route an LLM request to the configured provider.

        Reads LLM_DEFAULT_PROVIDER from settings to decide which
        client to use. Calls the client's generate() method and
        wraps the result in a normalized LLMResponse.

        Args:
            request: An LLMRequest containing the assembled prompt.

        Returns:
            An LLMResponse with the generated text and metadata.

        Raises:
            UpstreamError: If the provider call fails.
            ValueError: If LLM_DEFAULT_PROVIDER is not "openai" or "gemini".
        """
        provider = settings.llm_default_provider.lower()
        model = settings.llm_default_model

        logger.info(
            f"Routing LLM request: provider={provider}, model={model}, "
            f"use_case={request.use_case}, "
            f"prompt_length={len(request.prompt)} chars"
        )

        # Dispatch to the correct provider client
        if provider == "openai":
            response_text, latency_ms, prompt_tokens, completion_tokens = (
                self._get_openai_client().generate(
                    prompt=request.prompt,
                    model=model,
                )
            )

        elif provider == "gemini":
            response_text, latency_ms, prompt_tokens, completion_tokens = (
                self._get_gemini_client().generate(
                    prompt=request.prompt,
                    model=model,
                )
            )

        else:
            raise UpstreamError(
                f"Unknown LLM provider: '{provider}'. "
                "Set LLM_DEFAULT_PROVIDER to 'openai' or 'gemini' in your .env file."
            )

        logger.info(
            f"LLM response received: {len(response_text)} chars, "
            f"{latency_ms}ms"
        )

        # Return a normalized response regardless of which provider was used
        return LLMResponse(
            text=response_text,
            model_provider=provider,
            model_name=model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # ----------------------------------------------------------
    # Lazy client getters
    # Only create the client when it's actually needed.
    # ----------------------------------------------------------

    def _get_openai_client(self) -> OpenAIClient:
        """Return the OpenAI client, creating it if necessary."""
        if self._openai_client is None:
            logger.debug("Creating OpenAI client...")
            self._openai_client = OpenAIClient()
        return self._openai_client

    def _get_gemini_client(self) -> GeminiClient:
        """Return the Gemini client, creating it if necessary."""
        if self._gemini_client is None:
            logger.debug("Creating Gemini client...")
            self._gemini_client = GeminiClient()
        return self._gemini_client
