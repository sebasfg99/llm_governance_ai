# ============================================================
# app/domain/llm/router.py — LLM provider router
#
# WHY THIS FILE EXISTS:
# The router is the single point of dispatch for all LLM calls.
# It reads the configured provider from settings, calls the
# correct client, and returns a normalized LLMResponse.
#
# ROUTING LOGIC:
# Config-based dispatch (LLM_DEFAULT_PROVIDER) with automatic
# single-hop fallback to the other provider on upstream failures
# (rate limit, 5xx, auth, connection). Validation errors — e.g.
# prompt injection detected by guardrails — never trigger
# fallback: they would fail identically on any provider.
#
# Streaming fallback policy: we only fail over if the primary
# stream errors BEFORE emitting its first token. Once tokens have
# reached the client we cannot transparently switch providers
# (the partial output would be inconsistent), so mid-stream
# errors propagate to the caller.
#
# FUTURE ROUTING STRATEGIES:
# - Per-request: caller specifies the provider in the API request
# - Use-case-based: "summarization" → Gemini, "Q&A" → GPT
# - Cost-based: route cheap requests to cheaper models
# - Circuit breaker to skip a provider that is known-down
# ============================================================

from typing import Iterator

from app.core.config import settings
from app.core.logging import get_logger
from app.core.errors import UpstreamError

from app.domain.llm.models import LLMRequest, LLMResponse, StreamEvent
from app.domain.llm.openai_client import OpenAIClient
from app.domain.llm.gemini_client import GeminiClient

logger = get_logger(__name__)

# Safe per-provider defaults used when llm_fallback_model is not set.
_FALLBACK_DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.5-flash",
}


def _resolve_fallback() -> tuple[str, str] | None:
    """
    Return (provider, model) for the fallback hop, or None when
    fallback is disabled or misconfigured (same provider as primary).
    """
    if not settings.llm_fallback_enabled:
        return None

    primary = settings.llm_default_provider.lower()
    provider = (settings.llm_fallback_provider or "").lower()
    if not provider:
        provider = "gemini" if primary == "openai" else "openai"
    if provider == primary or provider not in _FALLBACK_DEFAULT_MODELS:
        return None

    model = settings.llm_fallback_model or _FALLBACK_DEFAULT_MODELS[provider]
    return provider, model


class LLMRouter:
    """
    Routes LLM generation requests to the configured provider,
    failing over once to the other provider on upstream errors.

    Instantiates provider clients lazily (only when needed).
    The same router instance can be reused across requests.

    Usage:
        router = LLMRouter()
        response = router.route(LLMRequest(
            prompt="Your full assembled prompt here...",
            use_case="default_rag",
        ))
        print(f"Used: {response.model_provider}/{response.model_name}")
    """

    def __init__(self) -> None:
        # Clients are created lazily — the OpenAI or Gemini SDK is
        # only configured when first needed, so the app still boots
        # when only one provider has credentials.
        self._openai_client: OpenAIClient | None = None
        self._gemini_client: GeminiClient | None = None

        logger.debug(
            f"LLM Router initialized. "
            f"Default provider: {settings.llm_default_provider}, "
            f"Default model: {settings.llm_default_model}, "
            f"Fallback enabled: {settings.llm_fallback_enabled}"
        )

    def route(self, request: LLMRequest) -> LLMResponse:
        """
        Route an LLM request to the configured provider, retrying
        once on the fallback provider if the primary raises an
        UpstreamError.

        Raises:
            UpstreamError: If the primary fails and fallback is
                disabled/unavailable, or if both providers fail.
        """
        provider = settings.llm_default_provider.lower()
        model = settings.llm_default_model

        logger.info(
            f"Routing LLM request: provider={provider}, model={model}, "
            f"use_case={request.use_case}, "
            f"prompt_length={len(request.prompt)} chars"
        )

        try:
            return self._call(provider, model, request)
        except UpstreamError as primary_err:
            fallback = _resolve_fallback()
            if fallback is None:
                raise
            fb_provider, fb_model = fallback
            logger.warning(
                f"Primary provider '{provider}' failed ({primary_err.message}). "
                f"Failing over to {fb_provider}/{fb_model}."
            )
            try:
                return self._call(fb_provider, fb_model, request)
            except UpstreamError as fallback_err:
                raise UpstreamError(
                    f"Both LLM providers failed. "
                    f"{provider}: {primary_err.message} | "
                    f"{fb_provider}: {fallback_err.message}",
                    detail={
                        "primary_provider": provider,
                        "primary_error": primary_err.message,
                        "fallback_provider": fb_provider,
                        "fallback_error": fallback_err.message,
                    },
                )

    def _call(self, provider: str, model: str, request: LLMRequest) -> LLMResponse:
        """Dispatch one non-streaming call to a specific provider."""
        if provider == "openai":
            response_text, latency_ms, prompt_tokens, completion_tokens = (
                self._get_openai_client().generate(prompt=request.prompt, model=model)
            )
        elif provider == "gemini":
            response_text, latency_ms, prompt_tokens, completion_tokens = (
                self._get_gemini_client().generate(prompt=request.prompt, model=model)
            )
        else:
            raise UpstreamError(
                f"Unknown LLM provider: '{provider}'. "
                "Set LLM_DEFAULT_PROVIDER to 'openai' or 'gemini' in your .env file."
            )

        logger.info(
            f"LLM response received from {provider}/{model}: "
            f"{len(response_text)} chars, {latency_ms}ms"
        )

        # Normalized response — model_provider/model_name reflect the
        # provider that actually answered, so traces stay accurate
        # even after a fallback hop.
        return LLMResponse(
            text=response_text,
            model_provider=provider,
            model_name=model,
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    # ----------------------------------------------------------
    # Streaming variant
    #
    # Mirrors route() but yields StreamEvent objects. Fallback is
    # only attempted when the primary stream fails BEFORE its first
    # token (see module docstring for the policy rationale).
    # ----------------------------------------------------------
    def stream_route(self, request: LLMRequest) -> Iterator[StreamEvent]:
        provider = settings.llm_default_provider.lower()
        model = settings.llm_default_model

        logger.info(
            f"Streaming LLM request: provider={provider}, model={model}, "
            f"use_case={request.use_case}, "
            f"prompt_length={len(request.prompt)} chars"
        )

        emitted = False
        try:
            for event in self._stream_call(provider, model, request):
                emitted = True
                yield event
            return
        except UpstreamError as primary_err:
            fallback = _resolve_fallback()
            if emitted or fallback is None:
                # Tokens already reached the client (can't switch
                # providers transparently) or no fallback available.
                raise
            fb_provider, fb_model = fallback
            logger.warning(
                f"Primary provider '{provider}' failed before first token "
                f"({primary_err.message}). Failing over to {fb_provider}/{fb_model}."
            )

        yield from self._stream_call(fb_provider, fb_model, request)

    def _stream_call(
        self, provider: str, model: str, request: LLMRequest
    ) -> Iterator[StreamEvent]:
        """Dispatch one streaming call to a specific provider."""
        if provider == "openai":
            yield from self._get_openai_client().stream_generate(
                prompt=request.prompt, model=model
            )
        elif provider == "gemini":
            yield from self._get_gemini_client().stream_generate(
                prompt=request.prompt, model=model
            )
        else:
            raise UpstreamError(
                f"Unknown LLM provider: '{provider}'. "
                "Set LLM_DEFAULT_PROVIDER to 'openai' or 'gemini' in your .env file."
            )

    # ----------------------------------------------------------
    # Lazy client getters
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
