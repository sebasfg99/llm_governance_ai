# ============================================================
# app/domain/llm/gemini_client.py — Google Gemini client
#
# WHY THIS FILE EXISTS:
# Mirror of openai_client.py for the Gemini provider.
# Same interface — the router calls generate() on either client
# without knowing which one it's talking to.
#
# GEMINI SDK NOTES:
# Google's SDK uses genai.GenerativeModel to create model instances
# and .generate_content() to call them. The interface is slightly
# different from OpenAI's but we normalize it here.
#
# TODO: Add streaming support
# TODO: Add safety settings configuration (Gemini has built-in filters)
# ============================================================

import time
import google.generativeai as genai

from app.core.config import settings
from app.core.logging import get_logger
from app.core.errors import UpstreamError

logger = get_logger(__name__)


class GeminiClient:
    """
    Wrapper around the Google Gemini Generative AI API.

    Provides the same generate() interface as OpenAIClient
    so the LLM router can call either client interchangeably.

    Usage:
        client = GeminiClient()
        text = client.generate(
            prompt="You are a helpful assistant.\n\nUser: What is RAG?",
            model="gemini-1.5-flash",
        )
    """

    def __init__(self) -> None:
        """
        Configure the Gemini SDK with the API key from settings.

        Unlike OpenAI, the Gemini SDK uses a module-level configure()
        call rather than a client object. The model is instantiated
        per-call in generate().
        """
        if not settings.gemini_api_key:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to your .env file."
            )

        # Configure the SDK globally — this stores the API key for all subsequent calls
        genai.configure(api_key=settings.gemini_api_key)
        logger.debug("Gemini client configured")

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> tuple[str, int, int | None, int | None]:
        """
        Generate a text response using the Gemini API.

        Args:
            prompt:      The fully assembled prompt.
            model:       The Gemini model to use. Defaults to settings.llm_default_model.
            temperature: Sampling temperature (0.0 = deterministic).
            max_tokens:  Max output tokens.

        Returns:
            Tuple of (response_text, latency_ms, prompt_tokens, completion_tokens).
            Note: Gemini may not always return token counts.

        Raises:
            UpstreamError: If the Gemini API call fails.
        """
        model = model or settings.llm_default_model

        logger.debug(f"Calling Gemini model={model}, temperature={temperature}")

        # Record start time for latency measurement
        start_time = time.time()

        try:
            # Instantiate the Gemini model
            # GenerativeModel is stateless — safe to create per-call
            gemini_model = genai.GenerativeModel(
                model_name=model,
                # GenerationConfig controls sampling behavior
                generation_config=genai.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )

            # Call the model with our assembled prompt
            # generate_content() is the synchronous API (async: generate_content_async)
            response = gemini_model.generate_content(prompt)

        except Exception as e:
            # Gemini SDK raises generic exceptions — inspect the message
            error_str = str(e).lower()
            if "api key" in error_str or "authentication" in error_str:
                raise UpstreamError(
                    "Gemini authentication failed. Check your GEMINI_API_KEY.",
                    detail={"provider": "gemini", "error": str(e)},
                )
            elif "quota" in error_str or "rate" in error_str:
                raise UpstreamError(
                    "Gemini rate limit or quota exceeded.",
                    detail={"provider": "gemini", "error": str(e)},
                )
            else:
                raise UpstreamError(
                    f"Gemini API error: {e}",
                    detail={"provider": "gemini", "error": str(e)},
                )

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Extract response text
        # Gemini returns response.text as a convenience property
        response_text = response.text if hasattr(response, "text") else ""

        if not response_text:
            # Check for safety filter blocks
            if hasattr(response, "candidates") and response.candidates:
                finish_reason = response.candidates[0].finish_reason
                raise UpstreamError(
                    f"Gemini returned empty response. Finish reason: {finish_reason}",
                    detail={"provider": "gemini", "finish_reason": str(finish_reason)},
                )
            raise UpstreamError(
                "Gemini returned an empty response with no explanation.",
                detail={"provider": "gemini"},
            )

        # Token counts (may not always be present in Gemini responses)
        prompt_tokens = None
        completion_tokens = None
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            prompt_tokens = getattr(response.usage_metadata, "prompt_token_count", None)
            completion_tokens = getattr(response.usage_metadata, "candidates_token_count", None)

        logger.debug(
            f"Gemini response received: {len(response_text)} chars, {latency_ms}ms"
        )

        return response_text, latency_ms, prompt_tokens, completion_tokens
