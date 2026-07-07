import time
from typing import Iterator

from google import genai
from google.genai import types

from app.core.config import settings
from app.core.logging import get_logger
from app.core.errors import UpstreamError
from app.domain.llm.models import StreamEvent

logger = get_logger(__name__)


class GeminiClient:
    def __init__(self) -> None:
        if not settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set. Add it to your .env file.")
        self._client = genai.Client(api_key=settings.gemini_api_key)
        logger.debug("Gemini client configured (google-genai SDK)")

    # Models that support (and default to) "thinking". Without an
    # explicit budget they spend thinking tokens billed as output —
    # unbounded thinking drained the prepaid credits in <1 month.
    _THINKING_MODEL_PREFIXES = ("gemini-2.5", "gemini-3")

    def _generation_config(
        self, temperature: float, max_tokens: int, model: str
    ) -> types.GenerateContentConfig:
        kwargs: dict = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if model.startswith(self._THINKING_MODEL_PREFIXES):
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=settings.gemini_thinking_budget,
            )
        return types.GenerateContentConfig(**kwargs)

    def _handle_error(self, e: Exception, streaming: bool = False) -> None:
        error_str = str(e).lower()
        if "api key" in error_str or "authentication" in error_str or "401" in error_str:
            raise UpstreamError("Gemini authentication failed. Check your GEMINI_API_KEY.",
                                detail={"provider": "gemini", "error": str(e)})
        elif "quota" in error_str or "rate" in error_str or "429" in error_str:
            raise UpstreamError("Gemini rate limit or quota exceeded.",
                                detail={"provider": "gemini", "error": str(e)})
        label = "streaming error" if streaming else "API error"
        raise UpstreamError(f"Gemini {label}: {e}", detail={"provider": "gemini", "error": str(e)})

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> tuple[str, int, int | None, int | None]:
        model = model or settings.llm_default_model
        logger.debug(f"Calling Gemini model={model}, temperature={temperature}")
        start_time = time.time()

        try:
            response = self._client.models.generate_content(
                model=model,
                contents=prompt,
                config=self._generation_config(temperature, max_tokens, model),
            )
        except Exception as e:
            self._handle_error(e)

        latency_ms = int((time.time() - start_time) * 1000)
        response_text = response.text or ""

        if not response_text:
            finish = None
            if response.candidates:
                finish = response.candidates[0].finish_reason
            raise UpstreamError(
                f"Gemini returned empty response. Finish reason: {finish}",
                detail={"provider": "gemini", "finish_reason": str(finish)},
            )

        prompt_tokens = None
        completion_tokens = None
        if response.usage_metadata:
            prompt_tokens = response.usage_metadata.prompt_token_count
            completion_tokens = response.usage_metadata.candidates_token_count

        logger.debug(f"Gemini response: {len(response_text)} chars, {latency_ms}ms")
        return response_text, latency_ms, prompt_tokens, completion_tokens

    def stream_generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> Iterator[StreamEvent]:
        model = model or settings.llm_default_model
        logger.debug(f"Streaming Gemini model={model}, temperature={temperature}")
        start_time = time.time()

        try:
            stream = self._client.models.generate_content_stream(
                model=model,
                contents=prompt,
                config=self._generation_config(temperature, max_tokens, model),
            )
        except Exception as e:
            self._handle_error(e, streaming=True)

        last_chunk = None
        try:
            for chunk in stream:
                last_chunk = chunk
                delta = chunk.text or ""
                if delta:
                    yield StreamEvent(type="token", text=delta)
        except Exception as e:
            raise UpstreamError(f"Gemini stream interrupted: {e}",
                                detail={"provider": "gemini", "error": str(e)})

        latency_ms = int((time.time() - start_time) * 1000)
        prompt_tokens = None
        completion_tokens = None
        if last_chunk is not None and last_chunk.usage_metadata:
            prompt_tokens = last_chunk.usage_metadata.prompt_token_count
            completion_tokens = last_chunk.usage_metadata.candidates_token_count

        logger.debug(f"Gemini stream done: {latency_ms}ms, {prompt_tokens}+{completion_tokens} tokens")
        yield StreamEvent(
            type="done",
            text="",
            latency_ms=latency_ms,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_provider="gemini",
            model_name=model,
        )
