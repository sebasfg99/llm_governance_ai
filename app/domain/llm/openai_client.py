# ============================================================
# app/domain/llm/openai_client.py — OpenAI chat completions client
#
# WHY THIS FILE EXISTS:
# Wraps the OpenAI SDK into a clean interface that the router
# can call without knowing any OpenAI-specific details.
#
# If OpenAI changes their SDK or we want to swap providers,
# we only change THIS file — not the router or the service.
#
# WHAT THIS CLIENT DOES:
#   - Initializes the OpenAI SDK with the API key
#   - Formats the prompt as a chat message (system + user)
#   - Calls the chat completions endpoint
#   - Returns the response text
#
# CHAT FORMAT:
# OpenAI uses a "messages" format: each message has a role
# (system, user, or assistant) and content (text).
# We send:
#   - role="system": The assembled instructions + context
#   - role="user":   The user's question (already embedded in prompt)
#
# TODO: Add retry logic with exponential backoff for rate limits
# TODO: Add streaming support for long responses
# ============================================================

import time
import openai as openai_sdk

from app.core.config import settings
from app.core.logging import get_logger
from app.core.errors import UpstreamError

logger = get_logger(__name__)


class OpenAIClient:
    """
    Wrapper around the OpenAI Chat Completions API.

    Provides a simple generate() method that the LLM router calls.
    Handles SDK initialization, error mapping, and token tracking.

    Usage:
        client = OpenAIClient()
        text = client.generate(
            prompt="You are a helpful assistant.\n\nUser: What is RAG?",
            model="gpt-4o-mini",
        )
    """

    def __init__(self) -> None:
        """
        Initialize the OpenAI client with the API key from settings.

        The client object is reused across all generate() calls —
        it manages connection pooling internally.
        """
        if not settings.openai_api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Add it to your .env file."
            )

        # Create the OpenAI client — this does NOT make an API call yet
        self._client = openai_sdk.OpenAI(api_key=settings.openai_api_key)
        logger.debug("OpenAI client initialized")

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 2048,
    ) -> tuple[str, int, int | None, int | None]:
        """
        Generate a text response from OpenAI's Chat Completions API.

        Args:
            prompt:      The fully assembled prompt (already includes
                         system instructions, context, and user query).
            model:       The OpenAI model to use. Defaults to settings.llm_default_model.
            temperature: Controls randomness (0.0 = deterministic, 1.0 = creative).
                         We default to 0.2 for factual RAG responses.
            max_tokens:  Maximum length of the response. 2048 is generous for most answers.

        Returns:
            Tuple of (response_text, latency_ms, prompt_tokens, completion_tokens).

        Raises:
            UpstreamError: If the OpenAI API call fails.
        """
        model = model or settings.llm_default_model

        logger.debug(f"Calling OpenAI model={model}, temperature={temperature}")

        # Record start time to measure latency
        start_time = time.time()

        try:
            # Call the Chat Completions API
            # We split the prompt into system + user message format
            # The entire assembled prompt goes in the system role for simplicity
            # TODO: Separate system instructions from user query for better token efficiency
            response = self._client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        # The assembled prompt already contains everything:
                        # system instructions + retrieved context + user question
                        "content": prompt,
                    }
                ],
                temperature=temperature,    # Low temperature = more factual, less creative
                max_tokens=max_tokens,      # Cap response length
            )

        except openai_sdk.AuthenticationError as e:
            raise UpstreamError(
                "OpenAI authentication failed. Check your OPENAI_API_KEY.",
                detail={"provider": "openai", "error": str(e)},
            )
        except openai_sdk.RateLimitError as e:
            raise UpstreamError(
                "OpenAI rate limit exceeded. Try again in a moment.",
                detail={"provider": "openai", "error": str(e)},
            )
        except openai_sdk.BadRequestError as e:
            raise UpstreamError(
                f"OpenAI rejected the request (prompt may be too long or invalid): {e}",
                detail={"provider": "openai", "error": str(e)},
            )
        except openai_sdk.APIError as e:
            raise UpstreamError(
                f"OpenAI API error: {e}",
                detail={"provider": "openai", "error": str(e)},
            )
        except Exception as e:
            raise UpstreamError(
                f"Unexpected error calling OpenAI: {e}",
                detail={"provider": "openai"},
            )

        # Calculate latency
        latency_ms = int((time.time() - start_time) * 1000)

        # Extract the response text
        # response.choices[0].message.content is the generated text
        response_text = response.choices[0].message.content or ""

        # Extract token usage (always present in OpenAI responses)
        prompt_tokens = response.usage.prompt_tokens if response.usage else None
        completion_tokens = response.usage.completion_tokens if response.usage else None

        logger.debug(
            f"OpenAI response received: {len(response_text)} chars, "
            f"{latency_ms}ms, "
            f"tokens: {prompt_tokens} prompt + {completion_tokens} completion"
        )

        return response_text, latency_ms, prompt_tokens, completion_tokens
