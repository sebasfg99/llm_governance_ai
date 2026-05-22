"""
Token counting utility for context budget management.

Strategy:
- OpenAI models: exact counting via tiktoken (in-process, no network calls).
- Gemini and unknown models: heuristic (chars / 4.0), ±15-20% accuracy.
  Gemini's tokenizer is not publicly available; the heuristic is sufficient
  for a budget guard — set Gemini caps conservatively to absorb the error.

Usage:
    counter = TokenCounter()
    n = counter.count("some text", model="gpt-4o-mini")
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# tiktoken encoding is cached per-process; subsequent calls are O(1) lookup.
_TIKTOKEN_AVAILABLE = False
try:
    import tiktoken  # type: ignore
    _TIKTOKEN_AVAILABLE = True
except ImportError:
    logger.warning(
        "tiktoken not installed — OpenAI token counts will use the char/4 heuristic. "
        "Install tiktoken for exact counts: pip install tiktoken"
    )

# Models whose encoding tiktoken resolves via the model name directly.
_OPENAI_MODEL_PREFIXES = ("gpt-", "o1", "o3", "text-embedding")

# Fallback encoding when the model name is unknown to tiktoken.
_FALLBACK_TIKTOKEN_ENCODING = "cl100k_base"

# Chars-per-token ratio used for Gemini and unrecognized models.
# Conservative (4.0) rather than the commonly-cited 4.5 to avoid
# underestimating Gemini's token counts.
_CHARS_PER_TOKEN_HEURISTIC = 4.0


class TokenCounter:
    """Provider-aware token estimator. Thread-safe; stateless after init."""

    def count(self, text: str, model: str) -> int:
        """
        Return an integer token estimate for *text* given *model*.

        Never raises — on any error falls back to the heuristic and logs a warning.
        Always returns at least 1 for non-empty strings.
        """
        if not text:
            return 0

        if _TIKTOKEN_AVAILABLE and self._is_openai_model(model):
            return self._count_tiktoken(text, model)

        return self._count_heuristic(text, model)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_openai_model(model: str) -> bool:
        lower = model.lower()
        return any(lower.startswith(prefix) for prefix in _OPENAI_MODEL_PREFIXES)

    @staticmethod
    def _count_tiktoken(text: str, model: str) -> int:
        try:
            enc = tiktoken.encoding_for_model(model)
        except KeyError:
            try:
                enc = tiktoken.get_encoding(_FALLBACK_TIKTOKEN_ENCODING)
                logger.debug(
                    "tiktoken: unknown model %r, using %s encoding",
                    model,
                    _FALLBACK_TIKTOKEN_ENCODING,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("tiktoken encoding load failed: %s — using heuristic", exc)
                return TokenCounter._count_heuristic(text, model)
        except Exception as exc:  # pragma: no cover
            logger.warning("tiktoken error for model %r: %s — using heuristic", model, exc)
            return TokenCounter._count_heuristic(text, model)

        try:
            return max(1, len(enc.encode(text)))
        except Exception as exc:  # pragma: no cover
            logger.warning("tiktoken encode failed: %s — using heuristic", exc)
            return TokenCounter._count_heuristic(text, model)

    @staticmethod
    def _count_heuristic(text: str, model: str) -> int:
        count = max(1, int(len(text) / _CHARS_PER_TOKEN_HEURISTIC))
        logger.debug("heuristic token count for model %r: %d tokens", model, count)
        return count
