# ============================================================
# app/domain/evals/graders.py — Rule-based response quality graders
#
# WHY THIS FILE EXISTS:
# After every AI response, we automatically check its quality
# using simple heuristics. These checks are:
#   - FREE (no API calls)
#   - FAST (pure Python, milliseconds)
#   - TRANSPARENT (easy to understand and adjust)
#
# WHY NOT LLM-AS-JUDGE:
# An LLM grader would be more accurate but adds cost and latency
# to every single response. For the MVP, rule-based heuristics
# give us useful signal without those costs.
#
# THE THREE GRADERS:
#
#   1. grounding_score: Does the response use words from the context?
#      Low score = possible hallucination or ignoring context.
#      Uses Jaccard similarity (word overlap).
#
#   2. length_ok: Is the response a reasonable length?
#      Very short responses (< 10 words) may be unhelpful.
#      Very long responses (> 2000 words) may be off-topic.
#
#   3. has_citations: Does the response mention source references?
#      Checks for patterns like [1], [2], "Source:", "According to..."
#      Not a hard requirement in MVP — informational only.
#
# TODO: Add LLM-as-judge grader as an optional high-quality checker
# TODO: Add factual consistency check against retrieved chunks
# ============================================================

import json
import re
from app.core.logging import get_logger

logger = get_logger(__name__)


def grounding_score(response_text: str, chunk_texts: list[str]) -> float:
    """
    Measure how grounded the response is in the retrieved context.

    Uses Jaccard similarity: the proportion of words in the response
    that also appear in the retrieved chunks.

    WHY THIS WORKS AS A PROXY:
    If the AI used the retrieved context, its response will naturally
    contain many of the same words. A very low overlap suggests the
    AI may be generating from its training data (hallucinating)
    rather than from the provided documents.

    LIMITATION:
    This is a crude heuristic — common words inflate the score.
    A proper hallucination detector would check factual claims.

    Args:
        response_text: The LLM's generated response.
        chunk_texts:   List of retrieved document chunk texts.

    Returns:
        A float between 0.0 and 1.0.
        Higher = more words from response appear in the context.
        0.0 = no overlap (response shares no words with context).
        1.0 = all response words appear in context (rare).

    Typical good responses: 0.15–0.50
    Concerning scores: < 0.10 (possible hallucination)
    """
    if not response_text.strip() or not chunk_texts:
        return 0.0

    # Tokenize: lowercase, split on whitespace, remove punctuation
    def tokenize(text: str) -> set[str]:
        # Remove punctuation and lowercase for fair comparison
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        words = set(clean.split())
        # Remove very common stop words that inflate scores
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "and", "or", "but", "not", "it", "this", "that", "which",
        }
        return words - stop_words

    # Get the unique meaningful words in the response
    response_words = tokenize(response_text)
    if not response_words:
        return 0.0

    # Get the union of all words across all retrieved chunks
    context_words: set[str] = set()
    for chunk in chunk_texts:
        context_words |= tokenize(chunk)

    if not context_words:
        return 0.0

    # Jaccard similarity: |intersection| / |union|
    # This measures the overlap between response words and context words
    intersection = response_words & context_words
    union = response_words | context_words

    score = len(intersection) / len(union) if union else 0.0

    logger.debug(
        f"Grounding score: {score:.3f} "
        f"({len(intersection)}/{len(union)} words overlap)"
    )

    return round(score, 4)


def length_ok(
    response_text: str,
    min_words: int = 10,
    max_words: int = 2000,
) -> bool:
    """
    Check if the response length is within an acceptable range.

    Very short responses often indicate the AI couldn't answer.
    Very long responses may be padding or off-topic rambling.

    Args:
        response_text: The LLM's generated response.
        min_words:     Minimum acceptable word count (default 10).
        max_words:     Maximum acceptable word count (default 2000).

    Returns:
        True if the response is within the acceptable length range.
    """
    if not response_text.strip():
        return False

    word_count = len(response_text.split())
    ok = min_words <= word_count <= max_words

    logger.debug(f"Length check: {word_count} words, ok={ok}")

    return ok


def has_citations(response_text: str) -> bool:
    """
    Check if the response contains citation or source reference markers.

    In the MVP, this is informational only — it doesn't affect
    whether the response "passes" evaluation. It's useful for
    measuring how often the AI explicitly cites its sources.

    Patterns detected:
      - [1], [2], [1,2], [1][2] — numbered citations
      - "Source:", "Sources:", "Reference:" — explicit attribution
      - "According to", "Based on", "Per the document" — hedging phrases

    Args:
        response_text: The LLM's generated response.

    Returns:
        True if any citation pattern is found.
    """
    # Numbered citation patterns: [1], [1,2], [1][2]
    numbered_citation = re.search(r"\[\d+(?:,\s*\d+)*\]", response_text)

    # Explicit source labels (English and Spanish)
    source_label = re.search(
        r"\b(source|sources|reference|references|fuente|fuentes):",
        response_text,
        re.IGNORECASE,
    )

    # Spanish inline citation format: [Fuente: ...]
    fuente_citation = re.search(r"\[Fuente:", response_text, re.IGNORECASE)

    # Attribution phrases
    attribution_phrase = re.search(
        r"\b(according to|based on|per the|as stated in|as mentioned in)",
        response_text,
        re.IGNORECASE,
    )

    result = bool(numbered_citation or source_label or fuente_citation or attribution_phrase)
    logger.debug(f"Citation check: {result}")
    return result


def json_valid(response_text: str) -> bool:
    """
    Check whether the response is parseable as valid JSON.

    Used when response_mode="structured" is requested. A False result
    means the model returned malformed JSON that will fail parsing in
    downstream application code.

    Args:
        response_text: The LLM's generated response.

    Returns:
        True if the response can be parsed by json.loads().
    """
    text = response_text.strip()
    # Strip markdown code fences if the model added them despite instructions
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(
            line for line in lines
            if not line.strip().startswith("```")
        ).strip()
    try:
        json.loads(text)
        result = True
    except (json.JSONDecodeError, ValueError):
        result = False
    logger.debug(f"JSON validity check: {result}")
    return result
