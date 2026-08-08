# ============================================================
# app/domain/evals/graders.py — Rule-based response quality graders
#
# WHY THIS FILE EXISTS:
# After every AI response, we automatically check its quality
# using cheap graders. Most are pure Python; grounding_score is the
# one exception (see below).
#
# WHY NOT LLM-AS-JUDGE:
# An LLM grader would be more accurate but adds cost and latency
# to every single response. These graders give useful signal at a
# fraction of that cost. The judge
# (app/domain/evals/llm_judge.py) remains the real hallucination
# check; these are the always-available baseline.
#
# THE THREE GRADERS:
#
#   1. grounding_score: Is the response semantically supported by the
#      retrieved context? Embedding cosine, NOT word overlap — the
#      lexical version was measurably inverted on a bilingual corpus
#      (see the function docstring). This is the only grader that
#      costs an API call.
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

import hashlib
import json
import math
import re
from app.core.logging import get_logger

logger = get_logger(__name__)


# ------------------------------------------------------------
# Embedding cache for grounding_score
#
# WHY: grounding_score() is computed TWICE per query — once by
# evaluator.evaluate_response() and once by
# EvalService.evaluate_trace() — over the same response and the same
# chunks. Without a cache the semantic implementation would pay for
# every embedding twice. Keyed by a hash of the exact text, so a
# repeat within the same request is free and a different text always
# misses.
#
# Bounded because this is a module-level dict in a long-lived process:
# an unbounded cache would grow with every distinct answer served.
# ------------------------------------------------------------
_EMBED_CACHE: dict[str, list[float]] = {}
_EMBED_CACHE_MAX = 512


def _cache_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _embed_cached(texts: list[str]) -> list[list[float]]:
    """Embed `texts`, reusing anything already seen in this process."""
    from app.domain.documents.embeddings import embed_texts

    missing = [t for t in texts if _cache_key(t) not in _EMBED_CACHE]
    if missing:
        # De-duplicate before the API call: the same chunk can appear
        # twice in one context.
        unique = list(dict.fromkeys(missing))
        vectors = embed_texts(unique)
        for text, vector in zip(unique, vectors):
            if len(_EMBED_CACHE) >= _EMBED_CACHE_MAX:
                # Cheap eviction: drop the oldest inserted key. Python
                # dicts keep insertion order, so this is FIFO.
                _EMBED_CACHE.pop(next(iter(_EMBED_CACHE)))
            _EMBED_CACHE[_cache_key(text)] = vector
    return [_EMBED_CACHE[_cache_key(t)] for t in texts]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if not norm_a or not norm_b:
        return 0.0
    return dot / (norm_a * norm_b)


def _lexical_containment(response_text: str, chunk_texts: list[str]) -> float:
    """
    The pre-v3 lexical metric: share of the response's meaningful words
    that also appear in the context.

    Kept ONLY as the degraded path for when the embedding API is
    unavailable. It is not the metric any more, and its scale is not
    comparable to the semantic one — see grounding_score().
    """
    if not response_text.strip() or not chunk_texts:
        return 0.0

    def tokenize(text: str) -> set[str]:
        clean = re.sub(r"[^\w\s]", " ", text.lower())
        words = set(clean.split())
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "must", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "and", "or", "but", "not", "it", "this", "that", "which",
        }
        return words - stop_words

    response_words = tokenize(response_text)
    if not response_words:
        return 0.0
    context_words: set[str] = set()
    for chunk in chunk_texts:
        context_words |= tokenize(chunk)
    if not context_words:
        return 0.0
    return round(len(response_words & context_words) / len(response_words), 4)


def grounding_score(
    response_text: str,
    chunk_texts: list[str],
    chunk_embeddings: list[list[float]] | None = None,
) -> float:
    """
    Measure how well the retrieved context supports the response.

    Cosine similarity between the response embedding and each chunk
    embedding, taking the MAXIMUM:

        score = max over chunks of cosine(embed(response), embed(chunk))

    Maximum rather than mean because a grounded answer usually rests on
    one or two chunks of the five retrieved; averaging would punish an
    answer for the irrelevant chunks retrieval happened to return.

    WHY THIS REPLACED THE LEXICAL METRIC (v3, 2026-08):
    The previous implementation measured word overlap. On a bilingual
    corpus that turned out to be INVERTED, not merely weak. Measured
    over 35 samples (scripts/calibrate_grounding_metric.py):

        group                     lexical   semantic
        real answer, ES sources     0.717      0.790
        real answer, EN sources     0.242      0.638
        fabricated but plausible    0.462      0.597
        off-topic (control)         0.293      0.148

    A fabricated answer scored nearly twice as high as a CORRECT answer
    whose evidence was in English (0.462 vs 0.242), and even off-topic
    text outscored it (0.293). The cause: word overlap can only see
    evidence written in the same language as the answer, and answers are
    always Spanish by prompt while much of the corpus (OWASP, MITRE
    ATLAS, PCI DSS EN) is English. Containment against English-only
    chunks measured ~0.07 — those chunks were effectively invisible.

    Embeddings are multilingual, so the semantic score reads evidence in
    either language and orders the four groups correctly.

    CALIBRATED BANDS (semantic scale — NOT comparable to the old one):
        >= 0.70   well grounded
        0.62–0.70 grounded; 0.62 is GROUNDING_THRESHOLD
                  (app/domain/evals/service.py), the point of maximum
                  separation: keeps 80% of real answers, admits 20% of
                  fabricated ones
        0.35–0.62 weak — plausible invention lives here
        < 0.35    off-topic; measured max for off-topic text was 0.148

    LIMITATION — READ BEFORE TRUSTING IT:
    At 0.62 this still admits 20% of plausible fabrications and rejects
    20% of genuine answers. Semantic similarity measures topical
    closeness, not factual support: an invented clause about the right
    subject scores close to a real one. The LLM judge
    (app/domain/evals/llm_judge.py) is the actual hallucination check.
    This is the free, always-available fallback, and it is better than
    the lexical one at exactly that job — nothing more.

    COST:
    One embedding call for the response, plus one per chunk unless the
    caller supplies chunk_embeddings. Results are cached per process, so
    the two calls per query (evaluator + eval service) pay once.

    Args:
        response_text:    The LLM's generated response.
        chunk_texts:      Retrieved document chunk texts.
        chunk_embeddings: Optional precomputed vectors for chunk_texts,
                          in the same order. Pass them when they are
                          already at hand (document_chunks.embedding) to
                          skip re-embedding.

    Returns:
        Float in 0.0–1.0. 0.0 when there is no response or no context.

    Degraded path:
        If the embedding call fails, falls back to the pre-v3 lexical
        containment and logs a warning. The value is then on the OLD
        scale and will read low against the 0.62 threshold — that is
        deliberate: a scoring outage should not silently pass answers.
    """
    if not response_text.strip() or not chunk_texts:
        return 0.0

    try:
        response_vector = _embed_cached([response_text])[0]
        if chunk_embeddings is not None:
            if len(chunk_embeddings) != len(chunk_texts):
                raise ValueError(
                    f"chunk_embeddings has {len(chunk_embeddings)} vectors "
                    f"for {len(chunk_texts)} chunks"
                )
            vectors = chunk_embeddings
        else:
            vectors = _embed_cached(chunk_texts)
    except Exception as e:  # noqa: BLE001 — scoring must not break the pipeline
        fallback = _lexical_containment(response_text, chunk_texts)
        logger.warning(
            f"Semantic grounding unavailable ({e}); fell back to lexical "
            f"containment={fallback:.3f}. This value is on the pre-v3 scale."
        )
        return fallback

    score = max((_cosine(response_vector, v) for v in vectors), default=0.0)
    score = round(max(0.0, min(1.0, score)), 4)

    logger.debug(
        f"Grounding score (semantic): {score:.3f} over {len(vectors)} chunk(s)"
    )
    return score



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


# ============================================================
# Governance graders (Task 1)
#
# These extend the legacy heuristics with the metrics required
# by the spec. They are pure-Python, dependency-free, and do
# NOT call any LLM. The LLM judge in llm_judge.py is the
# expensive, accurate counterpart for context_adherence /
# hallucination — these graders provide fast fallbacks and
# the metrics that do not benefit from a judge.
# ============================================================

# Words shared by both Spanish and English IDK answers. Used by
# is_idk_response to detect when the model abstained correctly.
_IDK_PHRASES = (
    "no s\u00e9",
    "no se sabe",
    "no tengo informaci\u00f3n",
    "no encontr\u00e9 informaci\u00f3n",
    "no se encontraron documentos",
    "no puedo responder",
    "no puedo determinar",
    "no es posible",
    "no dispongo de",
    "no aparece en",
    "i don't know",
    "i do not know",
    "i cannot determine",
    "i cannot answer",
    "no information",
    "insufficient context",
)


def is_idk_response(response_text: str) -> bool:
    """
    Detect whether the response is an explicit "I don't know" answer.

    Used by:
      - completeness_score (an IDK that was warranted is "complete")
      - the LLM judge cross-check (we already know if it abstained)
    """
    if not response_text:
        return False
    lower = response_text.lower()
    return any(phrase in lower for phrase in _IDK_PHRASES)


def retrieval_precision(
    chunks: list,
    similarity_threshold: float = 0.45,
) -> float:
    """
    Fraction of retrieved chunks whose similarity is above the threshold.

    This is a lightweight proxy for "how relevant were the retrieved
    chunks". Inputs are expected to be either floats (similarity scores)
    or objects exposing a `.similarity` attribute (ChunkResult).

    Args:
        chunks:               Either a list of ChunkResult-like objects
                              or a list of similarity floats.
        similarity_threshold: Minimum similarity to count as "precise".

    Returns:
        A float in [0, 1]. 0 if there are no chunks.
    """
    if not chunks:
        return 0.0
    similarities = []
    for c in chunks:
        if isinstance(c, (int, float)):
            similarities.append(float(c))
        else:
            sim = getattr(c, "similarity", None)
            if sim is not None:
                similarities.append(float(sim))
    if not similarities:
        return 0.0
    above = sum(1 for s in similarities if s >= similarity_threshold)
    return round(above / len(similarities), 4)


# Numbered citation pattern e.g. [1], [12], [3, 4]
_NUMBERED_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def citation_score(
    response_text: str,
    citations: list | None,
    chunks: list,
) -> float:
    """
    Score how traceable the answer's citations are to the retrieved chunks.

    The score combines two signals, each weighted at 0.5:
      A) "presence":    are citations present at all (numbered or in
                        the citations list)?
      B) "traceability": of those citations, how many resolve to a
                        chunk that was actually retrieved?

    Args:
        response_text: The LLM's answer.
        citations:     Optional list of citation dicts/objects with a
                       `chunk_id` field (from structured mode).
        chunks:        The retrieved chunks. Each must expose either
                       chunk_id or document_id (we accept both).

    Returns:
        A float in [0, 1].
    """
    if not response_text:
        return 0.0

    chunk_ids = {
        getattr(c, "chunk_id", None) or (c.get("chunk_id") if isinstance(c, dict) else None)
        for c in (chunks or [])
    }
    chunk_ids.discard(None)

    # --- Signal A: presence ---
    inline_numbers = _NUMBERED_CITATION_RE.findall(response_text)
    has_inline_numbers = len(inline_numbers) > 0
    has_citation_list = bool(citations)
    has_attribution = bool(
        re.search(
            r"\b(seg\u00fan|de acuerdo a|conforme a|fuente:|seg\u00fan el documento|"
            r"according to|based on|per the|as stated in)",
            response_text,
            re.IGNORECASE,
        )
    )
    presence = (
        1.0
        if (has_inline_numbers or has_citation_list)
        else (0.5 if has_attribution else 0.0)
    )

    # --- Signal B: traceability ---
    # Numbered citations need to fit within the chunk count to be valid.
    chunk_count = max(len(chunks or []), 0)
    valid_inline_refs = 0
    total_inline_refs = 0
    for match in inline_numbers:
        for token in match.split(","):
            try:
                idx = int(token.strip())
            except ValueError:
                continue
            total_inline_refs += 1
            if 1 <= idx <= chunk_count:
                valid_inline_refs += 1

    valid_chunk_id_refs = 0
    total_chunk_id_refs = 0
    for cit in citations or []:
        cid = (
            getattr(cit, "chunk_id", None)
            or (cit.get("chunk_id") if isinstance(cit, dict) else None)
        )
        if cid is None:
            continue
        total_chunk_id_refs += 1
        if cid in chunk_ids:
            valid_chunk_id_refs += 1

    total_refs = total_inline_refs + total_chunk_id_refs
    if total_refs == 0:
        traceability = 0.0 if presence == 0.0 else 0.5
    else:
        traceability = (valid_inline_refs + valid_chunk_id_refs) / total_refs

    score = 0.5 * presence + 0.5 * traceability
    return round(min(max(score, 0.0), 1.0), 4)


def completeness_score(
    query: str,
    response_text: str,
    chunks: list,
) -> float:
    """
    Heuristic estimate of how completely the answer addresses the query.

    The score blends three cheap signals:
      - keyword coverage: how many of the query's content words appear
        in the response (after stop-word removal).
      - length adequacy: too-short answers usually fail to answer.
      - IDK guard: a correctly-flagged IDK answer (i.e. with no chunks
        retrieved) is considered "complete" because abstention was the
        right call.

    Args:
        query:         The user question.
        response_text: The LLM's answer.
        chunks:        The retrieved chunks (used only for the IDK
                       guard — empty chunks + IDK answer = complete).

    Returns:
        A float in [0, 1].
    """
    if not response_text or not response_text.strip():
        return 0.0

    if is_idk_response(response_text) and not chunks:
        return 1.0

    def _tokenize(text: str) -> set[str]:
        cleaned = re.sub(r"[^\w\s]", " ", text.lower())
        words = set(cleaned.split())
        stop = {
            "the", "a", "an", "is", "are", "was", "were", "to", "of",
            "in", "on", "with", "for", "and", "or", "but", "what", "which",
            "que", "cual", "cu\u00e1l", "como", "c\u00f3mo", "donde", "d\u00f3nde",
            "el", "la", "los", "las", "un", "una", "unos", "unas",
            "y", "o", "u", "de", "del", "al", "en", "por", "para", "con",
            "se", "su", "sus", "es", "son", "fue", "fueron", "ha", "han",
        }
        return words - stop

    query_words = _tokenize(query)
    response_words = _tokenize(response_text)
    if not query_words:
        keyword_coverage = 0.5
    else:
        overlap = query_words & response_words
        keyword_coverage = len(overlap) / len(query_words)

    word_count = len(response_text.split())
    if word_count >= 30:
        length_adequacy = 1.0
    elif word_count >= 10:
        length_adequacy = 0.7
    elif word_count >= 5:
        length_adequacy = 0.4
    else:
        length_adequacy = 0.1

    score = 0.6 * keyword_coverage + 0.4 * length_adequacy
    return round(min(max(score, 0.0), 1.0), 4)


def instruction_adherence_score(
    response_text: str,
    instructions: str | None,
    response_mode: str = "text",
) -> float:
    """
    Heuristic adherence to system / user instructions.

    We check the cheapest invariants we always know:
      - structured mode  → must be valid JSON
      - language         → if instructions request Spanish, response
                           should be predominantly Spanish (and vice
                           versa for English).
      - length bounds    → if instructions ask for "1-2 sentences",
                           answer should be short.

    Returns 1.0 when no instructions are provided AND response_mode
    is text (no signal to penalise on).

    Args:
        response_text: The LLM's answer.
        instructions:  The system prompt / extra instructions.
        response_mode: "text" or "structured".

    Returns:
        A float in [0, 1].
    """
    if not response_text:
        return 0.0

    score = 1.0

    if response_mode == "structured":
        if not json_valid(response_text):
            score -= 0.5

    if instructions:
        lowered_instr = instructions.lower()
        # Spanish requested
        if any(
            kw in lowered_instr
            for kw in ("responde en espa\u00f1ol", "responder en espa\u00f1ol", "in spanish")
        ):
            if _looks_like_english(response_text) and not _looks_like_spanish(response_text):
                score -= 0.3
        # English requested
        if any(
            kw in lowered_instr
            for kw in ("respond in english", "answer in english", "responde en ingl\u00e9s")
        ):
            if _looks_like_spanish(response_text) and not _looks_like_english(response_text):
                score -= 0.3
        # Brevity request
        if any(kw in lowered_instr for kw in ("una oraci\u00f3n", "1-2 sentences", "br\u00edevemente", "brevemente")):
            if len(response_text.split()) > 60:
                score -= 0.2

    return round(min(max(score, 0.0), 1.0), 4)


# Quick language heuristics for instruction adherence. Avoids pulling
# a real language detection library for one cheap signal.
_SPANISH_MARKERS = (" el ", " la ", " los ", " las ", " que ", " de ", " y ", " o ", " es ", " un ", " una ", "\u00f1", "\u00e1", "\u00e9", "\u00ed", "\u00f3", "\u00fa")
_ENGLISH_MARKERS = (" the ", " is ", " are ", " of ", " and ", " or ", " a ", " to ", " in ", " on ")


def _looks_like_spanish(text: str) -> bool:
    lowered = " " + text.lower() + " "
    return sum(1 for m in _SPANISH_MARKERS if m in lowered) >= 3


def _looks_like_english(text: str) -> bool:
    lowered = " " + text.lower() + " "
    return sum(1 for m in _ENGLISH_MARKERS if m in lowered) >= 3
