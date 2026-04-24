# ============================================================
# app/domain/retrieval/query_expansion.py — Cross-lingual query
# expansion for RAG retrieval.
#
# WHY THIS FILE EXISTS:
# The document corpus mixes English (OWASP, ISO) and Spanish
# (Conassif, DPN, PCI translations) sources. text-embedding-3-small
# is multilingual but not language-neutral: a query in Spanish
# consistently matches Spanish chunks with ~0.05 higher cosine
# similarity than equally relevant English chunks. That bias is
# enough to push the "correct" English chunk out of the top-k for
# a Spanish query (and vice-versa).
#
# THE FIX:
# For each user query we generate a *bilingual* pair (ES + EN) and
# run the vector search with both. The results are merged keeping
# the max similarity per chunk_id, which effectively levels the
# playing field between languages without needing to re-embed the
# corpus or switch embedding models.
#
# COST / LATENCY:
# One gpt-4o-mini call returning JSON with both versions of the
# query (~100 output tokens, < $0.0001 per call, ~400-800 ms of
# latency). Falls back to the original query if translation fails
# so retrieval never breaks on a translation outage.
# ============================================================

from __future__ import annotations

import json

import openai

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ------------------------------------------------------------
# OpenAI client singleton (separate from the embeddings one to
# avoid coupling between modules).
# ------------------------------------------------------------
_openai_client: openai.OpenAI | None = None


def _get_openai_client() -> openai.OpenAI:
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=settings.openai_api_key)
    return _openai_client


# ------------------------------------------------------------
# Prompt used to obtain both language versions in a single call.
# The model is instructed to keep the semantics tight — no
# rephrasing, no paraphrasing, just a faithful translation so
# the embedding similarity is meaningful.
# ------------------------------------------------------------
_TRANSLATE_SYSTEM_PROMPT = (
    "You are a translation helper for a bilingual document search engine. "
    "Given a user search query, respond with a JSON object containing two "
    "fields: 'es' (the query in Spanish) and 'en' (the query in English). "
    "Preserve domain-specific terminology and acronyms exactly (e.g. OWASP, "
    "ISO 42001, SUGEF, MITRE ATLAS). Do not paraphrase, summarise, or add "
    "context that was not in the original query. Output JSON only."
)


def generate_bilingual_queries(query: str) -> list[str]:
    """
    Return the original query plus its cross-language translation.

    The output always contains at least the original query. When
    translation succeeds the list has 2 entries (ES version + EN
    version, order-preserving); when it fails it degrades
    gracefully to ``[query]``.

    Duplicates are removed case-insensitively so that English
    queries that already happen to match their "translation" (e.g.
    "OWASP ISO 42001") don't end up searched twice.

    Args:
        query: The raw user query as received by the API.

    Returns:
        A list of 1 or 2 query strings to embed and search with.
    """
    query = (query or "").strip()
    if not query:
        return []

    if not settings.retrieval_multilingual:
        return [query]

    try:
        client = _get_openai_client()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": _TRANSLATE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        content = response.choices[0].message.content or "{}"
        payload = json.loads(content)
        es = (payload.get("es") or "").strip()
        en = (payload.get("en") or "").strip()
    except Exception as e:  # noqa: BLE001 — never break retrieval on translation
        logger.warning(
            f"Bilingual query expansion failed, falling back to original: {e}"
        )
        return [query]

    # Always keep ES first, EN second — stable ordering helps logging and
    # makes the merge deterministic across runs. If one side is empty, fall
    # back to the original so we never lose the user's intent.
    variants: list[str] = []
    for candidate in (es or query, en or query):
        if candidate and candidate.lower() not in {v.lower() for v in variants}:
            variants.append(candidate)

    if not variants:
        return [query]

    logger.info(
        f"Query expansion produced {len(variants)} variant(s): "
        f"{[v[:60] for v in variants]}"
    )
    return variants
