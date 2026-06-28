# ============================================================
# app/domain/retrieval/ranking.py — Chunk re-ranking
#
# WHY THIS FILE EXISTS:
# After retrieving the top-k chunks by vector similarity, we
# may want to re-order them before passing to the LLM. This
# is called "re-ranking."
#
# CURRENT STATE (MVP Stub):
# The MVP just sorts by cosine similarity (already done by
# pgvector). No actual re-ranking logic is applied.
#
# WHY RE-RANKING MATTERS:
# Vector similarity is a good first filter but not perfect.
# A cross-encoder (a second ML model) reads the query AND
# each chunk together to score relevance more accurately.
# This two-stage approach (retrieve → re-rank) significantly
# improves RAG quality.
#
# FUTURE RE-RANKERS TO CONSIDER:
#   - Cohere Rerank API (easy, hosted, ~$1/1000 calls)
#   - cross-encoder/ms-marco-MiniLM-L-6-v2 (local, free)
#   - BGE-Reranker (local, free, high quality)
#
# TODO: Implement Cohere Rerank as the first upgrade from stub
# ============================================================

from dataclasses import dataclass
from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ChunkResult:
    """
    A retrieved document chunk with its relevance score.

    This is the data structure that flows from retrieval
    into the LLM prompt assembly step.

    Attributes:
        chunk_id:       UUID of the chunk in the database
        document_id:    UUID of the parent document
        tenant_id:      Which tenant this chunk belongs to
        content:        The actual text of the chunk
        chunk_index:    Position within the parent document
        similarity:     Cosine similarity score (0.0 to 1.0)
                        Higher = more relevant to the query
        section_ref:    Most recent legal section marker the chunk belongs
                        to (e.g. "Articulo 9", "Seccion 6.1.2"). None when
                        the parent document has no formal markers or the
                        chunk falls before the first marker.
        industry_codes: Industry codes this chunk applies to (e.g. ['bank',
                        'cooperative']). None or ['ALL'] means always include.
    """
    chunk_id: str
    document_id: str
    tenant_id: str
    content: str
    chunk_index: int
    similarity: float
    section_ref: str | None = None
    industry_codes: list[str] | None = None


# ============================================================
# rerank_chunks — feature-reweighting reranker
#
# WHY THIS UPGRADE:
# The previous implementation was a sort by similarity (which is
# what pgvector already returned). The spec (Task 6) requires a
# proper rerank step. We use feature reweighting because it:
#
#   - Adds zero new dependencies (no Cohere, no Hugging Face)
#   - Costs nothing per call
#   - Combines multiple signals: vector similarity, lexical
#     overlap with the query, section markers, content length
#
# The fused score is:
#   0.55 * similarity
# + 0.30 * keyword_overlap(query, chunk)
# + 0.10 * section_priority
# + 0.05 * length_quality
#
# Tunable in one place. Cross-encoder remains a future option.
# ============================================================
import re

# Stop words removed before computing keyword overlap. The list
# covers common Spanish + English fillers so overlap reflects
# meaningful tokens.
_OVERLAP_STOP_WORDS = {
    # English
    "the", "a", "an", "is", "are", "of", "to", "in", "on", "for",
    "and", "or", "but", "with", "at", "by", "from",
    "what", "which", "how", "who", "why", "when", "where",
    "i", "you", "we", "they", "it",
    # Spanish
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "y", "o", "u", "de", "del", "al", "en", "por", "para", "con",
    "se", "su", "sus", "es", "son", "fue", "fueron", "ha", "han",
    "que", "como", "cu\u00e1l", "cual", "qu\u00e9", "qu\u00e9", "donde", "d\u00f3nde",
}

# Section markers that signal high authority (legal articles,
# regulatory clauses). Chunks that carry one are slightly boosted.
_AUTHORITY_PATTERNS = re.compile(
    r"art\u00edculo|articulo|article|secci\u00f3n|seccion|section|clausula|cl\u00e1usula|"
    r"anexo|annex|capitulo|cap\u00edtulo|chapter|requisito|requirement",
    re.IGNORECASE,
)


def _tokenize_for_overlap(text: str) -> set[str]:
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    return {w for w in cleaned.split() if w and w not in _OVERLAP_STOP_WORDS}


def _keyword_overlap(query_tokens: set[str], chunk_text: str) -> float:
    """Jaccard overlap between query tokens and chunk tokens."""
    if not query_tokens:
        return 0.0
    chunk_tokens = _tokenize_for_overlap(chunk_text)
    if not chunk_tokens:
        return 0.0
    intersection = query_tokens & chunk_tokens
    if not intersection:
        return 0.0
    return len(intersection) / len(query_tokens | chunk_tokens)


def _section_priority(section_ref: str | None, content: str) -> float:
    """1.0 when the chunk has a structured section_ref, 0.5 when it
    mentions an authority marker inline, 0.0 otherwise."""
    if section_ref:
        return 1.0
    if content and _AUTHORITY_PATTERNS.search(content):
        return 0.5
    return 0.0


def _length_quality(content: str) -> float:
    """
    Penalise extremely short or extremely long chunks. The function
    peaks around 300-700 characters which is the typical chunk size
    used by the chunker.
    """
    if not content:
        return 0.0
    length = len(content)
    if length < 80:
        return 0.2
    if length < 200:
        return 0.6
    if length <= 1500:
        return 1.0
    if length <= 3000:
        return 0.7
    return 0.4


def rerank_chunks(
    query: str,
    chunks: list[ChunkResult],
    top_k: int | None = None,
    *,
    weight_similarity: float = 0.55,
    weight_overlap: float = 0.30,
    weight_section: float = 0.10,
    weight_length: float = 0.05,
) -> list[ChunkResult]:
    """
    Re-rank chunks by combining multiple features.

    NOTE on the signature: callers in the existing codebase call
    this function as `rerank_chunks(chunks, query)`. We accept both
    orders for backward compatibility (we detect which argument is
    the list and which is the query string).

    Args:
        query:             The user question.
        chunks:            ChunkResult list to rerank.
        top_k:             Optional truncation. Returns all when None.
        weight_*:          Linear weights of each feature. Must sum
                           approximately to 1.0.

    Returns:
        Re-ranked list (highest fused score first).
    """
    # --- Backward compatibility shim ---
    # The legacy call pattern is `rerank_chunks(chunks, query)`. If
    # the first arg is a list and the second is a string, swap them.
    if isinstance(query, list) and isinstance(chunks, str):
        query, chunks = chunks, query  # type: ignore[assignment]

    if not chunks:
        return []

    query_tokens = _tokenize_for_overlap(query or "")

    enriched: list[tuple[float, ChunkResult]] = []
    for chunk in chunks:
        sim = float(chunk.similarity or 0.0)
        overlap = _keyword_overlap(query_tokens, chunk.content or "")
        section = _section_priority(chunk.section_ref, chunk.content or "")
        length = _length_quality(chunk.content or "")

        score = (
            weight_similarity * sim
            + weight_overlap * overlap
            + weight_section * section
            + weight_length * length
        )
        enriched.append((score, chunk))

    enriched.sort(key=lambda pair: pair[0], reverse=True)
    ranked = [chunk for _s, chunk in enriched]
    logger.debug(
        f"Reranked {len(ranked)} chunks "
        f"(weights sim={weight_similarity}, overlap={weight_overlap}, "
        f"section={weight_section}, length={weight_length})"
    )
    if top_k is not None:
        return ranked[:top_k]
    return ranked


def apply_per_document_cap(
    chunks: list[ChunkResult],
    max_per_document: int,
) -> list[ChunkResult]:
    """
    Enforce diversity by limiting how many chunks a single document can
    contribute to the result list.

    WHY THIS EXISTS:
    When one document dominates vector similarity (e.g. because it shares
    the query's language while other relevant docs do not), top-k ends up
    saturated with a single source and the LLM cannot cross-reference other
    documents that would otherwise be relevant.

    HOW IT WORKS:
    Walks the input list in its current (similarity-sorted) order and
    keeps only the first `max_per_document` chunks per document_id. Any
    chunks exceeding the cap are appended at the tail, preserving
    original order so they can still be selected if top-k leaves slots.

    Args:
        chunks:           Input chunks, expected to be sorted by relevance.
        max_per_document: Max chunks allowed per document_id. If <= 0,
                          the original list is returned unchanged.

    Returns:
        A new list with the capped chunks at the front followed by the
        overflow chunks (also in original order).
    """
    if max_per_document <= 0 or not chunks:
        return chunks

    per_doc_count: dict[str, int] = {}
    kept: list[ChunkResult] = []
    overflow: list[ChunkResult] = []

    for chunk in chunks:
        count = per_doc_count.get(chunk.document_id, 0)
        if count < max_per_document:
            kept.append(chunk)
            per_doc_count[chunk.document_id] = count + 1
        else:
            overflow.append(chunk)

    return kept + overflow
