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
        chunk_id:     UUID of the chunk in the database
        document_id:  UUID of the parent document
        tenant_id:    Which tenant this chunk belongs to
        content:      The actual text of the chunk
        chunk_index:  Position within the parent document
        similarity:   Cosine similarity score (0.0 to 1.0)
                      Higher = more relevant to the query
    """
    chunk_id: str
    document_id: str
    tenant_id: str
    content: str
    chunk_index: int
    similarity: float


def rerank_chunks(chunks: list[ChunkResult], query: str) -> list[ChunkResult]:
    """
    Re-rank retrieved chunks by relevance to the query.

    CURRENT IMPLEMENTATION: Identity function (no re-ranking).
    Returns chunks sorted by similarity score (highest first).
    This is already the order returned by pgvector, so this
    function currently does nothing.

    Future implementation will use a cross-encoder model to
    re-score each (query, chunk) pair for more accurate ranking.

    Args:
        chunks: List of ChunkResult objects from pgvector search.
        query:  The original user question (used by cross-encoders).

    Returns:
        Re-ordered list of chunks (most relevant first).

    TODO: Replace with cross-encoder re-ranking:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        pairs = [[query, chunk.content] for chunk in chunks]
        scores = model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return [chunk for chunk, score in ranked]
    """
    # MVP: just sort by similarity score (descending)
    # The query parameter is intentionally unused until re-ranking is implemented
    logger.debug(f"Reranking {len(chunks)} chunks (stub: sorting by similarity)")

    return sorted(chunks, key=lambda c: c.similarity, reverse=True)


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
