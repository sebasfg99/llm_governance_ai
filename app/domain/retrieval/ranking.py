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
