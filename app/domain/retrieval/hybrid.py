# ============================================================
# app/domain/retrieval/hybrid.py — Hybrid retrieval (vector + keyword)
#
# WHY THIS FILE EXISTS:
# Pure vector search misses chunks that share an exact term
# (acronyms, regulatory codes like "ISO 42001", proper nouns)
# but happen to live in a semantically distant document.
# Pure keyword search misses paraphrases. The accepted
# enterprise solution is hybrid:
#
#   1. Run BOTH searches in the database (one RPC).
#   2. Merge the rankings via Reciprocal Rank Fusion (RRF):
#         rrf(d) = sum( 1 / (k + rank_i(d)) )
#      where k = 60 (the standard RRF constant).
#   3. Order by the fused score.
#
# WHY NATIVE POSTGRES:
# We use Postgres' built-in pgvector + tsvector — zero new
# dependencies, zero external API calls, zero cost. Migration
# 006 added match_chunks_hybrid + match_global_chunks_hybrid
# RPCs that return both ranks in a single round-trip.
# ============================================================

from __future__ import annotations

from supabase import Client

from app.core.logging import get_logger
from app.core.errors import UpstreamError
from app.domain.retrieval.ranking import ChunkResult

logger = get_logger(__name__)


# Reciprocal Rank Fusion constant. 60 is the value used in the
# original Cormack et al. paper and most search systems. Lowering
# it makes top ranks more dominant; raising it flattens the
# distribution. Keep it tunable but not configurable yet.
RRF_K = 60


def reciprocal_rank_fusion(
    rows: list[dict],
    k: int = RRF_K,
) -> list[ChunkResult]:
    """
    Combine vector and keyword ranks via RRF and return ChunkResult.

    Args:
        rows: Output of `match_chunks_hybrid` / `match_global_chunks_hybrid`.
              Every row carries `vector_rank`, `keyword_rank`, plus the
              chunk fields. Either rank may be NULL.
        k:    RRF constant. Default 60.

    Returns:
        ChunkResult list, ordered by fused score (highest first).
    """
    enriched: list[tuple[float, ChunkResult]] = []
    for row in rows or []:
        vector_rank = row.get("vector_rank")
        keyword_rank = row.get("keyword_rank")

        score = 0.0
        if vector_rank is not None:
            score += 1.0 / (k + int(vector_rank))
        if keyword_rank is not None:
            score += 1.0 / (k + int(keyword_rank))

        chunk = ChunkResult(
            chunk_id=str(row["id"]),
            document_id=str(row["document_id"]),
            tenant_id=row["tenant_id"],
            content=row["content"],
            chunk_index=row["chunk_index"],
            similarity=float(row.get("similarity") or 0.0),
            section_ref=row.get("section_ref"),
        )
        enriched.append((score, chunk))

    enriched.sort(key=lambda pair: pair[0], reverse=True)
    return [chunk for _score, chunk in enriched]


def hybrid_search(
    db: Client,
    *,
    tenant_id: str,
    query_embedding: list[float],
    query_text: str,
    top_k: int,
    framework_filter: list[str] | None = None,
) -> list[ChunkResult]:
    """
    Tenant-scoped hybrid retrieval.

    Calls match_chunks_hybrid which returns vector_rank and
    keyword_rank for every candidate. We fuse them with RRF and
    return the top-k results.
    """
    try:
        result = db.rpc(
            "match_chunks_hybrid",
            {
                "query_embedding": query_embedding,
                "query_text": query_text or "",
                "match_tenant_id": tenant_id,
                "match_count": top_k,
                "match_frameworks": framework_filter or None,
            },
        ).execute()
    except Exception as e:
        # Fail soft when the new RPC is missing (e.g. migration 006
        # not applied yet). The caller is expected to fall back to
        # the legacy vector-only path.
        raise UpstreamError(
            f"Hybrid retrieval failed: {e}",
            detail={"tenant_id": tenant_id, "rpc": "match_chunks_hybrid"},
        )

    rows = result.data or []
    fused = reciprocal_rank_fusion(rows)
    logger.debug(
        f"Hybrid retrieval: {len(fused)} fused chunks "
        f"(vector_hits + keyword_hits in one RPC)"
    )
    return fused[:top_k]


def hybrid_search_global(
    db: Client,
    *,
    query_embedding: list[float],
    query_text: str,
    top_k: int,
    country: str | None = None,
    framework_filter: list[str] | None = None,
) -> list[ChunkResult]:
    """Global library hybrid retrieval (mirror of hybrid_search)."""
    try:
        result = db.rpc(
            "match_global_chunks_hybrid",
            {
                "query_embedding": query_embedding,
                "query_text": query_text or "",
                "match_count": top_k,
                "match_country": country,
                "match_frameworks": framework_filter or None,
            },
        ).execute()
    except Exception as e:
        logger.warning(
            f"Hybrid global retrieval failed (continuing without it): {e}"
        )
        return []

    rows = result.data or []
    fused = reciprocal_rank_fusion(rows)
    return fused[:top_k]
