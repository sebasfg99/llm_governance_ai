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


# Hint appended to RPC errors raised while an agent filter is active.
# Migration 009_agent_tags.sql is what adds the p_agent_tags parameter to the
# four retrieval functions. If it has not been applied, PostgREST cannot find
# a function matching the argument names we send and fails the call. Telling
# the operator exactly that turns an opaque "function not found" into an
# actionable message.
AGENT_TAGS_MIGRATION_HINT = (
    "Agent-tag filtering requires migration 009_agent_tags.sql "
    "(which adds the p_agent_tags parameter to the retrieval RPCs) "
    "to be applied to the database."
)

# Same idea for the industry filter, added by migration 010. Before 010 the
# industry filter was a Python post-filter over a column the RPCs never
# returned, i.e. a silent no-op; sending p_industry_codes to a pre-010
# database fails loudly instead, which is the point.
INDUSTRY_CODES_MIGRATION_HINT = (
    "Industry filtering requires migration 010_industry_filter_fix.sql "
    "(which adds the p_industry_codes parameter to the retrieval RPCs) "
    "to be applied to the database."
)


def _normalize_array_filter(values: list[str] | None) -> list[str] | None:
    """
    Normalise a caller-supplied array filter into the value we send to SQL.

    Expected input:
        values: None, [] or a list of slugs such as ["normativa"] / ["bank"].

    Logic:
        None and [] both mean "no restriction", and the SQL side treats NULL
        and an empty array identically. Collapsing them to a single sentinel
        (None) here means every call site can use one simple truth test
        instead of repeating `if values and len(values) > 0`.

    Expected output:
        None when there is nothing to filter by, otherwise the original
        non-empty list unchanged.
    """
    return values or None


def migration_hints(
    agent_tags: list[str] | None,
    industry_codes: list[str] | None,
) -> str:
    """
    Build the migration hint(s) to append to a failed-RPC error message.

    Expected input:
        The already-normalised filters that were active on the failing call.

    Logic:
        A metadata filter can only fail at the PostgREST layer when the RPC
        lacks the matching parameter, i.e. when its migration has not been
        applied. Naming the exact migration turns an opaque "could not find
        function ... in the schema cache" into an actionable message.

    Expected output:
        "" when no optional filter was in play (the error is something else
        entirely and a migration hint would be misleading), otherwise one or
        both hint sentences joined by a space.
    """
    hints: list[str] = []
    if agent_tags:
        hints.append(AGENT_TAGS_MIGRATION_HINT)
    if industry_codes:
        hints.append(INDUSTRY_CODES_MIGRATION_HINT)
    return " ".join(hints)


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
            # industry_codes is only present from migration 010 onward. It is
            # NOT used for filtering here — the RPC already filtered in SQL —
            # but carrying it through lets RetrievalService verify after the
            # fact that the filter really applied. Omitting this mapping is
            # precisely what made the old industry post-filter a silent no-op.
            industry_codes=row.get("industry_codes"),
            # Present from migration 011 onward. Also not a filter — the
            # penalty is applied later, in rerank_chunks(), which is the
            # only stage that can see the query and therefore tell a
            # question about references from a question about content.
            is_reference_list=bool(row.get("is_reference_list") or False),
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
    agent_tags: list[str] | None = None,
    industry_codes: list[str] | None = None,
) -> list[ChunkResult]:
    """
    Tenant-scoped hybrid retrieval.

    Calls match_chunks_hybrid which returns vector_rank and
    keyword_rank for every candidate. We fuse them with RRF and
    return the top-k results.

    Args:
        db:               Supabase client used for the RPC round-trip.
        tenant_id:        Only this tenant's chunks are searched (SQL-level).
        query_embedding:  1536-dim vector of the query variant.
        query_text:       Raw text of the same variant, used by the keyword
                          (tsvector) branch inside the RPC.
        top_k:            Maximum number of fused chunks to return.
        framework_filter: Optional framework slugs. None / [] → no restriction.
        agent_tags:       Optional business-agent slugs (e.g. ["normativa"]).
                          None / [] → no restriction, and the p_agent_tags key
                          is omitted from the payload entirely so the request
                          stays byte-identical to the pre-009 one.
                          Non-empty → the RPC keeps only chunks whose
                          agent_tags array overlaps this list, in BOTH the
                          vector and the keyword branch.
        industry_codes:   Optional institution industry codes (e.g. ["bank"]).
                          Same omit-when-empty rule as agent_tags. Non-empty →
                          the RPC keeps chunks tagged 'ALL', chunks with no
                          industry tag, and chunks overlapping this list —
                          again in BOTH branches, so a sector-restricted chunk
                          cannot re-enter the pool through a keyword match.

    Returns:
        Up to top_k ChunkResult objects ordered by fused RRF score.

    Raises:
        UpstreamError: when the RPC call fails. When an optional metadata
                       filter was requested the message carries the matching
                       migration hint, because a missing parameter is by far
                       the most likely cause in that case.
    """
    normalized_agent_tags = _normalize_array_filter(agent_tags)
    normalized_industry_codes = _normalize_array_filter(industry_codes)

    rpc_args: dict = {
        "query_embedding": query_embedding,
        "query_text": query_text or "",
        "match_tenant_id": tenant_id,
        "match_count": top_k,
        "match_frameworks": framework_filter or None,
    }
    # Only send the optional filters when they carry something. Omitting the
    # keys keeps the payload identical to the pre-009/pre-010 one, so callers
    # that never use them (risk/service.py, policies/service.py) keep working
    # against a database where those migrations have not been applied.
    if normalized_agent_tags:
        rpc_args["p_agent_tags"] = normalized_agent_tags
    if normalized_industry_codes:
        rpc_args["p_industry_codes"] = normalized_industry_codes

    try:
        result = db.rpc("match_chunks_hybrid", rpc_args).execute()
    except Exception as e:
        # Fail soft when the new RPC is missing (e.g. migration 006
        # not applied yet). The caller is expected to fall back to
        # the legacy vector-only path.
        message = f"Hybrid retrieval failed: {e}"
        hints = migration_hints(normalized_agent_tags, normalized_industry_codes)
        if hints:
            message = f"{message}. {hints}"
        raise UpstreamError(
            message,
            detail={
                "tenant_id": tenant_id,
                "rpc": "match_chunks_hybrid",
                "agent_tags": normalized_agent_tags,
                "industry_codes": normalized_industry_codes,
            },
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
    agent_tags: list[str] | None = None,
    industry_codes: list[str] | None = None,
) -> list[ChunkResult]:
    """
    Global library hybrid retrieval (mirror of hybrid_search).

    Args:
        Same as hybrid_search, minus tenant_id (this always searches the
        shared 'global' tenant) and plus `country`, which the RPC applies to
        both the vector and the keyword branch.

    Returns:
        Up to top_k ChunkResult objects from the global library, or an empty
        list when the RPC fails and no optional metadata filter was requested.

    Error policy:
        Historically a failure here is NON-FATAL: the global library is a
        bonus on top of the tenant's own corpus, so we log and continue with
        whatever the tenant search produced.

        That policy is deliberately reversed when agent_tags or
        industry_codes is set. Both the agent corpus and the Layer-3 sector
        documents (SUGEF, CONASSIF) live in the global library, so swallowing
        the error would silently return zero filtered chunks and look exactly
        like "there is no such content" — the silent-failure mode these
        filters exist to eliminate. In that case we raise instead.
    """
    normalized_agent_tags = _normalize_array_filter(agent_tags)
    normalized_industry_codes = _normalize_array_filter(industry_codes)

    rpc_args: dict = {
        "query_embedding": query_embedding,
        "query_text": query_text or "",
        "match_count": top_k,
        "match_country": country,
        "match_frameworks": framework_filter or None,
    }
    # See hybrid_search: a key is only present when it carries a filter.
    if normalized_agent_tags:
        rpc_args["p_agent_tags"] = normalized_agent_tags
    if normalized_industry_codes:
        rpc_args["p_industry_codes"] = normalized_industry_codes

    try:
        result = db.rpc("match_global_chunks_hybrid", rpc_args).execute()
    except Exception as e:
        hints = migration_hints(normalized_agent_tags, normalized_industry_codes)
        if hints:
            raise UpstreamError(
                f"Hybrid global retrieval failed: {e}. {hints}",
                detail={
                    "rpc": "match_global_chunks_hybrid",
                    "agent_tags": normalized_agent_tags,
                    "industry_codes": normalized_industry_codes,
                },
            )
        logger.warning(
            f"Hybrid global retrieval failed (continuing without it): {e}"
        )
        return []

    rows = result.data or []
    fused = reciprocal_rank_fusion(rows)
    return fused[:top_k]
