# ============================================================
# app/domain/retrieval/service.py — RAG retrieval service
#
# WHY THIS FILE EXISTS:
# This is the retrieval half of RAG (Retrieval-Augmented Generation).
# When a user asks a question, this service:
#
#   1. Converts the question into a vector (embedding)
#   2. Searches the database for the most similar text chunks
#   3. Optionally adds chunks from the global shared library
#   4. Re-ranks the results (stub in MVP)
#   5. Returns the top chunks for the LLM to use as context
#
# WHY RETRIEVAL IS THE MOST IMPORTANT PART OF RAG:
# The LLM can only answer well if it receives relevant context.
# Garbage in → garbage out. If the wrong chunks are retrieved,
# the LLM will either hallucinate or say "I don't know."
# Good retrieval is the single biggest lever for RAG quality.
#
# TENANT ISOLATION:
# The pgvector search is filtered by tenant_id at the DB level.
# A tenant's chunks can NEVER appear in another tenant's search results.
# ============================================================

import re

from supabase import Client

from app.core.config import settings
from app.core.logging import get_logger
from app.core.errors import UpstreamError

from app.domain.documents.embeddings import embed_single
from app.domain.retrieval.query_expansion import generate_bilingual_queries
from app.domain.retrieval.ranking import (
    ChunkResult,
    apply_per_document_cap,
    rerank_chunks,
)

_INSTRUCTION_PREFIXES = re.compile(
    r"^(resumeme|resume|explícame|explicame|explica|qué es|que es|"
    r"cuéntame sobre|cuéntame acerca de|hablame de|háblame de|"
    r"describe|analiza|analízame|dime sobre|habla de|habla acerca de|"
    r"puedes explicar|puedes resumir|puedes describir)\s+(el|la|los|las|un|una|sobre|acerca de)?\s*",
    re.IGNORECASE,
)


def _extract_search_topic(query: str) -> str:
    cleaned = _INSTRUCTION_PREFIXES.sub("", query.strip())
    return cleaned if cleaned else query

# Logger for this module
logger = get_logger(__name__)


class RetrievalService:
    """
    Performs vector similarity search to find relevant document chunks.

    Uses Supabase RPC to call the match_chunks PostgreSQL function,
    which runs the pgvector cosine similarity search.

    Usage:
        db = get_supabase_client()
        service = RetrievalService(db)
        chunks = service.retrieve(
            tenant_id="acme",
            query="What are the termination clauses?",
            top_k=5,
        )
    """

    def __init__(self, db: Client) -> None:
        """
        Initialize with a Supabase client.

        Args:
            db: An initialized Supabase Client instance.
        """
        self.db = db

    def retrieve(
        self,
        tenant_id: str,
        query: str,
        top_k: int | None = None,
        include_global: bool = True,
        country: str | None = None,
    ) -> list[ChunkResult]:
        """
        Find the most relevant document chunks for a user query.

        Steps:
          1. Embed the query text into a vector
          2. Call match_chunks RPC for tenant-specific chunks
          3. Optionally call match_global_chunks for shared library
          4. Merge, deduplicate, and re-rank results
          5. Return top_k results

        Args:
            tenant_id:      Only search this tenant's documents.
            query:          The user's question or search text.
            top_k:          Max number of chunks to return.
                            Defaults to settings.retrieval_top_k (5).
            include_global: If True, also search the global shared library
                            and merge results. Default: True.
            country:        Country filter (e.g. "costa-rica"). If provided,
                            returns chunks from that country + global chunks
                            (country IS NULL). If None, returns only global chunks.

        Returns:
            List of ChunkResult objects, ordered by relevance (most relevant first).
            May return fewer than top_k results if not enough chunks exist.

        Raises:
            UpstreamError: If the embedding API or DB call fails.
        """
        top_k = top_k or settings.retrieval_top_k
        # Fetch more candidates than top_k so the threshold filter has room to
        # discard low-quality chunks without starving the final result set.
        candidate_k = max(settings.retrieval_candidate_k, top_k)

        logger.info(
            f"Retrieving top-{top_k} chunks (candidate_k={candidate_k}) for "
            f"tenant={tenant_id}, country={country}, query='{query[:80]}...'"
        )

        # --- Step 1: Prepare query variants ---
        # Strip instruction verbs ("Resumeme el X" → "X") so the embedding
        # captures the topic, not the instruction pattern.
        search_topic = _extract_search_topic(query)
        if search_topic != query:
            logger.debug(f"Query topic extracted: '{query[:60]}' → '{search_topic}'")

        # Cross-lingual expansion: turn the topic into a bilingual pair so the
        # search is not biased toward the corpus' majority language.
        query_variants = generate_bilingual_queries(search_topic)
        if not query_variants:
            query_variants = [search_topic]

        logger.debug(
            f"Searching with {len(query_variants)} query variant(s): "
            f"{[v[:60] for v in query_variants]}"
        )

        # --- Step 2+3: Embed each variant and run both searches ---
        # Chunks are accumulated in a dict keyed by chunk_id so duplicates
        # across variants collapse, keeping the max similarity seen.
        candidate_pool: dict[str, ChunkResult] = {}

        for variant in query_variants:
            variant_embedding = embed_single(variant)

            tenant_chunks = self._search_tenant_chunks(
                tenant_id=tenant_id,
                query_embedding=variant_embedding,
                top_k=candidate_k,
            )

            global_chunks: list[ChunkResult] = []
            if include_global and tenant_id != "global":
                global_chunks = self._search_global_chunks(
                    query_embedding=variant_embedding,
                    top_k=candidate_k,
                    country=country,
                )

            logger.debug(
                f"Variant '{variant[:40]}…' → tenant={len(tenant_chunks)} "
                f"global={len(global_chunks)}"
            )

            for chunk in tenant_chunks + global_chunks:
                existing = candidate_pool.get(chunk.chunk_id)
                if existing is None or chunk.similarity > existing.similarity:
                    candidate_pool[chunk.chunk_id] = chunk

        unique_chunks = list(candidate_pool.values())
        logger.debug(
            f"Merged candidate pool: {len(unique_chunks)} unique chunks across "
            f"{len(query_variants)} variant(s)"
        )

        # --- Step 5: Re-rank ---
        ranked_chunks = rerank_chunks(unique_chunks, query)

        # --- Step 5b: Per-document diversity cap ---
        # Keep all chunks but reorder so each document contributes at most
        # max_per_document chunks before any document contributes a second
        # batch. Prevents a single high-similarity document (e.g. one that
        # shares the query language) from monopolising every top-k slot.
        max_per_doc = settings.retrieval_max_chunks_per_document
        capped_chunks = apply_per_document_cap(ranked_chunks, max_per_doc)

        # --- Step 6: Filter by similarity threshold ---
        threshold = settings.similarity_threshold
        above_threshold = [c for c in capped_chunks if c.similarity >= threshold]
        discarded = len(capped_chunks) - len(above_threshold)
        if discarded:
            logger.info(
                f"Threshold filter ({threshold}): discarded {discarded} low-similarity chunks, "
                f"{len(above_threshold)} remaining"
            )

        if not above_threshold:
            logger.info(
                f"No chunks above similarity threshold {threshold}. Returning empty list."
            )
            return []

        final_chunks = above_threshold[:top_k]

        logger.info(
            f"Retrieval complete: returning {len(final_chunks)} chunks "
            f"(variants: {len(query_variants)}, unique_candidates: {len(unique_chunks)}, "
            f"threshold_discarded: {discarded})"
        )

        return final_chunks

    def _search_tenant_chunks(
        self,
        tenant_id: str,
        query_embedding: list[float],
        top_k: int,
    ) -> list[ChunkResult]:
        """
        Call the match_chunks Postgres RPC function for a specific tenant.

        Args:
            tenant_id:       Filter chunks to this tenant only.
            query_embedding: The vectorized query.
            top_k:           Number of results to return.

        Returns:
            List of ChunkResult objects.
        """
        try:
            # Call the match_chunks function defined in 001_initial.sql
            # db.rpc() calls a Postgres stored function by name
            result = self.db.rpc(
                "match_chunks",
                {
                    "query_embedding": query_embedding,  # The query vector
                    "match_tenant_id": tenant_id,        # Tenant filter
                    "match_count": top_k,                # How many to return
                }
            ).execute()

        except Exception as e:
            raise UpstreamError(
                f"Database vector search failed: {e}",
                detail={"tenant_id": tenant_id},
            )

        # Convert the raw DB rows into ChunkResult objects
        return [
            ChunkResult(
                chunk_id=str(row["id"]),
                document_id=str(row["document_id"]),
                tenant_id=row["tenant_id"],
                content=row["content"],
                chunk_index=row["chunk_index"],
                similarity=float(row["similarity"]),
            )
            for row in (result.data or [])
        ]

    def _search_global_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
        country: str | None = None,
    ) -> list[ChunkResult]:
        """
        Call the match_global_chunks Postgres RPC for the shared library.

        Args:
            query_embedding: The vectorized query.
            top_k:           Number of global chunks to return.
            country:         Country filter. If provided, returns chunks from
                             that country + global chunks (country IS NULL).
                             If None, returns only global chunks (country IS NULL).

        Returns:
            List of ChunkResult objects from the global library.
        """
        try:
            result = self.db.rpc(
                "match_global_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_count": top_k,
                    "match_country": country,
                }
            ).execute()

        except Exception as e:
            # Global library failure is non-fatal — log and return empty list
            logger.warning(f"Global library search failed (continuing without it): {e}")
            return []

        return [
            ChunkResult(
                chunk_id=str(row["id"]),
                document_id=str(row["document_id"]),
                tenant_id=row["tenant_id"],
                content=row["content"],
                chunk_index=row["chunk_index"],
                similarity=float(row["similarity"]),
            )
            for row in (result.data or [])
        ]
