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

from supabase import Client

from app.core.config import settings
from app.core.logging import get_logger
from app.core.errors import UpstreamError

from app.domain.documents.embeddings import embed_single
from app.domain.retrieval.ranking import ChunkResult, rerank_chunks

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

        Returns:
            List of ChunkResult objects, ordered by relevance (most relevant first).
            May return fewer than top_k results if not enough chunks exist.

        Raises:
            UpstreamError: If the embedding API or DB call fails.
        """
        top_k = top_k or settings.retrieval_top_k

        logger.info(
            f"Retrieving top-{top_k} chunks for tenant={tenant_id}, "
            f"query='{query[:80]}...'"
        )

        # --- Step 1: Embed the query ---
        # Convert the user's question into a 1536-dim vector
        # We'll find chunks whose embeddings are most similar to this vector
        logger.debug("Embedding query text...")
        query_embedding = embed_single(query)

        # --- Step 2: Search tenant-specific chunks ---
        tenant_chunks = self._search_tenant_chunks(
            tenant_id=tenant_id,
            query_embedding=query_embedding,
            top_k=top_k,
        )
        logger.debug(f"Retrieved {len(tenant_chunks)} tenant-specific chunks")

        # --- Step 3: (Optional) Search global shared library ---
        global_chunks: list[ChunkResult] = []
        if include_global and tenant_id != "global":
            # Fetch fewer global chunks (they supplement, not replace, tenant chunks)
            global_top_k = max(2, top_k // 2)
            global_chunks = self._search_global_chunks(
                query_embedding=query_embedding,
                top_k=global_top_k,
            )
            logger.debug(f"Retrieved {len(global_chunks)} global library chunks")

        # --- Step 4: Merge and deduplicate ---
        # Combine both lists; tenant chunks take priority
        all_chunks = tenant_chunks + global_chunks

        # Remove duplicates by chunk_id (shouldn't happen but defensive)
        seen_ids: set[str] = set()
        unique_chunks: list[ChunkResult] = []
        for chunk in all_chunks:
            if chunk.chunk_id not in seen_ids:
                unique_chunks.append(chunk)
                seen_ids.add(chunk.chunk_id)

        # --- Step 5: Re-rank and return top_k ---
        # In MVP, re-rank just sorts by similarity (stub)
        ranked_chunks = rerank_chunks(unique_chunks, query)

        # Limit to top_k after merging global chunks
        final_chunks = ranked_chunks[:top_k]

        logger.info(
            f"Retrieval complete: returning {len(final_chunks)} chunks "
            f"(tenant: {len(tenant_chunks)}, global: {len(global_chunks)})"
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
    ) -> list[ChunkResult]:
        """
        Call the match_global_chunks Postgres RPC for the shared library.

        Args:
            query_embedding: The vectorized query.
            top_k:           Number of global chunks to return.

        Returns:
            List of ChunkResult objects from the global library.
        """
        try:
            result = self.db.rpc(
                "match_global_chunks",
                {
                    "query_embedding": query_embedding,
                    "match_count": top_k,
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
