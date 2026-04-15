#!/usr/bin/env python3
# ============================================================
# scripts/backfill_embeddings.py — Re-generate missing embeddings
#
# WHY THIS SCRIPT EXISTS:
# If you change your embedding model (e.g., from text-embedding-3-small
# to text-embedding-3-large), existing chunks have the wrong-dimension
# embeddings. This script re-generates embeddings for all chunks
# where embedding IS NULL or for all chunks in a specific document.
#
# Also useful if initial embedding generation failed for some chunks.
#
# HOW TO RUN (from project root):
#   PYTHONPATH=. python scripts/backfill_embeddings.py
#   PYTHONPATH=. python scripts/backfill_embeddings.py --document-id <uuid>
#
# WARNING: Running this for all chunks will make many OpenAI API calls.
# Use --document-id to limit scope during testing.
# ============================================================

import sys
import os
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.supabase import get_supabase_client
from app.domain.documents.embeddings import embed_texts
from app.core.logging import get_logger

logger = get_logger(__name__)

# How many chunks to embed in one batch (OpenAI API batch limit is 2048)
BATCH_SIZE = 50


def backfill_embeddings(document_id: str | None = None) -> None:
    """
    Re-generate embeddings for chunks with missing or NULL embeddings.

    Args:
        document_id: If provided, only process chunks for this document.
                     If None, process ALL chunks with NULL embeddings.
    """
    db = get_supabase_client()

    logger.info(
        f"Starting embedding backfill... "
        f"{'(document: ' + document_id + ')' if document_id else '(all NULL embeddings)'}"
    )

    # --- Query for chunks that need embeddings ---
    query = db.table("document_chunks").select("id, content, document_id")

    if document_id:
        # Limit to specific document
        query = query.eq("document_id", document_id)
    # Note: filtering for NULL embedding is done after fetching
    # because Supabase client may not support IS NULL directly
    # TODO: Use .is_("embedding", "null") if Supabase client supports it

    result = query.execute()
    all_chunks = result.data or []

    # Filter to only chunks without embeddings (NULL or empty)
    chunks_to_process = [
        c for c in all_chunks
        if not c.get("embedding")
    ]

    if not chunks_to_process:
        logger.info("No chunks need embedding backfill. All good!")
        return

    logger.info(f"Found {len(chunks_to_process)} chunks needing embeddings")

    # --- Process in batches ---
    processed = 0
    failed = 0

    for batch_start in range(0, len(chunks_to_process), BATCH_SIZE):
        batch = chunks_to_process[batch_start : batch_start + BATCH_SIZE]
        chunk_ids = [c["id"] for c in batch]
        chunk_texts = [c["content"] for c in batch]

        logger.info(
            f"Processing batch {batch_start // BATCH_SIZE + 1}: "
            f"{len(batch)} chunks"
        )

        try:
            # Generate embeddings for the batch
            embeddings = embed_texts(chunk_texts)

            # Update each chunk in the DB with its new embedding
            for chunk_id, embedding in zip(chunk_ids, embeddings):
                db.table("document_chunks").update(
                    {"embedding": embedding}
                ).eq("id", chunk_id).execute()

            processed += len(batch)
            logger.info(f"Batch complete: {processed} chunks updated so far")

        except Exception as e:
            logger.error(f"Batch failed: {e}")
            failed += len(batch)

    logger.info(
        f"\nBackfill complete: {processed} updated, {failed} failed"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill missing embeddings")
    parser.add_argument(
        "--document-id",
        type=str,
        default=None,
        help="Only backfill chunks for this specific document UUID",
    )
    args = parser.parse_args()

    backfill_embeddings(document_id=args.document_id)
