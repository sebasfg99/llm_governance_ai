# ============================================================
# app/domain/documents/embeddings.py — Text embedding generation
#
# WHY THIS FILE EXISTS:
# Embeddings are the mathematical backbone of RAG.
# This module converts text into vectors (lists of numbers)
# that represent the semantic meaning of the text.
#
# HOW EMBEDDINGS WORK:
# Two texts with similar meaning produce vectors that are
# "close" to each other in vector space (cosine similarity).
# This is how we find relevant chunks: we embed the user's
# question and find the chunks with the most similar embeddings.
#
# EXAMPLE:
#   embed("What are the termination clauses?")
#   → [0.021, -0.045, 0.133, ...]  # 1536 numbers
#
#   embed("Section 8: Termination of Agreement")
#   → [0.019, -0.041, 0.128, ...]  # Very similar vector!
#
#   The vectors are close → this chunk is retrieved → the LLM
#   sees the relevant contract clause as context.
#
# WHY text-embedding-3-small:
#   - 1536 dimensions (good quality)
#   - Very fast and cheap ($0.02 per 1M tokens as of 2024)
#   - Same OpenAI API key as GPT models
#
# NOTE: This module is imported by BOTH:
#   - domain/documents/service.py (to embed chunks on ingestion)
#   - domain/retrieval/service.py (to embed query on retrieval)
# TODO: Move to app/core/embeddings.py to clarify this shared use
# ============================================================

import openai
from app.core.config import settings
from app.core.logging import get_logger
from app.core.errors import UpstreamError

# Logger for this module
logger = get_logger(__name__)

# ============================================================
# OpenAI client instance (module-level singleton)
#
# We create the client once at module load time.
# The client is thread-safe and reused across all calls.
# ============================================================
_openai_client: openai.OpenAI | None = None


def _get_openai_client() -> openai.OpenAI:
    """Lazily initialize and return the OpenAI client."""
    global _openai_client
    if _openai_client is None:
        _openai_client = openai.OpenAI(api_key=settings.openai_api_key)
    return _openai_client


def embed_texts(texts: list[str]) -> list[list[float]]:
    """
    Convert a list of text strings into embedding vectors.

    Calls the OpenAI Embeddings API and returns one vector
    per input text. Each vector has 1536 dimensions (for
    text-embedding-3-small).

    The order of output vectors matches the order of input texts:
    embeddings[0] is the embedding for texts[0], etc.

    Args:
        texts: List of text strings to embed. Should not be empty.
               Max 100 items per batch (API limit is 2048 but we
               batch at 100 for reliability).

    Returns:
        List of float vectors, one per input text.
        Shape: [[float, ...] * 1536] * len(texts)

    Raises:
        UpstreamError: If the OpenAI API call fails.
        ValueError: If texts is empty.

    Example:
        vectors = embed_texts(["Hello world", "How are you?"])
        # vectors[0] is the embedding for "Hello world"
        # len(vectors[0]) == 1536
    """
    if not texts:
        raise ValueError("embed_texts requires at least one text input")

    # Remove empty strings — the API rejects them
    clean_texts = [t.strip() for t in texts if t.strip()]
    if not clean_texts:
        raise ValueError("All input texts were empty after stripping whitespace")

    logger.debug(f"Embedding {len(clean_texts)} text(s) using {settings.embedding_model}")

    # ---- Batch processing ----
    # The API can handle many texts at once, but we batch at 100
    # to stay well within limits and avoid timeouts on large documents.
    BATCH_SIZE = 100
    all_embeddings: list[list[float]] = []

    # Process in batches
    for batch_start in range(0, len(clean_texts), BATCH_SIZE):
        batch = clean_texts[batch_start : batch_start + BATCH_SIZE]

        logger.debug(
            f"Embedding batch {batch_start // BATCH_SIZE + 1}: "
            f"{len(batch)} texts"
        )

        try:
            # Call the OpenAI Embeddings API
            # Returns an EmbeddingResponse object
            response = _get_openai_client().embeddings.create(
                model=settings.embedding_model,    # "text-embedding-3-small"
                input=batch,                        # List of text strings
                # encoding_format="float" is the default — returns Python floats
            )

        except openai.AuthenticationError as e:
            raise UpstreamError(
                "OpenAI API key is invalid or missing. Check OPENAI_API_KEY in your .env",
                detail={"provider": "openai", "error": str(e)},
            )
        except openai.RateLimitError as e:
            raise UpstreamError(
                "OpenAI rate limit exceeded. Consider adding retry logic.",
                detail={"provider": "openai", "error": str(e)},
            )
        except openai.APIError as e:
            raise UpstreamError(
                f"OpenAI API error during embedding: {e}",
                detail={"provider": "openai", "error": str(e)},
            )
        except Exception as e:
            raise UpstreamError(
                f"Unexpected error calling OpenAI embeddings: {e}",
                detail={"provider": "openai"},
            )

        # Extract the embedding vectors from the response
        # response.data is a list of Embedding objects, each with a .embedding field
        # The .embedding field is a list of floats (the vector)
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

    logger.debug(
        f"Embedding complete: {len(all_embeddings)} vectors, "
        f"each {len(all_embeddings[0])} dimensions"
    )

    return all_embeddings


def embed_single(text: str) -> list[float]:
    """
    Convenience wrapper to embed a single text string.

    Used by the retrieval service to embed user queries.

    Args:
        text: The text to embed (e.g., a user's question).

    Returns:
        A single 1536-dimensional float vector.

    Example:
        query_vector = embed_single("What are the termination clauses?")
        # Use query_vector to search document_chunks in the DB
    """
    vectors = embed_texts([text])
    return vectors[0]  # Return the single vector (not a list of lists)
