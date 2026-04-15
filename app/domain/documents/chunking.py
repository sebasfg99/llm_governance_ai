# ============================================================
# app/domain/documents/chunking.py — Text chunking for RAG
#
# WHY THIS FILE EXISTS:
# An LLM has a limited context window — you can't pass an entire
# 50-page document as context. Chunking splits the document into
# small overlapping pieces so:
#   1. Each chunk fits within the LLM's context budget
#   2. Relevant chunks can be retrieved individually
#   3. The overlap prevents important sentences from being cut off
#
# CURRENT STRATEGY: Fixed-size with overlap
#
#   Text: "The quick brown fox jumps over the lazy dog..."
#   chunk_size=5 tokens, overlap=2:
#     Chunk 0: "The quick brown fox jumps"
#     Chunk 1: "fox jumps over the lazy"
#     Chunk 2: "the lazy dog..."
#
# WHY FIXED-SIZE (not sentence-aware):
#   Simpler to implement and reason about for MVP.
#   Works well for dense prose documents.
#
# TODO: Add sentence-boundary-aware chunking using nltk.sent_tokenize
# TODO: Add recursive character splitting (better for code/structured text)
# TODO: Add semantic chunking (groups related sentences together)
# ============================================================

from app.core.config import settings
from app.core.logging import get_logger

# Logger for this module
logger = get_logger(__name__)


def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split text into fixed-size overlapping chunks for RAG.

    Uses whitespace tokenization — each "token" is approximately
    one word. This is a simple proxy for actual token counting
    (which would require a tokenizer like tiktoken).

    For the MVP, word count ≈ token count is accurate enough.
    A typical English word is ~1.3 tokens in GPT tokenizers.

    Args:
        text: The full document text to split.
        chunk_size: Max number of words per chunk. Defaults to settings.chunk_size (500).
        chunk_overlap: Number of overlapping words between adjacent chunks.
                       Defaults to settings.chunk_overlap (50).

    Returns:
        A list of text chunk strings. Each chunk is a substring
        of the original text. The list is ordered (chunk 0 comes
        first in the document).

    Example:
        chunks = chunk_text("Long document text here...", chunk_size=100, chunk_overlap=10)
        print(f"Produced {len(chunks)} chunks")
    """
    # Use defaults from settings if not explicitly provided
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    # Validate inputs
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )

    if not text or not text.strip():
        logger.warning("chunk_text called with empty text — returning empty list")
        return []

    # Split the text into "tokens" (words) by whitespace
    # This is O(n) and simple — no external tokenizer needed
    words = text.split()

    if not words:
        return []

    total_words = len(words)
    logger.debug(f"Chunking {total_words} words with size={chunk_size}, overlap={chunk_overlap}")

    chunks: list[str] = []

    # Sliding window:
    # - Start position advances by (chunk_size - chunk_overlap) each iteration
    # - This creates the overlap between consecutive chunks
    step = chunk_size - chunk_overlap  # How far to advance the window each step

    start = 0
    while start < total_words:
        # Calculate the end index for this chunk
        # min() prevents going past the end of the word list
        end = min(start + chunk_size, total_words)

        # Join the words in this window back into a text string
        chunk = " ".join(words[start:end])

        # Only add non-empty chunks (shouldn't happen but defensive)
        if chunk.strip():
            chunks.append(chunk)

        # If we've reached the end, stop
        if end == total_words:
            break

        # Advance the window
        start += step

    logger.debug(
        f"Chunking complete: {total_words} words → {len(chunks)} chunks "
        f"(avg {total_words // max(len(chunks), 1)} words/chunk)"
    )

    return chunks


def estimate_token_count(text: str) -> int:
    """
    Rough estimate of GPT token count for a text string.

    This is a heuristic approximation: 1 token ≈ 0.75 words
    (or about 4 characters) for English text.

    Used for logging and debugging — not for enforcing hard limits.
    For exact token counting, use tiktoken:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")
        tokens = enc.encode(text)

    TODO: Replace with tiktoken for accurate token counting
    """
    # Simple approximation based on word count
    word_count = len(text.split())
    # GPT tokenizers average ~1.3 tokens per word for English
    return int(word_count * 1.3)
