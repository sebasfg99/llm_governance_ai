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
# CURRENT STRATEGY: Word-based sliding window with section-aware
# boundary preference and section_ref tagging.
#
# WHY SECTION-AWARE:
# Normative documents (ISO 42001, NIST AI RMF, EU AI Act, SUGEF
# regulations, contracts) are organised by formal section markers
# ("Articulo X", "Seccion Y.Z", "Capitulo III", "Anexo A",
# "Apendice B", "Clausula 6.1.2"). Tagging each chunk with the
# section it belongs to lets the LLM cite at clause granularity
# instead of just at document granularity. The chunker also
# prefers to end a chunk just before the next section marker
# when one is close to the target chunk size, so chunks rarely
# cross section boundaries.
#
# Documents without formal markers (e.g. a NIST Playbook, a
# vendor whitepaper) gracefully degrade to plain sliding window
# with section_ref = None.
# ============================================================

from __future__ import annotations

import re
from typing import NamedTuple

from app.core.config import settings
from app.core.logging import get_logger
from app.domain.documents.reference_sections import (
    ReferenceRegionClassifier,
    find_reference_headings,
)

logger = get_logger(__name__)


# ------------------------------------------------------------
# Section marker regex
#
# Matches the start of a typical legal/normative section header.
# Designed to be permissive across Spanish and English corpora:
#
#   - Articulo 9, Article 9, Art. 9, Artículo 9.1
#   - Seccion 6.2, Sección 6.2, Section 6.2
#   - Capitulo III, Capítulo IV, Chapter II
#   - Anexo A, Anexo I, Annex A
#   - Apendice B, Apéndice C, Appendix C
#   - Clausula 6.1.2, Cláusula 6.1.2, Clause 6.1.2
#
# The (?m) inline flag turns on MULTILINE so '^' matches the
# start of any line, not only the start of the whole text.
# Hyphens in the section number ("6-1-2") are not currently
# matched — they're rare enough that we accept the miss.
# ------------------------------------------------------------
_SECTION_MARKER_RE = re.compile(
    r"(?m)^\s*("
    r"Art[íi]culo\s+\d+(?:[.\d]+)?(?:\s+[A-Za-zÀ-ÿ]+)?|"
    r"Article\s+\d+(?:[.\d]+)?|"
    r"Art\.\s*\d+(?:[.\d]+)?|"
    r"Secci[oó]n\s+[\d.]+|"
    r"Section\s+[\d.]+|"
    r"Cap[íi]tulo\s+[IVXLCDM]+|"
    r"Cap[íi]tulo\s+\d+(?:[.\d]+)?|"
    r"Chapter\s+[IVXLCDM]+|"
    r"Chapter\s+\d+(?:[.\d]+)?|"
    r"Anexo\s+[A-Z\d]+|"
    r"Annex\s+[A-Z\d]+|"
    r"Ap[eé]ndice\s+[A-Z\d]+|"
    r"Appendix\s+[A-Z\d]+|"
    r"Cl[aá]usula\s+[\d.]+|"
    r"Clause\s+[\d.]+"
    r")\b",
    re.IGNORECASE,
)


class ChunkWithSection(NamedTuple):
    """A chunk's text together with the section marker it belongs to."""

    content: str
    section_ref: str | None
    # True when the chunk is predominantly a bibliography / reference
    # list rather than substantive content. Defaults to False so
    # existing constructions of this tuple keep working unchanged.
    # Consumed by rerank_chunks(), which demotes these chunks so they
    # stop outranking the content that actually answers a question.
    is_reference_list: bool = False


# ------------------------------------------------------------
# Public API
# ------------------------------------------------------------
def chunk_text(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[str]:
    """
    Split text into overlapping chunks. Backward-compatible wrapper
    around chunk_text_with_sections() that returns only the chunk
    strings (drops section_ref). Tests and any caller that does
    not need section attribution can keep using this signature
    unchanged.
    """
    return [
        chunk.content
        for chunk in chunk_text_with_sections(
            text, chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
    ]


def chunk_text_with_sections(
    text: str,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[ChunkWithSection]:
    """
    Split text into overlapping chunks, each tagged with the section
    marker (Articulo / Seccion / Clausula / ...) it belongs to.

    Algorithm:

      1. Word-tokenise the document.
      2. Pre-scan the original text for legal section markers and
         project them onto word indices ("marker M starts at word i").
      3. Sliding window with the configured chunk_size and chunk_overlap.
      4. When the natural end of a window is within ~20% of the target
         and a section marker is close-by, prefer ending the chunk just
         BEFORE the marker so chunks don't straddle sections.
      5. Each emitted chunk is tagged with the most recent section
         marker that started at or before its first word.

    Documents with no markers degrade gracefully to plain sliding
    window with section_ref = None for every chunk.

    Args:
        text:           The full document text to split.
        chunk_size:     Max number of words per chunk. Defaults to
                        settings.chunk_size.
        chunk_overlap:  Number of overlapping words between adjacent
                        chunks. Defaults to settings.chunk_overlap.

    Returns:
        A list of ChunkWithSection in document order. May be empty
        if the input text is blank.
    """
    chunk_size = chunk_size or settings.chunk_size
    chunk_overlap = chunk_overlap or settings.chunk_overlap

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be less than chunk_size ({chunk_size})"
        )

    if not text or not text.strip():
        logger.warning("chunk_text_with_sections called with empty text — returning empty list")
        return []

    # Tokenise once. We keep the (start_offset_in_text, word) tuples so
    # we can map word indices back to character positions when locating
    # section markers.
    word_offsets: list[tuple[int, str]] = []
    for match in re.finditer(r"\S+", text):
        word_offsets.append((match.start(), match.group()))

    if not word_offsets:
        return []

    # Scan markers in the original text. Each entry is (word_index, marker).
    markers_by_word: list[tuple[int, str]] = _project_markers_to_word_indices(
        text, word_offsets
    )

    # Bibliography headings, projected onto the same word-index space.
    # This has to happen HERE, on the raw text, because the reference
    # heading is only distinguishable from the same word used inside a
    # sentence by the line break around it — and chunk content is
    # produced by " ".join(words), which destroys that evidence.
    reference_classifier = ReferenceRegionClassifier(
        _project_offsets_to_word_indices(find_reference_headings(text), word_offsets)
    )

    total_words = len(word_offsets)
    logger.debug(
        f"Chunking {total_words} words with size={chunk_size}, "
        f"overlap={chunk_overlap}, markers={len(markers_by_word)}"
    )

    chunks: list[ChunkWithSection] = []
    step = chunk_size - chunk_overlap
    # Tolerance for the boundary preference: if the natural end is within
    # this many words of a marker, we cut at the marker instead. Capped
    # at 20% of chunk_size to avoid producing chunks that are too short
    # or too long.
    boundary_tolerance = max(1, chunk_size // 5)

    start = 0
    while start < total_words:
        target_end = min(start + chunk_size, total_words)

        # Section-aware boundary: look for a marker inside
        # (target_end - tolerance, target_end + tolerance) and snap to it.
        snap_end = _find_snap_boundary(
            markers_by_word=markers_by_word,
            target_end=target_end,
            window_low=target_end - boundary_tolerance,
            window_high=min(target_end + boundary_tolerance, total_words),
            min_size=max(1, chunk_size - boundary_tolerance),
            start=start,
        )
        end = snap_end if snap_end is not None else target_end

        # Materialise the chunk text from the word slice.
        words_slice = [w for _, w in word_offsets[start:end]]
        if words_slice:
            content = " ".join(words_slice)
            section_ref = _resolve_section_for_chunk(markers_by_word, start)
            # Must be called for every chunk in document order: the
            # classifier carries state so a bibliography that spills
            # past a chunk boundary is still recognised in the
            # continuation chunk, which contains no heading at all.
            is_reference_list = reference_classifier.classify(
                start, end, words_slice
            )
            chunks.append(
                ChunkWithSection(
                    content=content,
                    section_ref=section_ref,
                    is_reference_list=is_reference_list,
                )
            )

        if end >= total_words:
            break

        # Advance the window. When we snapped to a section boundary,
        # start the next chunk at the marker (no overlap into a new
        # section — overlapping into a new article would muddy
        # section_ref attribution). Otherwise, keep the configured
        # overlap as before.
        if snap_end is not None and snap_end != target_end:
            start = end
        else:
            start = end - chunk_overlap if end - chunk_overlap > start else end

    reference_chunks = sum(1 for c in chunks if c.is_reference_list)
    logger.debug(
        f"Chunking complete: {total_words} words -> {len(chunks)} chunks "
        f"(avg {total_words // max(len(chunks), 1)} words/chunk, "
        f"{reference_chunks} flagged as reference lists)"
    )

    return chunks


def estimate_token_count(text: str) -> int:
    """
    Rough estimate of GPT token count. 1 token ≈ 0.75 words for
    English text. Used for logging / debugging only.
    """
    word_count = len(text.split())
    return int(word_count * 1.3)


# ------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------
def _project_markers_to_word_indices(
    text: str,
    word_offsets: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """
    Walk every section marker hit in `text` and convert its character
    offset into a word index. Returns the list ordered by word index,
    which makes binary searches over it trivial later.
    """
    if not word_offsets:
        return []

    markers: list[tuple[int, str]] = []
    # Two-pointer scan: word offsets are already sorted, regex hits
    # arrive in document order, so we never need to rewind.
    next_word_idx = 0
    for match in _SECTION_MARKER_RE.finditer(text):
        marker_char = match.start()
        # Advance the word pointer until its char offset >= marker_char.
        while (
            next_word_idx < len(word_offsets)
            and word_offsets[next_word_idx][0] < marker_char
        ):
            next_word_idx += 1
        marker_text = match.group(1).strip()
        # Cap absurdly long matches; section refs should be terse.
        if len(marker_text) > 80:
            marker_text = marker_text[:80] + "…"
        # Avoid duplicates at the same word boundary (rare but possible
        # if regex alternatives overlap).
        if markers and markers[-1][0] == next_word_idx:
            continue
        markers.append((next_word_idx, marker_text))

    return markers


def _project_offsets_to_word_indices(
    char_offsets: list[int],
    word_offsets: list[tuple[int, str]],
) -> list[int]:
    """
    Convert a sorted list of character offsets into word indices,
    using the same forward-only two-pointer scan as
    _project_markers_to_word_indices(). Each result is the index of
    the first word starting at or after its character offset.
    """
    if not word_offsets or not char_offsets:
        return []

    indices: list[int] = []
    next_word_idx = 0
    for char_offset in char_offsets:
        while (
            next_word_idx < len(word_offsets)
            and word_offsets[next_word_idx][0] < char_offset
        ):
            next_word_idx += 1
        indices.append(next_word_idx)

    return indices


def _find_snap_boundary(
    markers_by_word: list[tuple[int, str]],
    target_end: int,
    window_low: int,
    window_high: int,
    min_size: int,
    start: int,
) -> int | None:
    """
    Look for a section marker inside [window_low, window_high]. If found,
    return its word index so the caller can use it as the chunk end.
    Returns None when no nearby marker exists or the resulting chunk
    would be smaller than min_size.

    Preference rule:
      - If a marker exists strictly between (start, target_end], snap to
        it (cut chunk a little early to avoid straddling).
      - Otherwise if a marker exists in (target_end, window_high], snap
        to it (slightly oversize chunk, but cleaner attribution).
    """
    if not markers_by_word:
        return None

    best_before: int | None = None
    best_after: int | None = None

    for word_idx, _ in markers_by_word:
        if word_idx <= start:
            continue
        if word_idx <= target_end and word_idx >= window_low:
            best_before = word_idx  # may overwrite — we want the latest one inside the window
        elif word_idx > target_end and word_idx <= window_high:
            if best_after is None:
                best_after = word_idx  # earliest marker after target wins
            break  # markers are sorted; anything later is farther out

    candidate = best_before if best_before is not None else best_after
    if candidate is None:
        return None
    if candidate - start < min_size:
        return None
    return candidate


def _resolve_section_for_chunk(
    markers_by_word: list[tuple[int, str]],
    chunk_start_word: int,
) -> str | None:
    """
    Find the most recent section marker that starts at or before
    `chunk_start_word`. Returns None if there are no preceding markers
    (typical for the document preface / front matter).
    """
    if not markers_by_word:
        return None

    current: str | None = None
    for word_idx, marker_text in markers_by_word:
        if word_idx <= chunk_start_word:
            current = marker_text
        else:
            break
    return current
