# ============================================================
# app/domain/documents/front_matter.py — Body-start detection
#
# WHY THIS FILE EXISTS:
# Covers, tables of contents, copyright notices and committee credit
# lists ("front matter") share no vocabulary with the questions users
# ask, yet they reach the answer context whenever their text happens
# to collide with a query — the measured case: a committee list full
# of Costa Rican universities outscoring real content for the query
# "¿Qué normativa aplica en Costa Rica?".
#
# Three cheaper approaches were measured and rejected before this one
# (see the task log for the numbers):
#
#   1. Single text features (relative position, regulatory-verb
#      density, dot-leaders, digit density): every range OVERLAPS
#      between front matter and legitimate low-index body text.
#   2. Per-subtype rules (cover / TOC / legal boilerplate): 11/17
#      recall with 4 false positives ON REAL REQUIREMENTS, and it
#      missed exactly the chunks that caused the regression.
#   3. LLM classification per chunk: better (29/35), but it cannot
#      be right about a chunk that is HALF credits and HALF
#      introduction — and the standard sliding window produces
#      exactly such chunks, soldering the last line of front matter
#      to the first paragraph of the body.
#
# The conclusion of (3) is the design here: the problem is a
# CHUNKING BOUNDARY, not a classification. Detect where the body
# starts in the RAW text (where line structure still exists — the
# chunker collapses it), force a hard cut there, and flag everything
# before it. On cleanly-cut chunks the binary label is correct by
# construction, exactly as it was for is_reference_list.
#
# WHAT MARKS THE BODY START:
# A heading occupying its own line — "0 INTRODUCCIÓN", "Executive
# Summary", "1 Scope"... The line anchoring does the heavy lifting,
# same trick as the bibliography detector: a table-of-contents entry
# ("0 Introducción vii") carries a trailing page number, so an
# anchored match rejects it while accepting the real heading.
#
# DELIBERATE LIMITS (measured, not guessed):
#   - No marker found -> None -> document is chunked exactly as
#     before and nothing is flagged. Under-flagging is the safe
#     direction; ~half the corpus (Spanish regulations that open
#     directly with "considerando..." clauses) has no such heading.
#   - Front matter that RESUMES after the body starts (SAFEAI's TOC
#     sits after its Executive Summary; OWASP-GenAI interleaves a
#     signature block) is NOT caught. One boundary, one prefix.
#   - Documents whose extraction shatters lines one-word-per-line
#     (Agentic-AI's "Introduction" is inline with the paragraph)
#     yield no anchor and stay unflagged.
# ============================================================

from __future__ import annotations

import re

# Body-start headings, ES + EN. Each must occupy its own line.
#
# "PRÓLOGO" is deliberately ABSENT: in this corpus the prologue is
# editorial material about the issuing body (INTECO's self-description),
# not document content — it belongs to the front matter it introduces.
#
# "Overview" and "Scope" are generic words; they are safe here only
# because the search takes the EARLIEST match and caps it to the first
# MAX_BODY_START_FRACTION of the document, so a mid-document
# subsection heading never wins over a real front-matter boundary.
_BODY_START_RE = re.compile(
    r"(?m)^[ \t]*(?:"
    r"(?:0\.?[ \t]+)?INTRODUCCI[ÓO]N"
    r"|(?:1\.?[ \t]+)?Introduction"
    r"|Executive[ \t]+Summary"
    r"|Resumen[ \t]+Ejecutivo"
    r"|(?:1\.?[ \t]+)?Objeto[ \t]+y[ \t]+campo[ \t]+de[ \t]+aplicaci[óo]n"
    r"|(?:1\.?[ \t]+)?Scope"
    r"|Overview"
    r")[ \t]*$",
    re.IGNORECASE,
)

# A boundary deeper than this fraction of the document is not a
# front-matter boundary — it is some interior heading that happens to
# share the name. Front matter in the measured corpus ends by 6.3%
# (ISO 42001, the deepest case); 20% leaves margin without letting a
# chapter-level "Introduction" at 40% wipe half a document.
MAX_BODY_START_FRACTION = 0.20

# A boundary this close to the start means the document has no real
# front matter (title-only). Cutting there would only manufacture a
# stub chunk, so it is treated as "no boundary".
MIN_BODY_START_CHARS = 80


def find_body_start(text: str) -> int | None:
    """
    Character offset where the document body begins, or None.

    None means "no reliable boundary": the caller must chunk the
    document exactly as if this module did not exist. Every guard
    errs toward None — a missed boundary leaves a document unflagged
    (the pre-existing behaviour), while a wrong boundary would flag
    real content as front matter.

    Args:
        text: The full raw document text, WITH its line structure.
              This must run before chunking — chunk content has been
              re-joined with spaces and no longer anchors.

    Returns:
        Offset of the first character of the body-start heading, or
        None when no heading is found inside the accepted window.
    """
    if not text:
        return None

    match = _BODY_START_RE.search(text)
    if match is None:
        return None

    offset = match.start()
    if offset < MIN_BODY_START_CHARS:
        return None
    if offset > len(text) * MAX_BODY_START_FRACTION:
        return None
    return offset
