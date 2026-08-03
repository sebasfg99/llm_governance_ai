# ============================================================
# app/domain/documents/reference_sections.py — Bibliography detection
#
# WHY THIS FILE EXISTS:
# A bibliography is a list of standard TITLES. Embedded, it lands
# semantically very close to any question about those standards —
# closer than the substantive clause that merely mentions them in
# passing. On the ISO 42001 corpus this was measured directly: the
# chunk carrying the transparency/explicability clause ranked 227th
# by cosine similarity while two chunks of Anexo D bibliography
# ranked 3rd and 5th.
#
# So bibliography chunks must be identifiable, in order to be
# DEMOTED at rerank time (never deleted — see rerank_chunks()).
#
# HOW WE IDENTIFY THEM (and why not the simpler options):
#
#   section_ref alone — useless. The two offending chunks and the
#   substantive one it displaced all carry the same "ANEXO D" /
#   "ANEXO C" marker; the chunker's section regex has no notion of
#   a bibliography.
#
#   A heading match alone — too many false positives. "referencias"
#   occurs constantly inside running prose in the Spanish regulatory
#   corpus ("Toda referencia en la reglamentación...", "Para las
#   referencias con fecha..."), and the NIST AI RMF Playbook carries
#   72 per-subsection "References" blocks. Flagging on the word
#   alone demotes ~45% of that document.
#
#   Citation density alone — also too many false positives. Ranking
#   the whole corpus by "fraction of characters inside an ISO/NIST/
#   RFC identifier" puts vendor comparison tables, tables of
#   contents and diagram captions ABOVE the real bibliographies.
#
# What works is the conjunction of three signals:
#
#   1. A heading anchored to its own LINE in the raw document text.
#      This is why classification happens at chunking time: the
#      chunker collapses newlines when it joins words, so by the
#      time a chunk exists the line structure that distinguishes a
#      heading from the same word in a sentence is already gone.
#
#   2. The heading's material must DOMINATE the chunk (>= 60% of
#      its words). A chunk that is mostly substantive content with
#      a short reference block at the end is still useful content
#      and keeps its full rerank score.
#
#   3. The text after the heading must not read as prose, measured
#      as function-word ratio. Connected prose in both Spanish and
#      English runs 27-50% function words; lists of standard
#      titles, author names and citation identifiers run 6-20%.
#      Measured on the corpus, reference regions averaged 0.18 and
#      substantive regions 0.38.
#
# Plus carry-forward for CONTINUATION chunks: a bibliography longer
# than one chunk produces a second chunk with no heading in it at
# all (this is exactly the ISO 42001 idx=70 case). Once a chunk is
# seen to end inside a reference region, the following chunk is
# treated as still inside it as long as it stays non-prose.
# ============================================================

from __future__ import annotations

import re

# ------------------------------------------------------------
# Tunables
#
# Calibrated against the corpus. The gap between the two
# populations is wide (0.18 vs 0.38 mean function-word ratio), so
# 0.22 sits in the valley rather than on either slope.
# ------------------------------------------------------------

# Minimum share of a chunk's words that must fall after the heading
# before the chunk counts as "a bibliography" rather than "content
# that happens to end with a few references".
MIN_REFERENCE_FRACTION = 0.60

# Maximum function-word ratio for text to still count as a list of
# references rather than prose.
MAX_FUNCTION_WORD_RATIO = 0.22

# Continuation chunks shorter than this are not judged on their
# function-word ratio — the estimate is too noisy on a small sample.
# They inherit the carried state instead.
_MIN_WORDS_FOR_RATIO = 10

# How many words at the head of a continuation chunk decide whether
# the reference region really survived the chunk boundary.
#
# WHY A LEADING WINDOW AND NOT THE WHOLE CHUNK:
# Averaging over the whole chunk lets a long reference tail hide a
# substantive opening. That is not hypothetical — it is how the NIST
# AI RMF Playbook breaks. Its body style is bulleted question lists
# ("• What are the roles, responsibilities, and delegation of
# authorities...") whose whole-chunk function-word ratio already sits
# near the threshold, so carry-forward leaked into eight chunks of
# real content. A reference region that genuinely continues past a
# boundary continues from the FIRST word; if prose is what resumes
# there, the region ended with the previous chunk.
_CONTINUATION_HEAD_WORDS = 40


# ------------------------------------------------------------
# Reference-section heading
#
# Anchored with ^...$ (MULTILINE) so it only matches a heading that
# occupies its own line. That single anchor is what rejects both
# failure modes seen in the corpus:
#
#   - inline prose:  "...para las referencias con fecha..."  (no
#     line break around it)
#   - TOC entries:   "— Bibliografía 63" / "Referencias
#     normativas ........ 12"  (trailing page number or dot
#     leaders, so the line does not end at the heading)
#
# The optional ANEXO/APPENDIX prefix catches the common
# "APPENDIX I. REFERENCES" style where the bibliography is itself
# a numbered annex.
# ------------------------------------------------------------
_REFERENCE_HEADING_RE = re.compile(
    r"(?m)^[\s\-—–•*]*"
    r"(?:(?:ANEXO|ANNEX|AP[EÉ]NDICE|APPENDIX)\s+[A-Z0-9IVXLC]+\s*[.\-—–:]?\s*)?"
    r"(?:\d+(?:\.\d+)*\s*[.\-—–:]?\s*)?"
    r"(?:BIBLIOGRAF[IÍ]A"
    r"|BIBLIOGRAPHY"
    r"|REFERENCIAS(?:\s+(?:NORMATIVAS|BIBLIOGR[AÁ]FICAS))?"
    r"|REFERENCES"
    r"|NORMATIVE\s+REFERENCES"
    r"|DOCUMENTOS?\s+DE\s+REFERENCIA"
    r"|OBRAS?\s+CITADAS"
    r"|WORKS\s+CITED)"
    r"\s*:?\s*$",
    re.IGNORECASE,
)


# ------------------------------------------------------------
# Function words (Spanish + English)
#
# Deliberately limited to closed-class words — articles,
# prepositions, pronouns, auxiliaries, conjunctions. These are the
# words that connected prose cannot avoid and that a list of titles
# and author names barely uses.
# ------------------------------------------------------------
_FUNCTION_WORDS = frozenset(
    """
    the a an is are was were be been being of to in on for and or but with at by
    from as that this these those it its their they we you have has had not no if
    when while which who whose how what why where than then so such can could may
    might must shall should will would do does did there here about into over
    under between during through each other any all more most some own same
    el la los las un una unos unas y o u de del al en por para con se su sus es
    son fue fueron ha han que como cual donde sobre entre cuando si lo le les nos
    este esta estos estas ese esa esos esas aquel ademas tambien pero porque
    debe deben debera deberia podra puede pueden ser estar hacer todo toda todos
    todas cada otro otra segun desde hasta ante bajo tras sin ya muy mas menos
    """.split()
)

_WORD_RE = re.compile(r"[^\W\d_]+", re.UNICODE)


def function_word_ratio(words: list[str]) -> float:
    """
    Share of `words` that are closed-class function words.

    Numbers and punctuation-only tokens are ignored so that a
    citation like "ISO/IEC 24027:2021" does not dilute the measure
    in either direction. Returns 0.0 for input with no alphabetic
    tokens, which reads as "not prose" — correct for the dot-leader
    and identifier-only fragments this is applied to.
    """
    alpha: list[str] = []
    for word in words:
        alpha.extend(m.group(0).lower() for m in _WORD_RE.finditer(word))
    if not alpha:
        return 0.0
    return sum(1 for w in alpha if w in _FUNCTION_WORDS) / len(alpha)


def find_reference_headings(text: str) -> list[int]:
    """
    Character offsets of every standalone reference/bibliography
    heading in `text`. Offsets point at the START of the heading
    match, i.e. the boundary where reference material begins.
    """
    return [m.start() for m in _REFERENCE_HEADING_RE.finditer(text)]


class ReferenceRegionClassifier:
    """
    Stateful, document-order classifier for chunks of one document.

    Stateful because a bibliography can outlive the chunk that
    introduced it: the heading appears in chunk N and the rest of
    the list spills into chunk N+1, which contains no heading at
    all and is otherwise indistinguishable from any other chunk.

    Usage — call classify() once per chunk, in document order:

        classifier = ReferenceRegionClassifier(heading_word_indices)
        for start, end, words in chunks:
            flag = classifier.classify(start, end, words)
    """

    def __init__(self, heading_word_indices: list[int]) -> None:
        # Word indices at which reference material starts, ascending.
        self._headings = sorted(heading_word_indices)
        # True when the previous chunk ended inside a reference region.
        self._in_region = False

    def classify(self, start: int, end: int, words: list[str]) -> bool:
        """
        Decide whether the chunk spanning word indices [start, end)
        is predominantly a reference list.

        Args:
            start: Index of the chunk's first word in the document.
            end:   Index one past the chunk's last word.
            words: The chunk's words (already sliced by the caller).

        Returns:
            True when the chunk should be flagged is_reference_list.
        """
        if not words:
            self._in_region = False
            return False

        # The LAST heading inside the chunk is the one that matters:
        # everything after it is the reference tail. An earlier
        # heading in the same chunk would already have been closed
        # by that later one.
        heading = self._last_heading_in(start, end)

        if heading is not None:
            tail = words[heading - start:]
            if not tail:
                self._in_region = False
                return False
            tail_fraction = len(tail) / len(words)
            tail_is_list = function_word_ratio(tail) <= MAX_FUNCTION_WORD_RATIO
            # The region continues into the next chunk only if the
            # tail really is a list. When prose resumed after a short
            # reference block, the region is already over.
            self._in_region = tail_is_list
            return tail_is_list and tail_fraction >= MIN_REFERENCE_FRACTION

        if self._in_region:
            # Continuation chunk. Too-short chunks keep the carried
            # state rather than be judged on a ratio computed from a
            # handful of words.
            if len(words) < _MIN_WORDS_FOR_RATIO:
                return True
            # The region survives the boundary only if the chunk OPENS
            # as a list — prose at the head means the bibliography
            # ended with the previous chunk. Both the head and the
            # whole chunk must qualify, so a substantive opening
            # cannot be averaged away by a long reference tail.
            head = words[:_CONTINUATION_HEAD_WORDS]
            still_list = (
                function_word_ratio(head) <= MAX_FUNCTION_WORD_RATIO
                and function_word_ratio(words) <= MAX_FUNCTION_WORD_RATIO
            )
            self._in_region = still_list
            return still_list

        return False

    def _last_heading_in(self, start: int, end: int) -> int | None:
        found: int | None = None
        for word_idx in self._headings:
            if word_idx >= end:
                break
            if word_idx >= start:
                found = word_idx
        return found
