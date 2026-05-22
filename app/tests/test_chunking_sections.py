# ============================================================
# app/tests/test_chunking_sections.py — Section-aware chunker tests
#
# Validates the new chunk_text_with_sections() in
# app/domain/documents/chunking.py:
#
#   - Markers in Spanish + English are detected.
#   - Each chunk gets the correct section_ref attribution.
#   - Chunks prefer to end at section boundaries.
#   - Documents without markers degrade to plain sliding window.
#   - Empty / whitespace input returns [].
#   - Backward-compatible chunk_text() drops sections cleanly.
# ============================================================

from app.domain.documents.chunking import (
    ChunkWithSection,
    chunk_text,
    chunk_text_with_sections,
)


# ------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------
def _padded(words: int) -> str:
    """Generate `words` filler tokens to bulk up a section."""
    return " ".join(f"w{i}" for i in range(words))


# ------------------------------------------------------------
# Documents WITH formal section markers
# ------------------------------------------------------------
class TestSectionAwareWithMarkers:
    def test_articulo_markers_are_detected_in_spanish(self):
        text = (
            "Articulo 1\n"
            f"{_padded(40)}\n\n"
            "Articulo 2\n"
            f"{_padded(40)}\n\n"
            "Articulo 3\n"
            f"{_padded(40)}\n"
        )

        chunks = chunk_text_with_sections(text, chunk_size=30, chunk_overlap=5)

        assert len(chunks) >= 3
        # Some chunk should be tagged with each article we wrote.
        tagged = {c.section_ref for c in chunks if c.section_ref}
        assert any("Articulo 1" in t or "articulo 1" in t.lower() for t in tagged)
        assert any("Articulo 2" in t or "articulo 2" in t.lower() for t in tagged)
        assert any("Articulo 3" in t or "articulo 3" in t.lower() for t in tagged)

    def test_seccion_markers_with_decimal_numbering(self):
        text = (
            "Seccion 6.1\n"
            f"{_padded(60)}\n\n"
            "Seccion 6.2\n"
            f"{_padded(60)}\n"
        )

        chunks = chunk_text_with_sections(text, chunk_size=40, chunk_overlap=5)

        tagged = [c.section_ref for c in chunks if c.section_ref]
        assert any("6.1" in t for t in tagged)
        assert any("6.2" in t for t in tagged)

    def test_capitulo_with_roman_numerals(self):
        text = (
            "Capitulo III\n"
            f"{_padded(50)}\n\n"
            "Capitulo IV\n"
            f"{_padded(50)}\n"
        )

        chunks = chunk_text_with_sections(text, chunk_size=40, chunk_overlap=5)

        tagged = [c.section_ref for c in chunks if c.section_ref]
        assert any("III" in t for t in tagged)
        assert any("IV" in t for t in tagged)

    def test_clausula_marker_for_iso_style_documents(self):
        text = (
            "Clausula 6.1.2\n"
            f"{_padded(40)}\n\n"
            "Clausula 6.1.3\n"
            f"{_padded(40)}\n"
        )

        chunks = chunk_text_with_sections(text, chunk_size=30, chunk_overlap=5)

        tagged = [c.section_ref for c in chunks if c.section_ref]
        assert any("6.1.2" in t for t in tagged)
        assert any("6.1.3" in t for t in tagged)

    def test_english_article_marker(self):
        text = (
            "Article 9\n"
            f"{_padded(50)}\n\n"
            "Article 10\n"
            f"{_padded(50)}\n"
        )

        chunks = chunk_text_with_sections(text, chunk_size=40, chunk_overlap=5)

        tagged = [c.section_ref for c in chunks if c.section_ref]
        assert any("Article 9" in t or "article 9" in t.lower() for t in tagged)
        assert any("Article 10" in t or "article 10" in t.lower() for t in tagged)

    def test_anexo_and_apendice_markers(self):
        text = (
            "Anexo A\n"
            f"{_padded(40)}\n\n"
            "Apendice B\n"
            f"{_padded(40)}\n"
        )

        chunks = chunk_text_with_sections(text, chunk_size=30, chunk_overlap=5)

        tagged = [c.section_ref for c in chunks if c.section_ref]
        assert any("Anexo A" in t or "anexo a" in t.lower() for t in tagged)
        assert any("Apendice B" in t or "apendice b" in t.lower() for t in tagged)


# ------------------------------------------------------------
# Section attribution semantics
# ------------------------------------------------------------
class TestSectionAttributionRules:
    def test_chunk_inherits_most_recent_marker(self):
        """A chunk after Article 1 must be tagged with Article 1, not None."""
        text = "Articulo 1\n" + _padded(20)
        chunks = chunk_text_with_sections(text, chunk_size=30, chunk_overlap=5)

        assert chunks[0].section_ref is not None
        assert "Articulo 1" in chunks[0].section_ref or "articulo 1" in chunks[0].section_ref.lower()

    def test_preface_before_first_marker_is_section_none(self):
        """
        Content before the first section marker has no preceding section,
        so section_ref must be None on those chunks.
        """
        text = (
            f"{_padded(40)}\n\n"  # preface, no marker
            "Articulo 1\n"
            f"{_padded(40)}\n"
        )

        chunks = chunk_text_with_sections(text, chunk_size=20, chunk_overlap=5)

        # First chunk falls in the preface and must NOT carry Articulo 1.
        assert chunks[0].section_ref is None
        # At least one later chunk should carry Articulo 1.
        assert any(
            c.section_ref and "rticulo 1" in c.section_ref for c in chunks[1:]
        )

    def test_section_change_propagates_forward(self):
        """
        After a new marker appears, subsequent chunks must use it (not the
        old one), even when the chunker creates several consecutive
        chunks per section.
        """
        text = (
            "Articulo 1\n"
            f"{_padded(80)}\n\n"
            "Articulo 2\n"
            f"{_padded(80)}\n"
        )

        chunks = chunk_text_with_sections(text, chunk_size=20, chunk_overlap=5)

        first_a2_idx = next(
            (
                i
                for i, c in enumerate(chunks)
                if c.section_ref and "rticulo 2" in c.section_ref
            ),
            None,
        )
        assert first_a2_idx is not None
        # Once we see Article 2, no later chunk should regress to Article 1.
        for c in chunks[first_a2_idx:]:
            assert not (c.section_ref and "rticulo 1" in c.section_ref)


# ------------------------------------------------------------
# Documents WITHOUT markers
# ------------------------------------------------------------
class TestSectionAwareWithoutMarkers:
    def test_plain_prose_returns_chunks_with_none_section(self):
        """
        Documents with no formal markers (whitepapers, NIST playbooks, etc.)
        must still be chunked. Every chunk gets section_ref = None.
        """
        text = _padded(200)

        chunks = chunk_text_with_sections(text, chunk_size=50, chunk_overlap=10)

        assert len(chunks) >= 2
        for c in chunks:
            assert c.section_ref is None

    def test_overlap_is_preserved_in_unmarked_text(self):
        """Unmarked text falls back to plain sliding window with overlap."""
        text = " ".join([f"unique{i}" for i in range(80)])

        chunks = chunk_text_with_sections(text, chunk_size=20, chunk_overlap=5)

        if len(chunks) >= 2:
            words0 = set(chunks[0].content.split())
            words1 = set(chunks[1].content.split())
            assert words0 & words1, "Overlap missing in unmarked text"


# ------------------------------------------------------------
# Backward compatibility
# ------------------------------------------------------------
class TestBackwardCompatibility:
    def test_chunk_text_returns_strings_only(self):
        """
        chunk_text() must keep returning a list[str] so existing callers
        and tests don't break.
        """
        text = "Articulo 1\n" + _padded(60)
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)

        assert isinstance(chunks, list)
        assert all(isinstance(c, str) for c in chunks)
        assert len(chunks) >= 1

    def test_chunk_text_with_sections_returns_named_tuples(self):
        text = "Articulo 1\n" + _padded(40)
        chunks = chunk_text_with_sections(text, chunk_size=20, chunk_overlap=5)

        assert all(isinstance(c, ChunkWithSection) for c in chunks)
        assert all(hasattr(c, "content") and hasattr(c, "section_ref") for c in chunks)


# ------------------------------------------------------------
# Edge cases
# ------------------------------------------------------------
class TestEdgeCases:
    def test_empty_text(self):
        assert chunk_text_with_sections("") == []

    def test_whitespace_only_text(self):
        assert chunk_text_with_sections("   \n\n\t  ") == []

    def test_marker_only_document(self):
        """Pathological case: just headers, no body. Should still produce chunks."""
        text = "Articulo 1\nArticulo 2\nArticulo 3"
        chunks = chunk_text_with_sections(text, chunk_size=20, chunk_overlap=2)

        assert len(chunks) >= 1
        assert any(c.section_ref for c in chunks)

    def test_invalid_overlap_raises(self):
        import pytest

        with pytest.raises(ValueError):
            chunk_text_with_sections(_padded(20), chunk_size=10, chunk_overlap=10)
        with pytest.raises(ValueError):
            chunk_text_with_sections(_padded(20), chunk_size=10, chunk_overlap=15)
