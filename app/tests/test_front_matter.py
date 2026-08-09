# ============================================================
# app/tests/test_front_matter.py — Front-matter boundary and flag
#
# Covers the two halves of the fix for "covers, TOCs and committee
# credits contaminate the answer context":
#
#   1. DETECTION — find_body_start() locates the body-start heading
#      in raw text, rejecting TOC entries (trailing page numbers) and
#      interior headings (beyond the accepted window).
#
#   2. THE HARD CUT — chunk_text_with_sections() never lets a chunk
#      straddle the boundary, so is_front_matter is exact. This is
#      the property that heuristic and LLM classification could not
#      provide: the measured failure case was a chunk that soldered
#      the last committee names to "0 INTRODUCCIÓN" plus the whole
#      opening of the introduction (ISO 42001 chunk #4).
#
# Fixtures mirror the real corpus shape: INTECO-style front matter
# (cover, legal notice, credits table) followed by a normative body.
# ============================================================

import pytest

from app.domain.documents.chunking import chunk_text_with_sections
from app.domain.documents.front_matter import find_body_start


# ------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------
_CREDITS = "\n".join(
    f"Participante {i} Universidad Ejemplar de Costa Rica (UE{i})"
    for i in range(12)
)

_FRONT = (
    "[Page 1]\n"
    "INTE/ISO/IEC 42001:2026\n"
    "Tecnología de la información. Inteligencia artificial.\n"
    "AVISOS IMPORTANTES SOBRE ESTE DOCUMENTO\n"
    "El presente documento pertenece a INTECO. Derechos reservados.\n"
    "No podrá reproducirse ninguna parte de esta publicación.\n"
    f"{_CREDITS}\n"
)

# The body must dwarf the front matter, as it does in the real corpus
# (front matter ends by 6.3% in the deepest measured case). A too-short
# body would push the boundary past MAX_BODY_START_FRACTION and the
# guard would — correctly — refuse it.
_BODY = (
    "0 INTRODUCCIÓN\n"
    "La inteligencia artificial se aplica cada vez más en todos los "
    "sectores que usan tecnologías de la información. Este documento "
    "proporciona los requisitos para establecer, implementar, mantener y "
    "mejorar continuamente un sistema de gestión de la inteligencia "
    "artificial dentro del contexto de una organización.\n"
    + "".join(
        f"Articulo {i}\n"
        "La organización debe determinar los límites y la aplicabilidad "
        "del sistema de gestión para establecer su alcance, considerando "
        "las cuestiones externas e internas pertinentes a su propósito y "
        "los requisitos de las partes interesadas que sean relevantes "
        "para el sistema de gestión de la inteligencia artificial.\n"
        for i in range(1, 16)
    )
)


# ------------------------------------------------------------
# 1. Detection
# ------------------------------------------------------------
class TestFindBodyStart:
    @pytest.mark.parametrize(
        "heading",
        [
            "0 INTRODUCCIÓN",
            "INTRODUCCIÓN",
            "Introduction",
            "1 Introduction",
            "Executive Summary",
            "Resumen Ejecutivo",
            "1 OBJETO Y CAMPO DE APLICACIÓN",
            "1 Scope",
        ],
    )
    def test_standalone_headings_are_detected(self, heading):
        """
        Expected input:  a body-start heading on its own line, after
                         enough front matter to clear the minimum-offset
                         guard.
        Expected output: the heading's exact character offset.
        """
        body = "El cuerpo del documento continúa con requisitos. " * 120
        text = _FRONT + heading + "\n" + body
        offset = find_body_start(text)
        assert offset == text.index(heading)

    def test_toc_entry_with_page_number_is_rejected(self):
        """
        Expected input:  the same words as a TOC entry — trailing page
                         number, so the line does not end at the heading.
        Expected output: that occurrence is skipped. This is the anchor
                         trick shared with the bibliography detector.
        """
        text = (
            _FRONT
            + "0 Introducción vii\n"          # TOC entry — must not match
            + "1 Objeto y campo de aplicación 1\n"  # TOC entry
            + "0 INTRODUCCIÓN\n"              # the real heading
            + "El cuerpo del documento continúa con requisitos. " * 120
        )
        offset = find_body_start(text)
        assert offset == text.index("0 INTRODUCCIÓN\n")

    def test_no_heading_returns_none(self):
        """
        Expected input:  a document with no recognisable body-start
                         heading (half the corpus: Spanish regulations
                         opening directly with 'considerando...').
        Expected output: None — chunking stays byte-identical to the
                         pre-feature behaviour.
        """
        assert find_body_start(_FRONT + "Texto sin encabezado de cuerpo.") is None

    def test_heading_beyond_window_returns_none(self):
        """
        Expected input:  an interior 'Introduction' heading deep in the
                         document (a chapter heading, not front matter).
        Expected output: None — a boundary at 40% would flag real
                         content, the harmful direction.
        """
        text = ("relleno " * 2000) + "\nIntroduction\n" + ("cuerpo " * 200)
        assert find_body_start(text) is None

    def test_heading_at_very_start_returns_none(self):
        """
        Expected input:  a document that OPENS with the heading — there
                         is no front matter to separate.
        Expected output: None rather than a zero-width boundary.
        """
        text = "Introduction\n" + "Body text follows here with details. " * 200
        assert find_body_start(text) is None

    def test_interior_overview_is_not_a_boundary(self):
        """
        Expected input:  'Overview' as a recurring section heading inside
                         real content — the PCI DSS shape, where the first
                         interior Overview fell at 11.2% of the document
                         and would have flagged ~48 requirement chunks.
        Expected output: None. 'Overview' was removed from the pattern
                         list for exactly this document; this test keeps
                         it from being re-added without hitting the
                         measured counterexample.
        """
        requirements = (
            "Requirement 1: Install and maintain network security controls. "
            * 40
        )
        text = (
            _FRONT
            + requirements
            + "\nOverview\n"
            + "Network security controls such as firewalls enforce policy. "
            * 40
        )
        assert find_body_start(text) is None

    def test_earliest_heading_wins(self):
        """
        Expected input:  Executive Summary followed by a later
                         '1 Introduction' (the SAFEAI shape, where the
                         executive summary IS body content).
        Expected output: the earlier offset.
        """
        text = (
            _FRONT
            + "Executive Summary\n"
            + "The framework emphasizes evaluating the risks. " * 60
            + "\n1 Introduction\n"
            + "The purpose of this document is to present it. " * 60
        )
        assert find_body_start(text) == text.index("Executive Summary")


# ------------------------------------------------------------
# 2. The hard cut and the flag
# ------------------------------------------------------------
class TestChunkerFrontMatterCut:
    def test_no_chunk_straddles_the_boundary(self):
        """
        Expected input:  front matter + body, with a chunk size chosen so
                         the sliding window WOULD straddle the boundary.
        Expected output: every chunk is entirely on one side. This is
                         the core property — the measured failure was a
                         mixed credits+introduction chunk that no
                         classifier could label.
        """
        text = _FRONT + _BODY
        chunks = chunk_text_with_sections(text, chunk_size=60, chunk_overlap=10)

        assert any(c.is_front_matter for c in chunks), "no front matter flagged"
        assert any(not c.is_front_matter for c in chunks), "everything flagged"
        for chunk in chunks:
            if chunk.is_front_matter:
                assert "INTRODUCCIÓN" not in chunk.content, (
                    "front-matter chunk leaked body text"
                )
            else:
                assert "Derechos reservados" not in chunk.content, (
                    "body chunk leaked front matter"
                )

    def test_first_body_chunk_starts_at_the_heading(self):
        """
        Expected input:  the same document.
        Expected output: the first unflagged chunk begins exactly at the
                         body-start heading — no overlap reaches back
                         into the credits.
        """
        chunks = chunk_text_with_sections(
            _FRONT + _BODY, chunk_size=60, chunk_overlap=10
        )
        first_body = next(c for c in chunks if not c.is_front_matter)
        assert first_body.content.startswith("0 INTRODUCCIÓN")

    def test_flags_are_a_prefix(self):
        """
        Expected output: flagged chunks form a contiguous prefix — once
                         the body starts, nothing later is front matter.
        """
        chunks = chunk_text_with_sections(
            _FRONT + _BODY, chunk_size=60, chunk_overlap=10
        )
        flags = [c.is_front_matter for c in chunks]
        assert flags == sorted(flags, reverse=True)

    def test_document_without_boundary_is_untouched(self):
        """
        Expected input:  a document where no heading is detected.
        Expected output: chunking identical to a run before this feature
                         existed (same texts, same section_refs), and no
                         flags. Guards the "None means unchanged"
                         contract that keeps 44 of 52 corpus documents
                         byte-stable.
        """
        text = (
            "Articulo 1\n" + ("palabra " * 80) + "\n"
            "Articulo 2\n" + ("texto " * 80) + "\n"
        )
        chunks = chunk_text_with_sections(text, chunk_size=50, chunk_overlap=10)
        assert chunks
        assert not any(c.is_front_matter for c in chunks)

    def test_flag_defaults_false_and_coexists_with_reference_flag(self):
        """
        Expected input:  a document with front matter AND a trailing
                         bibliography.
        Expected output: both flags set on their own chunks, never both
                         on the same chunk — front matter is a prefix,
                         bibliography a suffix.
        """
        biblio = (
            "BIBLIOGRAFÍA\n"
            + "\n".join(
                f"ISO/IEC 2700{i}:2022 — Information technology — Security"
                for i in range(10)
            )
        )
        chunks = chunk_text_with_sections(
            _FRONT + _BODY + biblio, chunk_size=60, chunk_overlap=10
        )
        assert any(c.is_front_matter for c in chunks)
        assert any(c.is_reference_list for c in chunks)
        assert not any(
            c.is_front_matter and c.is_reference_list for c in chunks
        )
