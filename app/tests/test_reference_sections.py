# ============================================================
# app/tests/test_reference_sections.py — Bibliography demotion
#
# Covers the two halves of the fix for "bibliography chunks
# outrank substantive content":
#
#   1. DETECTION — chunk_text_with_sections() flags a chunk as
#      is_reference_list when it is predominantly a reference
#      list, including the continuation chunk of a bibliography
#      that spills past a chunk boundary (the ISO 42001 idx=70
#      case, which contains no heading at all).
#
#   2. DEMOTION — rerank_chunks() pushes flagged chunks below
#      substantive ones, EXCEPT when the query is itself asking
#      about references, where the bibliography is the answer.
#
# The fixtures below are modelled on the real corpus text that
# produced the bug: Anexo D of the ISO 42001 replica, whose
# bibliography of ISO/IEC standard names ranked 3rd and 5th while
# the Anexo C clause that actually answered the question ranked
# 227th.
# ============================================================

import pytest

from app.domain.documents.chunking import chunk_text_with_sections
from app.domain.documents.reference_sections import (
    function_word_ratio,
    find_reference_headings,
)
from app.domain.retrieval.ranking import ChunkResult, rerank_chunks


# ------------------------------------------------------------
# Fixture builders
# ------------------------------------------------------------
_BIBLIOGRAPHY_ENTRIES = "\n".join(
    [
        "ISO/IEC 22989:2022 — Information technology — Artificial intelligence",
        "— Artificial intelligence concepts and terminology",
        "ISO/IEC 23894:2023 — Information technology — Artificial intelligence",
        "— Guidance on risk management",
        "ISO/IEC 38507:2022 — Information technology — Governance of IT",
        "ISO/IEC 5259-1 — Data quality for analytics and machine learning (ML)",
        "ISO/IEC 5338 — Information technology — AI system life cycle processes",
        "ISO/IEC 23053:2022 — Framework for Artificial Intelligence Systems",
        "ISO/IEC 25059 — Software engineering — Quality model for AI systems",
        "ISO/IEC 27001:2022 — Information security management systems",
        "ISO/IEC 29100:2011 — Security techniques — Privacy framework",
        "ISO 31000:2018 — Risk management — Guidelines",
        "ISO 8000-2:2022 — Data quality — Part 2: Vocabulary",
        "ISO/IEC TR 24027:2021 — Bias in AI systems and AI aided decision making",
        "NIST AI Risk Management Framework (AI RMF 1.0)",
    ]
)

# Connected Spanish prose, in the register the corpus actually uses.
_SUBSTANTIVE_PROSE = (
    "La organizacion deberia determinar y documentar los detalles sobre la "
    "adquisicion y seleccion de los datos que se utilizan en los sistemas de "
    "inteligencia artificial. Los detalles sobre la adquisicion de los datos "
    "pueden incluir las categorias de datos que son necesarias para los "
    "sistemas, la cantidad de datos que se requiere, y las fuentes de las que "
    "se obtienen los datos. La organizacion tambien deberia asegurarse de que "
    "los datos que se utilizan para desarrollar y para operar el sistema son "
    "adecuados para el proposito con el que se han recopilado."
)


def _chunk(**overrides) -> ChunkResult:
    """A ChunkResult with sensible defaults, overridable per test."""
    defaults = dict(
        chunk_id="c1",
        document_id="d1",
        tenant_id="global",
        content="contenido",
        chunk_index=0,
        similarity=0.5,
        section_ref="ANEXO D",
        is_reference_list=False,
    )
    defaults.update(overrides)
    return ChunkResult(**defaults)


# ------------------------------------------------------------
# 1. Detection — heading anchoring
# ------------------------------------------------------------
class TestReferenceHeadingDetection:
    @pytest.mark.parametrize(
        "heading",
        [
            "BIBLIOGRAFÍA",
            "BIBLIOGRAFIA",
            "Bibliography",
            "Referencias normativas",
            "Referencias bibliográficas",
            "References",
            "APPENDIX I. REFERENCES",
            "ANEXO D BIBLIOGRAFIA",
        ],
    )
    def test_standalone_headings_are_detected(self, heading):
        """
        Expected input:  a heading occupying its own line.
        Expected output: exactly one heading offset.
        """
        text = f"Texto previo.\n{heading}\nISO 31000:2018 — Risk management"
        assert len(find_reference_headings(text)) == 1

    @pytest.mark.parametrize(
        "sentence",
        [
            "Para las referencias con fecha, solo se aplica la edicion citada.",
            "Toda referencia en la reglamentacion emitida por el CONASSIF.",
            "Este documento contiene referencias cruzadas a otras politicas.",
            "The requirement references external documentation where relevant.",
        ],
    )
    def test_the_same_words_inside_prose_are_not_headings(self, sentence):
        """
        Expected input:  "referencias"/"references" used mid-sentence — the
                         dominant false positive in the Spanish regulatory
                         corpus, where it appears in running prose far more
                         often than as a heading.
        Expected output: no heading detected, because the word does not sit
                         on a line of its own.
        """
        text = f"Articulo 5\n{sentence}\nMas texto sustantivo aqui."
        assert find_reference_headings(text) == []

    def test_table_of_contents_entries_are_not_headings(self):
        """
        Expected input:  TOC lines, where the heading word is followed by dot
                         leaders or a page number on the same line.
        Expected output: no heading — the line does not END at the heading,
                         so the ^...$ anchor rejects it. Without this, every
                         document with a TOC would flag its own front matter.
        """
        toc = (
            "INDICE\n"
            "— Bibliografía 63\n"
            "Referencias normativas ................... 12\n"
        )
        assert find_reference_headings(toc) == []


# ------------------------------------------------------------
# 2. Detection — function-word ratio
# ------------------------------------------------------------
class TestFunctionWordRatio:
    def test_prose_scores_far_above_a_reference_list(self):
        """
        Expected input:  connected prose vs a list of standard titles.
        Expected output: prose ratio is at least double the list's. This gap
                         is what makes the threshold robust; it was measured
                         at 0.38 vs 0.18 across the real corpus.
        """
        prose_ratio = function_word_ratio(_SUBSTANTIVE_PROSE.split())
        list_ratio = function_word_ratio(_BIBLIOGRAPHY_ENTRIES.split())
        assert prose_ratio > 0.30
        assert list_ratio < 0.20
        assert prose_ratio > 2 * list_ratio

    def test_digits_only_input_is_not_prose(self):
        """
        Expected input:  a fragment with no alphabetic tokens (dot leaders,
                         page numbers) — common in front matter.
        Expected output: 0.0, i.e. "not prose", rather than a crash.
        """
        assert function_word_ratio(["....", "63", "12", "—"]) == 0.0


# ------------------------------------------------------------
# 3. Detection — end-to-end through the chunker
# ------------------------------------------------------------
class TestChunkerFlagsReferenceLists:
    def test_bibliography_at_end_of_document_is_flagged(self):
        """
        Expected input:  a document of substantive prose closing with a
                         BIBLIOGRAFIA section — the ISO 42001 Anexo D shape.
        Expected output: the trailing chunk(s) flagged, the prose chunks not.
        """
        text = (
            "ANEXO C\n"
            + _SUBSTANTIVE_PROSE
            + "\n"
            + _SUBSTANTIVE_PROSE
            + "\nBIBLIOGRAFÍA\n"
            + _BIBLIOGRAPHY_ENTRIES
            + "\n"
        )
        chunks = chunk_text_with_sections(text, chunk_size=120, chunk_overlap=20)

        assert any(c.is_reference_list for c in chunks), "bibliography not flagged"
        # Every flagged chunk must actually be bibliography, not prose.
        for chunk in chunks:
            if chunk.is_reference_list:
                assert "ISO" in chunk.content
                assert "La organizacion deberia determinar" not in chunk.content

    def test_bibliography_spanning_chunks_flags_the_continuation(self):
        """
        Expected input:  a bibliography long enough to span several chunks.
                         The continuation chunks contain NO heading — this is
                         exactly ISO 42001 idx=70, which a heading-only rule
                         misses entirely.
        Expected output: every chunk of the bibliography run is flagged, not
                         just the one carrying the heading.
        """
        text = (
            "ANEXO D\n"
            + _SUBSTANTIVE_PROSE
            + "\nBIBLIOGRAFÍA\n"
            + _BIBLIOGRAPHY_ENTRIES
            + "\n"
            + _BIBLIOGRAPHY_ENTRIES
            + "\n"
        )
        chunks = chunk_text_with_sections(text, chunk_size=60, chunk_overlap=10)
        flagged = [i for i, c in enumerate(chunks) if c.is_reference_list]

        assert len(flagged) >= 2, f"expected a run of flagged chunks, got {flagged}"
        # A continuation chunk is one with no heading text in it.
        continuations = [
            i for i in flagged if "BIBLIOGRAF" not in chunks[i].content.upper()
        ]
        assert continuations, "continuation chunk was not flagged"

    def test_prose_resuming_after_a_short_reference_block_ends_the_region(self):
        """
        Expected input:  a short References block followed by substantive
                         prose — the NIST AI RMF Playbook shape, which repeats
                         this structure 72 times.
        Expected output: the prose that follows is NOT flagged. Without the
                         leading-window check this leaked, demoting real
                         content across the whole document.
        """
        text = (
            "References\n"
            "ISO 31000:2018 — Risk management — Guidelines\n"
            "ISO/IEC 27001:2022 — Information security management systems\n"
            + _SUBSTANTIVE_PROSE
            + "\n"
            + _SUBSTANTIVE_PROSE
            + "\n"
            + _SUBSTANTIVE_PROSE
            + "\n"
        )
        chunks = chunk_text_with_sections(text, chunk_size=80, chunk_overlap=10)

        prose_chunks = [
            c for c in chunks if "La organizacion deberia determinar" in c.content
        ]
        assert prose_chunks, "fixture did not produce prose chunks"
        assert not any(c.is_reference_list for c in prose_chunks)

    def test_document_without_any_bibliography_flags_nothing(self):
        """
        Expected input:  ordinary normative text, no reference section.
        Expected output: no chunk flagged — the default must stay off so
                         nothing is demoted without cause.
        """
        text = "Articulo 1\n" + (_SUBSTANTIVE_PROSE + "\n") * 4
        chunks = chunk_text_with_sections(text, chunk_size=100, chunk_overlap=20)
        assert chunks
        assert not any(c.is_reference_list for c in chunks)

    def test_flag_defaults_to_false_and_section_ref_still_works(self):
        """
        Expected input:  a plain sectioned document.
        Expected output: section attribution is unchanged by this feature —
                         guards the NamedTuple field addition.
        """
        text = "Seccion 7.2\n" + _SUBSTANTIVE_PROSE
        chunks = chunk_text_with_sections(text, chunk_size=200, chunk_overlap=20)
        assert chunks[0].section_ref == "Seccion 7.2"
        assert chunks[0].is_reference_list is False


# ------------------------------------------------------------
# 4. Demotion — rerank_chunks()
# ------------------------------------------------------------
class TestReferenceListDemotion:
    def test_reference_chunk_loses_to_substantive_chunk_it_used_to_outrank(self):
        """
        Expected input:  a bibliography chunk with HIGHER similarity than a
                         substantive one — the exact inversion measured on
                         Case A, where Anexo D bibliography beat the Anexo C
                         clause that answered the question.
        Expected output: the substantive chunk ranks first.
        """
        biblio = _chunk(
            chunk_id="biblio",
            content=_BIBLIOGRAPHY_ENTRIES,
            similarity=0.72,
            is_reference_list=True,
        )
        substantive = _chunk(
            chunk_id="substantive",
            content=_SUBSTANTIVE_PROSE,
            similarity=0.55,
            is_reference_list=False,
        )

        ranked = rerank_chunks("requisitos de transparencia y explicabilidad",
                              [biblio, substantive])

        assert ranked[0].chunk_id == "substantive"

    def test_reference_chunk_is_demoted_not_removed(self):
        """
        Expected input:  a flagged chunk alongside a substantive one.
        Expected output: BOTH are still returned. The design decision is
                         demotion, never exclusion — the content stays
                         retrievable.
        """
        biblio = _chunk(chunk_id="biblio", similarity=0.9, is_reference_list=True)
        other = _chunk(chunk_id="other", similarity=0.5)

        ranked = rerank_chunks("politica de gestion de riesgos", [biblio, other])

        assert {c.chunk_id for c in ranked} == {"biblio", "other"}

    @pytest.mark.parametrize(
        "query",
        [
            "¿Qué normas ISO cita el Anexo D de ISO 42001?",
            "¿Cuáles son las normas citadas en el documento?",
            "Muéstrame la bibliografía de la norma",
            "referencias normativas del documento",
            "What standards are cited in this document?",
            "list of standards referenced",
        ],
    )
    def test_penalty_is_waived_for_reference_seeking_queries(self, query):
        """
        Expected input:  a question whose correct answer IS the bibliography.
        Expected output: the flagged chunk keeps its lead. This is the
                         control case — the penalty must reduce priority
                         without making the content unreachable.
        """
        biblio = _chunk(
            chunk_id="biblio",
            content=_BIBLIOGRAPHY_ENTRIES,
            similarity=0.72,
            is_reference_list=True,
        )
        substantive = _chunk(
            chunk_id="substantive",
            content=_SUBSTANTIVE_PROSE,
            similarity=0.55,
        )

        ranked = rerank_chunks(query, [biblio, substantive])

        assert ranked[0].chunk_id == "biblio", (
            f"reference-seeking query {query!r} should surface the bibliography"
        )

    def test_ordinary_topic_query_does_not_waive_the_penalty(self):
        """
        Expected input:  a question that merely mentions a standard by name.
        Expected output: penalty still applied. The waiver vocabulary has to
                         stay narrow — matching on "norma" alone would waive
                         the penalty for most of this corpus's questions.
        """
        biblio = _chunk(
            chunk_id="biblio",
            content=_BIBLIOGRAPHY_ENTRIES,
            similarity=0.72,
            is_reference_list=True,
        )
        substantive = _chunk(
            chunk_id="substantive",
            content=_SUBSTANTIVE_PROSE,
            similarity=0.55,
        )

        ranked = rerank_chunks(
            "¿Qué exige la norma ISO 42001 sobre la calidad de los datos?",
            [biblio, substantive],
        )

        assert ranked[0].chunk_id == "substantive"

    def test_penalty_of_one_is_a_no_op(self):
        """
        Expected input:  reference_penalty=1.0 (the documented disable value).
        Expected output: ordering identical to an unflagged run, so operators
                         can turn the behaviour off cleanly.
        """
        biblio = _chunk(chunk_id="biblio", similarity=0.9, is_reference_list=True)
        other = _chunk(chunk_id="other", similarity=0.5)

        ranked = rerank_chunks(
            "gestion de riesgos", [biblio, other], reference_penalty=1.0
        )

        assert ranked[0].chunk_id == "biblio"

    @pytest.mark.parametrize(
        "query,expected_top",
        [
            ("¿Qué exige ISO 42001 sobre la calidad de los datos?", "substantive"),
            ("¿Qué normas ISO cita el Anexo D?", "biblio"),
        ],
    )
    def test_penalty_works_through_the_legacy_argument_order(
        self, query, expected_top
    ):
        """
        Expected input:  rerank_chunks(chunks, query) — the REVERSED order,
                         which is how RetrievalService actually calls it
                         ("rerank_chunks(unique_chunks, query)").
        Expected output: identical behaviour to the keyword form. The
                         back-compat shim swaps the arguments, and the
                         reference logic reads the query AFTER that swap —
                         if it ever read it before, production would
                         silently classify the chunk list as the query and
                         never detect a reference-seeking question.
        """
        biblio = _chunk(
            chunk_id="biblio",
            content=_BIBLIOGRAPHY_ENTRIES,
            similarity=0.72,
            is_reference_list=True,
        )
        substantive = _chunk(
            chunk_id="substantive",
            content=_SUBSTANTIVE_PROSE,
            similarity=0.55,
        )

        ranked = rerank_chunks([biblio, substantive], query)

        assert ranked[0].chunk_id == expected_top

    def test_unflagged_chunks_are_scored_exactly_as_before(self):
        """
        Expected input:  a chunk list with no flags set (every row in a
                         database that has not been reprocessed yet).
        Expected output: pure similarity ordering, unchanged — the feature
                         is inert until the corpus is reclassified.
        """
        chunks = [
            _chunk(chunk_id="a", content=_SUBSTANTIVE_PROSE, similarity=0.4),
            _chunk(chunk_id="b", content=_SUBSTANTIVE_PROSE, similarity=0.8),
            _chunk(chunk_id="c", content=_SUBSTANTIVE_PROSE, similarity=0.6),
        ]

        ranked = rerank_chunks("calidad de los datos", chunks)

        assert [c.chunk_id for c in ranked] == ["b", "c", "a"]
