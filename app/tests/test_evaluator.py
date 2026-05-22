# ============================================================
# app/tests/test_evaluator.py — Eight-metric evaluation tests
# ============================================================

from __future__ import annotations

import pytest

from app.domain.evals.evaluator import evaluate_response
from app.domain.evals import graders


# ============================================================
# Public facade
# ============================================================
class TestEvaluateResponseShape:
    """Output shape contract."""

    def test_returns_all_required_keys(self, sample_chunks, mock_judge_clean):
        result = evaluate_response(
            query="\u00bfQu\u00e9 dice ISO 42001 sobre el comit\u00e9?",
            chunks=sample_chunks,
            response="ISO 42001 [1] establece que el sistema de gesti\u00f3n debe identificar riesgos de equidad.",
        )
        for key in (
            "retrieval_precision",
            "groundedness_score",
            "context_adherence_score",
            "citation_score",
            "completeness_score",
            "instruction_adherence_score",
            "hallucination",
            "should_have_said_idk",
        ):
            assert key in result, f"Missing key {key} in evaluator output"

    def test_scores_are_floats_in_unit_range(self, sample_chunks, mock_judge_clean):
        result = evaluate_response(
            query="ISO 42001",
            chunks=sample_chunks,
            response="ISO 42001 dice que se deben identificar riesgos.",
        )
        for key in (
            "retrieval_precision",
            "groundedness_score",
            "context_adherence_score",
            "citation_score",
            "completeness_score",
            "instruction_adherence_score",
        ):
            v = result[key]
            assert isinstance(v, float), f"{key} should be float, got {type(v)}"
            assert 0.0 <= v <= 1.0, f"{key}={v} out of [0,1]"


# ============================================================
# Hallucination detection
# ============================================================
class TestHallucinationDetection:
    def test_hallucination_flagged_when_judge_says_so(
        self, sample_chunks, mock_judge_hallucinated
    ):
        result = evaluate_response(
            query="\u00bfQui\u00e9n certific\u00f3 el sistema?",
            chunks=sample_chunks,
            response="El sistema fue certificado por Naciones Unidas y ISO 42001.",
        )
        assert result["hallucination"] is True
        assert result["claims_total"] == 2
        assert result["claims_supported"] == 1
        # context_adherence = 1/2 = 0.5
        assert result["context_adherence_score"] == 0.5

    def test_clean_response_not_flagged(self, sample_chunks, mock_judge_clean):
        result = evaluate_response(
            query="ISO 42001",
            chunks=sample_chunks,
            response="ISO 42001 establece que se deben identificar riesgos.",
        )
        assert result["hallucination"] is False
        assert result["context_adherence_score"] == 1.0


# ============================================================
# IDK behavior
# ============================================================
class TestIDKBehavior:
    def test_should_have_said_idk_flagged(self, sample_chunks, mock_judge_idk_failure):
        result = evaluate_response(
            query="\u00bfCu\u00e1l es el horario de atenci\u00f3n del banco?",
            chunks=sample_chunks,  # irrelevant chunks
            response="El horario es de 9 a 5pm.",
        )
        assert result["should_have_said_idk"] is True

    def test_no_chunks_idk_response_is_complete(self):
        """Correctly abstaining with no chunks should NOT lower completeness."""
        result = evaluate_response(
            query="\u00bfPregunta sin contexto?",
            chunks=[],
            response="No tengo informaci\u00f3n suficiente para responder.",
            use_llm_judge=False,
        )
        assert result["completeness_score"] == 1.0


# ============================================================
# Citation scoring
# ============================================================
class TestCitationScore:
    def test_inline_numbered_citations_resolve(self, sample_chunks):
        score = graders.citation_score(
            response_text="Los riesgos est\u00e1n descritos [1] y tambi\u00e9n [2].",
            citations=None,
            chunks=sample_chunks,
        )
        # presence=1.0, traceability=1.0 (both refs valid since 3 chunks)
        assert score == 1.0

    def test_invalid_inline_reference_lowers_score(self, sample_chunks):
        score = graders.citation_score(
            response_text="Algo dijo [99].",  # invalid index
            citations=None,
            chunks=sample_chunks,
        )
        assert score < 1.0
        assert score >= 0.0

    def test_no_citation_yields_zero(self, sample_chunks):
        score = graders.citation_score(
            response_text="Una respuesta sin ninguna cita ni atribuci\u00f3n.",
            citations=None,
            chunks=sample_chunks,
        )
        assert score == 0.0

    def test_explicit_citations_traceable(self, sample_chunks):
        cits = [{"chunk_id": sample_chunks[0].chunk_id}]
        score = graders.citation_score(
            response_text="Respuesta sin n\u00fameros pero con citation list.",
            citations=cits,
            chunks=sample_chunks,
        )
        assert score == 1.0


# ============================================================
# Retrieval precision
# ============================================================
class TestRetrievalPrecision:
    def test_high_precision(self, sample_chunks):
        # 2 of 3 chunks above 0.45 threshold
        score = graders.retrieval_precision(sample_chunks, similarity_threshold=0.45)
        assert score == round(2 / 3, 4)

    def test_no_chunks_zero(self):
        assert graders.retrieval_precision([]) == 0.0

    def test_all_relevant(self, sample_chunks):
        score = graders.retrieval_precision(
            sample_chunks, similarity_threshold=0.0
        )
        assert score == 1.0


# ============================================================
# Instruction adherence
# ============================================================
class TestInstructionAdherence:
    def test_no_instructions_perfect(self):
        score = graders.instruction_adherence_score(
            "Hola, esta es una respuesta normal.", instructions=None
        )
        assert score == 1.0

    def test_structured_invalid_json_penalised(self):
        score = graders.instruction_adherence_score(
            "este no es JSON",
            instructions="Respond with valid JSON.",
            response_mode="structured",
        )
        assert score < 1.0

    def test_structured_valid_json_full_score(self):
        score = graders.instruction_adherence_score(
            '{"answer": "ok"}',
            instructions="Respond with valid JSON.",
            response_mode="structured",
        )
        assert score == 1.0

    def test_spanish_request_english_response_penalised(self):
        score = graders.instruction_adherence_score(
            "the answer is the policy of the company is to be honest in the long run",
            instructions="Responde en espa\u00f1ol siempre.",
        )
        # English response when Spanish requested → penalised
        assert score < 1.0


# ============================================================
# Completeness
# ============================================================
class TestCompleteness:
    def test_too_short_low_score(self):
        score = graders.completeness_score(
            query="\u00bfQu\u00e9 es ISO 42001?",
            response_text="No.",
            chunks=[],
        )
        assert score < 0.5

    def test_keyword_coverage_high(self):
        score = graders.completeness_score(
            query="\u00bfQu\u00e9 dice ISO 42001 sobre comit\u00e9 de gobernanza?",
            response_text=(
                "ISO 42001 establece que el comit\u00e9 de gobernanza debe revisar "
                "los riesgos del sistema y aprobar las pol\u00edticas relevantes."
            ),
            chunks=[],
        )
        assert score > 0.5


# ============================================================
# Empty / edge cases
# ============================================================
class TestEvaluatorEdgeCases:
    def test_empty_response_returns_zero_grounding(self, sample_chunks):
        result = evaluate_response(
            query="x",
            chunks=sample_chunks,
            response="",
            use_llm_judge=False,
        )
        assert result["groundedness_score"] == 0.0
        assert result["completeness_score"] == 0.0

    def test_use_llm_judge_false_skips_judge(self, sample_chunks):
        # Should not call any external API even though we did not patch it
        result = evaluate_response(
            query="ISO 42001",
            chunks=sample_chunks,
            response="ISO 42001 establece las reglas de IA.",
            use_llm_judge=False,
        )
        assert result["judge_ok"] is False
        # context_adherence falls back to grounding when judge skipped
        assert isinstance(result["context_adherence_score"], float)
