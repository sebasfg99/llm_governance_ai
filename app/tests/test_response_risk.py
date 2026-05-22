# ============================================================
# app/tests/test_response_risk.py
#
# Validates compute_risk() across the three risk levels.
# ============================================================

from __future__ import annotations

from app.domain.response_risk.scorer import (
    compute_risk,
    HIGH_CONTEXT_ADHERENCE_FLOOR,
    MEDIUM_CONTEXT_ADHERENCE_FLOOR,
)


class TestHighRisk:
    def test_hallucination_is_high(self):
        risk = compute_risk(
            evaluation={"hallucination": True, "context_adherence_score": 0.95},
            guardrails={"violations": []},
            metadata={"chunks": [{"similarity": 0.8}]},
        )
        assert risk["risk_level"] == "high"
        assert any("hallucination" in r for r in risk["risk_reasons"])

    def test_low_context_adherence_is_high(self):
        risk = compute_risk(
            evaluation={"hallucination": False, "context_adherence_score": 0.4},
            guardrails={"violations": []},
            metadata={"chunks": [{"similarity": 0.8}]},
        )
        assert risk["risk_level"] == "high"

    def test_pii_exposure_is_high(self):
        risk = compute_risk(
            evaluation={"hallucination": False, "context_adherence_score": 0.95},
            guardrails={"violations": [], "pii_findings": [{"type": "EMAIL"}]},
            metadata={"chunks": [{"similarity": 0.8}]},
        )
        assert risk["risk_level"] == "high"
        assert any("pii_exposure" in r for r in risk["risk_reasons"])

    def test_legal_without_disclaimer_is_high(self):
        risk = compute_risk(
            evaluation={"hallucination": False, "context_adherence_score": 0.95},
            guardrails={
                "violations": [],
                "requires_disclaimer": True,
                "safe_response": "respuesta sin disclaimer",
            },
            metadata={"chunks": [{"similarity": 0.8}]},
        )
        assert risk["risk_level"] == "high"
        assert any("disclaimer" in r for r in risk["risk_reasons"])


class TestMediumRisk:
    def test_partial_adherence_is_medium(self):
        adherence = (HIGH_CONTEXT_ADHERENCE_FLOOR + MEDIUM_CONTEXT_ADHERENCE_FLOOR) / 2
        risk = compute_risk(
            evaluation={
                "hallucination": False,
                "context_adherence_score": adherence,
                "groundedness_score": 0.5,
                "citation_score": 0.5,
                "completeness_score": 0.7,
                "instruction_adherence_score": 1.0,
            },
            guardrails={"violations": []},
            metadata={"chunks": [{"similarity": 0.7}]},
        )
        assert risk["risk_level"] == "medium"

    def test_incomplete_answer_is_medium(self):
        risk = compute_risk(
            evaluation={
                "hallucination": False,
                "context_adherence_score": 1.0,
                "groundedness_score": 0.6,
                "citation_score": 0.5,
                "completeness_score": 0.3,  # < 0.5
                "instruction_adherence_score": 1.0,
            },
            guardrails={"violations": []},
            metadata={"chunks": [{"similarity": 0.8}]},
        )
        assert risk["risk_level"] == "medium"

    def test_should_have_said_idk_is_medium(self):
        risk = compute_risk(
            evaluation={
                "hallucination": False,
                "context_adherence_score": 1.0,
                "groundedness_score": 0.6,
                "citation_score": 0.5,
                "completeness_score": 0.7,
                "instruction_adherence_score": 1.0,
                "should_have_said_idk": True,
            },
            guardrails={"violations": []},
            metadata={"chunks": [{"similarity": 0.8}]},
        )
        assert risk["risk_level"] == "medium"


class TestLowRisk:
    def test_perfect_signals_is_low(self):
        risk = compute_risk(
            evaluation={
                "hallucination": False,
                "context_adherence_score": 1.0,
                "groundedness_score": 0.6,
                "citation_score": 1.0,
                "completeness_score": 1.0,
                "instruction_adherence_score": 1.0,
                "should_have_said_idk": False,
            },
            guardrails={"violations": []},
            metadata={"chunks": [{"similarity": 0.85}, {"similarity": 0.75}]},
        )
        assert risk["risk_level"] == "low"
        assert "all_signals_clean" in risk["risk_reasons"]


class TestRiskOutputShape:
    def test_returns_required_keys(self):
        risk = compute_risk(None, None, None)
        for key in (
            "risk_level",
            "confidence_score",
            "risk_reasons",
            "retrieval_confidence",
        ):
            assert key in risk

    def test_confidence_in_unit_range(self):
        risk = compute_risk(
            evaluation={
                "context_adherence_score": 0.9,
                "groundedness_score": 0.6,
                "instruction_adherence_score": 1.0,
            },
            guardrails={"violations": []},
            metadata={"chunks": [{"similarity": 0.8}, {"similarity": 0.7}]},
        )
        assert 0.0 <= risk["confidence_score"] <= 1.0
        assert 0.0 <= risk["retrieval_confidence"] <= 1.0

    def test_high_risk_has_low_confidence(self):
        risk_high = compute_risk(
            evaluation={"hallucination": True, "context_adherence_score": 0.4},
            guardrails={"violations": []},
            metadata={"chunks": [{"similarity": 0.5}]},
        )
        risk_low = compute_risk(
            evaluation={
                "hallucination": False,
                "context_adherence_score": 1.0,
                "groundedness_score": 0.6,
                "citation_score": 1.0,
                "completeness_score": 1.0,
                "instruction_adherence_score": 1.0,
            },
            guardrails={"violations": []},
            metadata={"chunks": [{"similarity": 0.9}, {"similarity": 0.85}]},
        )
        assert risk_high["confidence_score"] < risk_low["confidence_score"]
