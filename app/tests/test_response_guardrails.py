# ============================================================
# app/tests/test_response_guardrails.py
#
# Validates app/domain/llm/guardrails.validate_response_guardrails
# and app/domain/llm/pii_detector.
# ============================================================

from __future__ import annotations

import pytest

from app.domain.llm.guardrails import validate_response_guardrails, STRICT_CONTEXT_THRESHOLD
from app.domain.llm import pii_detector


# ============================================================
# client_id is mandatory
# ============================================================
class TestClientIdMandatory:
    def test_missing_client_id_blocks(self):
        result = validate_response_guardrails(
            query="x",
            response="any response",
            chunks=[],
            client_id=None,
            user_id="u1",
        )
        assert result["allowed"] is False
        assert result["safe_response"] == ""
        codes = [v["code"] for v in result["violations"]]
        assert "missing_client_id" in codes

    def test_empty_client_id_blocks(self):
        result = validate_response_guardrails(
            query="x",
            response="any response",
            chunks=[],
            client_id="   ",
            user_id="u1",
        )
        assert result["allowed"] is False


# ============================================================
# PII redaction
# ============================================================
class TestPIIRedaction:
    def test_email_redacted(self):
        text = "Contacta a alice@example.com para m\u00e1s informaci\u00f3n."
        result = validate_response_guardrails(
            query="?",
            response=text,
            chunks=[],
            client_id="acme",
        )
        assert "alice@example.com" not in result["safe_response"]
        assert "[REDACTED:EMAIL]" in result["safe_response"]
        assert any(v["code"] == "pii_leak" for v in result["violations"])

    def test_phone_redacted(self):
        text = "Llama al +506 8888-9999 cualquier d\u00eda."
        findings = pii_detector.detect_pii(text)
        assert findings, "Detector should find a phone"
        red = pii_detector.redact_pii(text, findings)
        assert "8888-9999" not in red

    def test_pii_alone_does_not_block(self):
        """The redacted text is still delivered; PII doesn't fully block."""
        result = validate_response_guardrails(
            query="?",
            response="Mi correo es x@y.com.",
            chunks=[],
            client_id="acme",
        )
        assert result["allowed"] is True
        assert "[REDACTED:EMAIL]" in result["safe_response"]


# ============================================================
# Hallucination block
# ============================================================
class TestHallucinationBlock:
    def test_hallucination_rewrites_response(self):
        result = validate_response_guardrails(
            query="?",
            response="El BCE certific\u00f3 esto en 1999.",
            chunks=[],
            client_id="acme",
            evaluation={
                "hallucination": True,
                "context_adherence_score": 0.4,
                "should_have_said_idk": False,
            },
        )
        assert result["allowed"] is False
        assert "1999" not in result["safe_response"]
        # The response is rewritten to a safe abstention
        assert "no se pueden verificar" in result["safe_response"].lower() or \
               "afirmaciones que" in result["safe_response"].lower()


# ============================================================
# Strict context adherence
# ============================================================
class TestStrictContextAdherence:
    def test_legal_use_case_low_adherence_blocks(self):
        result = validate_response_guardrails(
            query="?",
            response="Cualquier respuesta legal aqu\u00ed.",
            chunks=[],
            client_id="acme",
            evaluation={
                "hallucination": False,
                "context_adherence_score": 0.6,
                "should_have_said_idk": False,
            },
            use_case="risk_analysis",
        )
        assert result["allowed"] is False
        codes = [v["code"] for v in result["violations"]]
        assert "low_context_adherence" in codes

    def test_chatbot_use_case_low_adherence_does_not_block(self):
        result = validate_response_guardrails(
            query="?",
            response="Cualquier respuesta.",
            chunks=[],
            client_id="acme",
            evaluation={
                "hallucination": False,
                "context_adherence_score": 0.6,
                "should_have_said_idk": False,
            },
            use_case="chatbot",
            strict_context=False,
        )
        assert result["allowed"] is True

    def test_high_adherence_passes_strict(self):
        result = validate_response_guardrails(
            query="?",
            response="Texto que cumple.",
            chunks=[],
            client_id="acme",
            evaluation={
                "hallucination": False,
                "context_adherence_score": 0.95,
                "should_have_said_idk": False,
            },
            use_case="risk_analysis",
        )
        assert result["allowed"] is True


# ============================================================
# Disclaimer
# ============================================================
class TestDisclaimer:
    def test_legal_use_case_appends_disclaimer(self):
        result = validate_response_guardrails(
            query="?",
            response="Texto legal de ejemplo.",
            chunks=[],
            client_id="acme",
            evaluation={
                "hallucination": False,
                "context_adherence_score": 1.0,
                "should_have_said_idk": False,
            },
            use_case="policy_generation",
        )
        assert result["requires_disclaimer"] is True
        assert "Disclaimer" in result["safe_response"]

    def test_chatbot_no_disclaimer(self):
        result = validate_response_guardrails(
            query="?",
            response="Hola, c\u00f3mo te ayudo.",
            chunks=[],
            client_id="acme",
            evaluation={
                "hallucination": False,
                "context_adherence_score": 1.0,
                "should_have_said_idk": False,
            },
            use_case="chatbot",
        )
        assert result["requires_disclaimer"] is False
        assert "Disclaimer" not in result["safe_response"]

    def test_disclaimer_only_added_once(self):
        """Calling guardrails twice should not duplicate the disclaimer."""
        first = validate_response_guardrails(
            query="?",
            response="Texto.",
            chunks=[],
            client_id="acme",
            evaluation={
                "hallucination": False,
                "context_adherence_score": 1.0,
                "should_have_said_idk": False,
            },
            use_case="policy_generation",
        )
        second = validate_response_guardrails(
            query="?",
            response=first["safe_response"],
            chunks=[],
            client_id="acme",
            evaluation={
                "hallucination": False,
                "context_adherence_score": 1.0,
                "should_have_said_idk": False,
            },
            use_case="policy_generation",
        )
        # Sentinel emoji "\u26a0\ufe0f" should appear exactly once
        assert second["safe_response"].count("\u26a0\ufe0f") == 1


# ============================================================
# IDK guard
# ============================================================
class TestIDKGuard:
    def test_should_have_abstained_rewrites(self):
        result = validate_response_guardrails(
            query="?",
            response="Una respuesta inventada porque el modelo no se abstuvo.",
            chunks=[],
            client_id="acme",
            evaluation={
                "hallucination": False,
                "context_adherence_score": 0.95,
                "should_have_said_idk": True,
            },
            use_case="chatbot",
        )
        codes = [v["code"] for v in result["violations"]]
        assert "should_have_abstained" in codes
        assert "informaci\u00f3n suficiente" in result["safe_response"]
