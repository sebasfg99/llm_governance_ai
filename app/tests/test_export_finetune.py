# ============================================================
# app/tests/test_export_finetune.py
#
# Validates the curated export logic in scripts/export_finetune_data.py.
# ============================================================

from __future__ import annotations

import json
import os
from unittest.mock import MagicMock, patch

import pytest

from scripts.export_finetune_data import (
    _passes_governance,
    export_finetune_data,
)


class TestPassesGovernance:
    def test_low_risk_clean_passes(self):
        assert _passes_governance(
            eval_row={
                "hallucination": False,
                "context_adherence_score": 0.95,
                "groundedness_score": 0.85,
            },
            risk_row={"risk_level": "low"},
        )

    def test_high_risk_blocks(self):
        assert not _passes_governance(
            eval_row={
                "hallucination": False,
                "context_adherence_score": 0.95,
                "groundedness_score": 0.85,
            },
            risk_row={"risk_level": "high"},
        )

    def test_hallucination_blocks(self):
        assert not _passes_governance(
            eval_row={
                "hallucination": True,
                "context_adherence_score": 0.95,
                "groundedness_score": 0.85,
            },
            risk_row={"risk_level": "low"},
        )

    def test_low_adherence_blocks(self):
        assert not _passes_governance(
            eval_row={
                "hallucination": False,
                "context_adherence_score": 0.85,  # < 0.9
                "groundedness_score": 0.85,
            },
            risk_row={"risk_level": "low"},
        )

    def test_low_groundedness_blocks(self):
        assert not _passes_governance(
            eval_row={
                "hallucination": False,
                "context_adherence_score": 0.95,
                "groundedness_score": 0.5,  # < 0.8
            },
            risk_row={"risk_level": "low"},
        )

    def test_legacy_fallback_to_passed(self):
        """When no governance columns exist, legacy `passed` is used."""
        assert _passes_governance(eval_row={"passed": True}, risk_row=None)
        assert not _passes_governance(eval_row={"passed": False}, risk_row=None)


class TestExportFinetuneIntegration:
    """
    End-to-end: stub the Supabase client to return a curated mix and
    verify that ONLY the low-risk row reaches the JSONL.
    """

    @patch("scripts.export_finetune_data.get_supabase_client")
    def test_only_low_risk_no_hallucination_exported(self, mock_get_db, tmp_path):
        # ---- Build fake DB ----
        db = MagicMock()

        good_id = "00000000-0000-0000-0000-000000000001"
        bad_hallu_id = "00000000-0000-0000-0000-000000000002"
        bad_risk_id = "00000000-0000-0000-0000-000000000003"

        traces = [
            {
                "id": good_id,
                "user_query": "\u00bfQu\u00e9 dice ISO 42001?",
                "response_text": "ISO 42001 establece el sistema de gesti\u00f3n.",
                "use_case": "chatbot",
                "model_provider": "openai",
                "model_name": "gpt-4o-mini",
                "client_id": "acme",
                "tenant_id": "acme",
            },
            {
                "id": bad_hallu_id,
                "user_query": "\u00bfQui\u00e9n cre\u00f3 ISO 42001?",
                "response_text": "Lo cre\u00f3 Naciones Unidas en 1953.",
                "use_case": "chatbot",
                "model_provider": "openai",
                "model_name": "gpt-4o-mini",
                "client_id": "acme",
                "tenant_id": "acme",
            },
            {
                "id": bad_risk_id,
                "user_query": "?",
                "response_text": "Algo.",
                "use_case": "chatbot",
                "model_provider": "openai",
                "model_name": "gpt-4o-mini",
                "client_id": "acme",
                "tenant_id": "acme",
            },
        ]

        evaluations = [
            {
                "trace_id": good_id,
                "hallucination": False,
                "context_adherence_score": 0.97,
                "groundedness_score": 0.85,
            },
            {
                "trace_id": bad_hallu_id,
                "hallucination": True,
                "context_adherence_score": 0.5,
                "groundedness_score": 0.4,
            },
            {
                "trace_id": bad_risk_id,
                "hallucination": False,
                "context_adherence_score": 0.95,
                "groundedness_score": 0.85,
            },
        ]

        risks = [
            {"trace_id": good_id, "risk_level": "low"},
            {"trace_id": bad_hallu_id, "risk_level": "high"},
            {"trace_id": bad_risk_id, "risk_level": "high"},
        ]

        # Configure the chainable mock
        def _table(name: str) -> MagicMock:
            t = MagicMock()
            # Make every chained method return self
            for chain in (
                "select",
                "order",
                "limit",
                "eq",
                "in_",
                "gte",
                "lt",
            ):
                getattr(t, chain).return_value = t

            response = MagicMock()
            if name == "traces":
                response.data = traces
            elif name == "evaluations":
                response.data = evaluations
            elif name == "response_risks":
                response.data = risks
            elif name == "feedback":
                response.data = []
            else:
                response.data = []
            t.execute.return_value = response
            return t

        db.table.side_effect = _table
        mock_get_db.return_value = db

        # ---- Run ----
        out = tmp_path / "fine.jsonl"
        stats = export_finetune_data(
            use_case="chatbot",
            min_rating=4,
            output_path=str(out),
            limit=10,
        )

        # ---- Assert ----
        assert out.exists()
        lines = [l for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        assert len(lines) == 1, f"Expected 1 example, got {len(lines)}"
        entry = json.loads(lines[0])
        assert entry["metadata"]["risk_level"] == "low"
        assert entry["metadata"]["trace_id"] == good_id
        assert stats["examples_written"] == 1
        assert stats["passed_governance"] == 1
