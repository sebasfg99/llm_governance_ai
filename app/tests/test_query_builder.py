# ============================================================
# app/tests/test_query_builder.py — Unit tests for
# app/domain/retrieval/query_builder.py
#
# All functions are pure (no I/O), so tests need no mocking.
# ============================================================

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.domain.retrieval.query_builder import (
    build_retrieval_query,
    extract_user_question,
    infer_framework_filter,
    split_top_k_across_frameworks,
)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _make_project(
    name: str = "Motor de Preaprobación",
    ai_type: str | None = "classification",
    data_used: list[str] | None = None,
    industry: str | None = "banking",
    autonomy_level: str | None = "human_in_loop",
    existing_vendor: str | None = None,
    project_description: str | None = None,
) -> MagicMock:
    """Build a minimal Project mock with a ProjectProfile."""
    profile = MagicMock()
    profile.ai_type = ai_type
    profile.data_used = data_used or ["pii", "pci"]
    profile.industry = industry
    profile.autonomy_level = autonomy_level
    profile.existing_vendor = existing_vendor
    profile.project_description = project_description

    project = MagicMock()
    project.name = name
    project.project_profile = profile
    return project


# ==================================================================
# infer_framework_filter
# ==================================================================


class TestInferFrameworkFilter:
    def test_iso_and_nist_detected(self):
        """Both ISO 42001 and NIST mentions resolve to their canonical tags."""
        result = infer_framework_filter(
            "¿Qué evidencias debo generar según ISO 42001 y NIST?",
            explicit=None,
        )
        assert result is not None
        assert "iso_42001" in result
        assert "nist_airmf" in result

    def test_explicit_filter_respected(self):
        """An explicit list is returned unchanged — no inference is run."""
        result = infer_framework_filter(
            "¿Qué evidencias debo generar según ISO 42001 y NIST?",
            explicit=["pci_dss"],
        )
        assert result == ["pci_dss"]

    def test_no_framework_returns_none(self):
        """Query with no framework keywords returns None (no filter applied)."""
        result = infer_framework_filter(
            "¿Cómo puedo mejorar mi sistema de IA?",
            explicit=None,
        )
        assert result is None

    def test_eu_ai_act_detected(self):
        result = infer_framework_filter("Reqs under EU AI Act", explicit=None)
        assert result == ["eu_ai_act"]

    def test_owasp_detected(self):
        result = infer_framework_filter("OWASP LLM Top 10 controls", explicit=None)
        assert "owasp_llm_top10" in result

    def test_gdpr_detected_case_insensitive(self):
        result = infer_framework_filter("GDPR compliance checklist", explicit=None)
        assert result == ["gdpr"]

    def test_pci_dss_detected(self):
        result = infer_framework_filter("PCI DSS v4 requirements", explicit=None)
        assert "pci_dss" in result

    def test_empty_explicit_list_triggers_inference(self):
        """An empty list (falsy) falls through to inference."""
        result = infer_framework_filter("ISO 42001 controls", explicit=[])
        assert result is not None
        assert "iso_42001" in result

    def test_deduplication(self):
        """Duplicate framework mentions produce a single tag."""
        result = infer_framework_filter(
            "ISO 42001 and ISO/IEC 42001 and iso42001",
            explicit=None,
        )
        assert result is not None
        assert result.count("iso_42001") == 1


# ==================================================================
# extract_user_question
# ==================================================================


class TestExtractUserQuestion:
    def test_short_query_returned_unchanged(self):
        """A single-line query is returned as-is (stripped)."""
        q = "¿Qué evidencias debo generar según ISO 42001 y NIST?"
        assert extract_user_question(q) == q

    def test_long_form_last_line_question_extracted(self):
        """
        A form-fill query where the last non-empty line ends with '?'
        should return only that line.
        """
        long_query = (
            "Nombre del Proyecto: Motor de Preaprobación\n"
            "Tipo de IA: clasificación\n"
            "Datos usados: historial crediticio, PII\n"
            "Sector: banca\n"
            "\n"
            "¿Qué evidencias debo generar según ISO 42001 y NIST?"
        )
        result = extract_user_question(long_query)
        assert result == "¿Qué evidencias debo generar según ISO 42001 y NIST?"

    def test_inverted_question_mark_line_extracted(self):
        """Last line starting with '¿' is extracted even without trailing '?'."""
        q = (
            "Descripción del sistema de IA\n"
            "Sector financiero\n"
            "¿Qué controles aplican?"
        )
        result = extract_user_question(q)
        assert result == "¿Qué controles aplican?"

    def test_multiline_no_question_returns_full(self):
        """Multi-line text without a question marker returns the full stripped query."""
        q = "Describe the system.\nIt processes credit data.\nList the controls."
        result = extract_user_question(q)
        assert result == q.strip()

    def test_empty_string_returns_empty(self):
        assert extract_user_question("") == ""

    def test_whitespace_only_returns_empty(self):
        assert extract_user_question("   \n  ") == ""

    def test_last_inverted_question_mark_wins_over_earlier(self):
        """When multiple '¿' lines exist, the LAST one is returned."""
        q = (
            "¿Primera pregunta?\n"
            "Texto intermedio\n"
            "¿Segunda pregunta?"
        )
        result = extract_user_question(q)
        assert result == "¿Segunda pregunta?"


# ==================================================================
# build_retrieval_query
# ==================================================================


class TestBuildRetrievalQuery:
    def test_without_project_returns_question(self):
        """Without a project the output is simply the user question."""
        q = "¿Qué evidencias debo generar?"
        result = build_retrieval_query(user_question=q, project=None)
        assert result == q

    def test_with_project_includes_profile_fields(self):
        """Project fields are appended after the question."""
        project = _make_project(
            name="Motor de Preaprobación",
            ai_type="classification",
            industry="banking",
            existing_vendor="buró de crédito",
        )
        result = build_retrieval_query(
            user_question="¿Qué evidencias debo generar?",
            project=project,
        )
        assert "¿Qué evidencias debo generar?" in result
        assert "Motor de Preaprobación" in result
        assert "classification" in result
        assert "banking" in result
        assert "buró de crédito" in result

    def test_null_fields_omitted(self):
        """Profile fields that are None/empty are not included in the output."""
        project = _make_project(
            ai_type=None,
            industry=None,
            existing_vendor=None,
            data_used=[],
        )
        result = build_retrieval_query(
            user_question="Test query",
            project=project,
        )
        assert "Tipo IA" not in result
        assert "Sector" not in result
        assert "Proveedores" not in result

    def test_description_truncated_to_150_chars(self):
        """project_description is trimmed to 150 chars to avoid dilution."""
        long_desc = "A" * 300
        project = _make_project(project_description=long_desc)
        result = build_retrieval_query(user_question="Q", project=project)
        included_desc = result.split("Descripción: ")[-1].split(" | ")[0]
        assert len(included_desc) <= 150

    def test_result_is_single_line(self):
        """The retrieval query must not contain newlines."""
        project = _make_project(project_description="Line1\nLine2")
        result = build_retrieval_query(user_question="Q", project=project)
        assert "\n" not in result

    def test_data_used_list_joined(self):
        """A data_used list is joined into a single string."""
        project = _make_project(data_used=["pii", "phi"])
        result = build_retrieval_query(user_question="Q", project=project)
        assert "pii" in result
        assert "phi" in result


class TestSplitTopKAcrossFrameworks:
    def test_splits_evenly_for_two_frameworks(self):
        result = split_top_k_across_frameworks(8, ["iso_42001", "nist_airmf"])
        assert result == [("iso_42001", 4), ("nist_airmf", 4)]

    def test_assigns_remainder_to_first_frameworks(self):
        result = split_top_k_across_frameworks(5, ["iso_42001", "nist_airmf"])
        assert result == [("iso_42001", 3), ("nist_airmf", 2)]

    def test_empty_frameworks_returns_empty(self):
        assert split_top_k_across_frameworks(8, []) == []


# ==================================================================
# Integration — _prepare_pipeline wiring
# ==================================================================


class TestPipelineIntegration:
    """
    Verify that _prepare_pipeline passes the enriched retrieval_query
    (not the raw request.query) to RetrievalService.retrieve, and
    that effective_top_k is bumped to 8 when project_id is present.
    """

    def _make_service(self):
        """Instantiate AIService with fully mocked dependencies."""
        from app.domain.ai_service import AIService

        svc = AIService.__new__(AIService)
        svc.llm_router = MagicMock()
        svc.prompt_service = MagicMock()
        svc.retrieval_service = MagicMock()
        svc.trace_service = MagicMock()
        svc.eval_service = MagicMock()
        svc.project_service = MagicMock()
        svc.thread_service = MagicMock()
        svc.response_risk_service = MagicMock()
        svc.db = MagicMock()
        return svc

    def _configure_stubs(self, svc, project=None):
        """Wire common stubs that satisfy the full pipeline."""
        mock_prompt = MagicMock()
        mock_prompt.content = "You are an AI governance assistant."
        mock_prompt.name = "chatbot"
        mock_prompt.version = 1
        svc.prompt_service.get_active_prompt.return_value = mock_prompt

        svc.retrieval_service.retrieve.return_value = []
        svc.project_service.get_optional.return_value = project

        mock_thread = MagicMock()
        mock_thread.id = "00000000-0000-0000-0000-000000000001"
        svc.thread_service.get_or_create.return_value = mock_thread
        svc.thread_service.get_recent_messages.return_value = []
        svc.trace_service.list_project_traces.return_value = []

    def test_retrieval_receives_enriched_query_not_raw(self):
        """
        When project_id is provided, retrieve() must be called with a
        retrieval_query that includes project context — not the raw query.
        """
        from uuid import UUID
        from app.schemas.ai import AIQueryRequest

        svc = self._make_service()
        project = _make_project(
            name="Motor de Preaprobación",
            ai_type="classification",
            industry="banking",
        )
        self._configure_stubs(svc, project=project)

        raw_query = "¿Qué evidencias debo generar según ISO 42001 y NIST?"
        req = AIQueryRequest(
            query=raw_query,
            use_case="chatbot",
            project_id=UUID("8dd1894a-8977-468c-b3ae-682d548d400b"),
        )

        svc._prepare_pipeline(req, tenant_id="acme", user_id="user-1")

        call_kwargs = svc.retrieval_service.retrieve.call_args
        used_query = call_kwargs.kwargs.get("query") or call_kwargs.args[1]

        # The retrieval query must be enriched with project context
        assert used_query != raw_query, (
            "retrieve() should receive the enriched retrieval_query, not the raw query"
        )
        assert "Motor de Preaprobación" in used_query or "classification" in used_query

    def test_effective_top_k_is_8_with_project_id(self):
        """
        When project_id is present and top_k is at its default of 5,
        retrieve() must be called with top_k=8.
        """
        from uuid import UUID
        from app.schemas.ai import AIQueryRequest

        svc = self._make_service()
        project = _make_project()
        self._configure_stubs(svc, project=project)

        req = AIQueryRequest(
            query="¿Qué evidencias necesito?",
            use_case="chatbot",
            project_id=UUID("8dd1894a-8977-468c-b3ae-682d548d400b"),
            # top_k is not set → defaults to 5
        )

        svc._prepare_pipeline(req, tenant_id="acme", user_id="user-1")

        call_kwargs = svc.retrieval_service.retrieve.call_args
        used_top_k = call_kwargs.kwargs.get("top_k") or call_kwargs.args[2]
        assert used_top_k == 8, f"Expected effective_top_k=8, got {used_top_k}"

    def test_explicit_top_k_respected_with_project_id(self):
        """
        When the caller sets an explicit top_k != 5, it must be preserved
        even when project_id is provided.
        """
        from uuid import UUID
        from app.schemas.ai import AIQueryRequest

        svc = self._make_service()
        project = _make_project()
        self._configure_stubs(svc, project=project)

        req = AIQueryRequest(
            query="¿Qué evidencias necesito?",
            use_case="chatbot",
            project_id=UUID("8dd1894a-8977-468c-b3ae-682d548d400b"),
            top_k=3,
        )

        svc._prepare_pipeline(req, tenant_id="acme", user_id="user-1")

        call_kwargs = svc.retrieval_service.retrieve.call_args
        used_top_k = call_kwargs.kwargs.get("top_k") or call_kwargs.args[2]
        assert used_top_k == 3, f"Explicit top_k=3 should be respected, got {used_top_k}"

    def test_no_project_id_uses_request_top_k(self):
        """Without project_id, top_k passes through unchanged."""
        from app.schemas.ai import AIQueryRequest

        svc = self._make_service()
        self._configure_stubs(svc, project=None)

        req = AIQueryRequest(
            query="What is ISO 42001?",
            use_case="chatbot",
            # no project_id
        )

        svc._prepare_pipeline(req, tenant_id="acme", user_id="user-1")

        call_kwargs = svc.retrieval_service.retrieve.call_args
        used_top_k = call_kwargs.kwargs.get("top_k") or call_kwargs.args[2]
        assert used_top_k == 5

    def test_framework_filter_inferred_from_question(self):
        """
        When framework_filter is not in the request but the query mentions
        ISO 42001, resolved_frameworks passed to retrieve() includes iso_42001.
        """
        from uuid import UUID
        from app.schemas.ai import AIQueryRequest

        svc = self._make_service()
        project = _make_project()
        self._configure_stubs(svc, project=project)

        req = AIQueryRequest(
            query="¿Evidencias ISO 42001?",
            use_case="chatbot",
            project_id=UUID("8dd1894a-8977-468c-b3ae-682d548d400b"),
            # framework_filter not set
        )

        svc._prepare_pipeline(req, tenant_id="acme", user_id="user-1")

        call_kwargs = svc.retrieval_service.retrieve.call_args
        used_filter = call_kwargs.kwargs.get("framework_filter")
        assert used_filter is not None
        assert "iso_42001" in used_filter

    def test_explicit_framework_filter_not_overridden(self):
        """
        An explicit framework_filter in the request must reach retrieve()
        unchanged, regardless of what the query text contains.
        """
        from uuid import UUID
        from app.schemas.ai import AIQueryRequest

        svc = self._make_service()
        project = _make_project()
        self._configure_stubs(svc, project=project)

        req = AIQueryRequest(
            query="¿Evidencias ISO 42001?",
            use_case="chatbot",
            project_id=UUID("8dd1894a-8977-468c-b3ae-682d548d400b"),
            framework_filter=["pci_dss"],
        )

        svc._prepare_pipeline(req, tenant_id="acme", user_id="user-1")

        call_kwargs = svc.retrieval_service.retrieve.call_args
        used_filter = call_kwargs.kwargs.get("framework_filter")
        assert used_filter == ["pci_dss"]

    def test_balanced_retrieval_splits_across_inferred_frameworks(self):
        """
        When ISO and NIST are inferred, retrieve() is called once per
        framework with a split top_k budget.
        """
        from uuid import UUID
        from app.schemas.ai import AIQueryRequest

        svc = self._make_service()
        project = _make_project()
        self._configure_stubs(svc, project=project)

        req = AIQueryRequest(
            query="¿Qué evidencias debo generar según ISO 42001 y NIST?",
            use_case="chatbot",
            project_id=UUID("8dd1894a-8977-468c-b3ae-682d548d400b"),
        )

        svc._prepare_pipeline(req, tenant_id="acme", user_id="user-1")

        assert svc.retrieval_service.retrieve.call_count == 2
        calls = svc.retrieval_service.retrieve.call_args_list
        filters = [c.kwargs.get("framework_filter") for c in calls]
        top_ks = [c.kwargs.get("top_k") for c in calls]
        assert ["iso_42001"] in filters
        assert ["nist_airmf"] in filters
        assert sum(top_ks) == 8
