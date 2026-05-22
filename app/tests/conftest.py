# ============================================================
# app/tests/conftest.py — Shared fixtures
#
# pytest discovers this file automatically. Anything defined
# here is available to every test module without import.
# ============================================================

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest


# ============================================================
# Sample chunks
# ============================================================
@pytest.fixture
def sample_chunks() -> list:
    """Three ChunkResult-like objects with varied similarity scores."""
    from app.domain.retrieval.ranking import ChunkResult

    return [
        ChunkResult(
            chunk_id="11111111-1111-1111-1111-111111111111",
            document_id="aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa",
            tenant_id="acme",
            content=(
                "ISO 42001 Articulo 9 establece que el sistema de gesti\u00f3n de "
                "IA debe identificar riesgos de equidad y discriminaci\u00f3n."
            ),
            chunk_index=0,
            similarity=0.82,
            section_ref="Articulo 9",
        ),
        ChunkResult(
            chunk_id="22222222-2222-2222-2222-222222222222",
            document_id="bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb",
            tenant_id="acme",
            content=(
                "El EU AI Act clasifica como sistemas de alto riesgo aquellos "
                "que afectan derechos fundamentales o seguridad."
            ),
            chunk_index=2,
            similarity=0.71,
            section_ref="Anexo III",
        ),
        ChunkResult(
            chunk_id="33333333-3333-3333-3333-333333333333",
            document_id="cccccccc-cccc-cccc-cccc-cccccccccccc",
            tenant_id="acme",
            content="Texto irrelevante sobre cualquier otra cosa.",
            chunk_index=1,
            similarity=0.42,
        ),
    ]


# ============================================================
# Mock Supabase client
# ============================================================
class _ChainableTable:
    """
    Minimal in-memory Supabase table mock that supports the chain
    pattern used everywhere in the code base:

        db.table("x").select("*").eq("k", "v").execute()

    It returns whatever the test configured via .set_response().
    """

    def __init__(self) -> None:
        self._response_data: list[dict] = []

    def set_response(self, data: list[dict]) -> "_ChainableTable":
        self._response_data = data
        return self

    def select(self, *_: Any, **__: Any) -> "_ChainableTable":
        return self

    def insert(self, payload: Any) -> "_ChainableTable":
        # Simulate inserting and echoing back with an id
        if isinstance(payload, dict):
            payload = {**payload}
            payload.setdefault("id", "00000000-0000-0000-0000-000000000099")
            self._response_data = [payload]
        return self

    def update(self, *_: Any, **__: Any) -> "_ChainableTable":
        return self

    def eq(self, *_: Any, **__: Any) -> "_ChainableTable":
        return self

    def in_(self, *_: Any, **__: Any) -> "_ChainableTable":
        return self

    def gte(self, *_: Any, **__: Any) -> "_ChainableTable":
        return self

    def lt(self, *_: Any, **__: Any) -> "_ChainableTable":
        return self

    def limit(self, *_: Any, **__: Any) -> "_ChainableTable":
        return self

    def order(self, *_: Any, **__: Any) -> "_ChainableTable":
        return self

    def range(self, *_: Any, **__: Any) -> "_ChainableTable":
        return self

    def execute(self) -> Any:
        result = MagicMock()
        result.data = list(self._response_data)
        return result


@pytest.fixture
def mock_db() -> MagicMock:
    """A Supabase client mock with chainable .table() and .rpc()."""
    db = MagicMock()
    tables: dict[str, _ChainableTable] = {}

    def _table(name: str) -> _ChainableTable:
        if name not in tables:
            tables[name] = _ChainableTable()
        return tables[name]

    db.table.side_effect = _table
    db._tables = tables  # expose for tests that need to seed responses
    rpc_mock = MagicMock()
    rpc_result = MagicMock()
    rpc_result.data = []
    rpc_mock.execute.return_value = rpc_result
    db.rpc.return_value = rpc_mock
    return db


# ============================================================
# Mock LLM judge
# ============================================================
@pytest.fixture
def mock_judge_clean(monkeypatch):
    """Patch the LLM judge to return a "clean" verdict (3 supported claims)."""
    def _fake(*_args: Any, **_kwargs: Any) -> dict:
        return {
            "claims": [
                {
                    "text": "ISO 42001 exige identificar riesgos de equidad",
                    "supported": True,
                    "supporting_chunk_indexes": [1],
                    "is_hallucination": False,
                },
                {
                    "text": "El EU AI Act clasifica los sistemas de alto riesgo",
                    "supported": True,
                    "supporting_chunk_indexes": [2],
                    "is_hallucination": False,
                },
                {
                    "text": "El gobierno de IA es relevante",
                    "supported": True,
                    "supporting_chunk_indexes": [1, 2],
                    "is_hallucination": False,
                },
            ],
            "claims_total": 3,
            "claims_supported": 3,
            "hallucination": False,
            "should_have_said_idk": False,
            "reasoning": "Todas las afirmaciones est\u00e1n soportadas por los chunks.",
            "raw": {"mocked": True},
            "ok": True,
        }

    monkeypatch.setattr("app.domain.evals.evaluator.judge_response", _fake)
    return _fake


@pytest.fixture
def mock_judge_hallucinated(monkeypatch):
    """Patch the LLM judge to flag a hallucination."""
    def _fake(*_args: Any, **_kwargs: Any) -> dict:
        return {
            "claims": [
                {
                    "text": "El sistema fue certificado por Naciones Unidas",
                    "supported": False,
                    "supporting_chunk_indexes": [],
                    "is_hallucination": True,
                },
                {
                    "text": "ISO 42001 exige identificar riesgos",
                    "supported": True,
                    "supporting_chunk_indexes": [1],
                    "is_hallucination": False,
                },
            ],
            "claims_total": 2,
            "claims_supported": 1,
            "hallucination": True,
            "should_have_said_idk": False,
            "reasoning": "Una afirmaci\u00f3n no encontrada en los chunks.",
            "raw": {"mocked": True},
            "ok": True,
        }

    monkeypatch.setattr("app.domain.evals.evaluator.judge_response", _fake)
    return _fake


@pytest.fixture
def mock_judge_idk_failure(monkeypatch):
    """Patch the judge to flag should_have_said_idk."""
    def _fake(*_args: Any, **_kwargs: Any) -> dict:
        return {
            "claims": [],
            "claims_total": 0,
            "claims_supported": 0,
            "hallucination": False,
            "should_have_said_idk": True,
            "reasoning": "Sin contexto relevante; el modelo deber\u00eda haberse abstenido.",
            "raw": {"mocked": True},
            "ok": True,
        }

    monkeypatch.setattr("app.domain.evals.evaluator.judge_response", _fake)
    return _fake
