# ============================================================
# app/tests/test_grounding_metric.py
#
# Tests for the grounding metric.
#
# WHAT THE METRIC IS NOW (v3, 2026-08):
# grounding_score() is embedding cosine between the response and the
# retrieved chunks (maximum over chunks). It replaced a lexical
# word-overlap metric that was measurably INVERTED on this bilingual
# corpus: a fabricated answer scored 0.462 while a CORRECT answer whose
# evidence was in English scored 0.242, and off-topic text scored 0.293.
# Word overlap can only see evidence written in the same language as the
# answer, and answers are always Spanish while much of the corpus is
# English.
#
# HOW THESE TESTS ARE SPLIT:
#   - Behavioural tests stub the embedding call, so they are fast,
#     deterministic and assert the FORMULA (max over chunks, bounds,
#     guards, degraded path) rather than model output.
#   - One calibration test uses the real embedding model, because the
#     whole point of the change is what real vectors do with real
#     bilingual text. It is the only networked test here.
# ============================================================

import math

import pytest

from app.domain.evals import graders
from app.domain.evals.graders import grounding_score
from app.domain.evals.service import GROUNDING_THRESHOLD


@pytest.fixture(autouse=True)
def _clear_embedding_cache():
    """grounding_score memoises embeddings per process; isolate tests."""
    graders._EMBED_CACHE.clear()
    yield
    graders._EMBED_CACHE.clear()


def _unit(*components: float) -> list[float]:
    """A normalised vector, so cosine values are easy to reason about."""
    norm = math.sqrt(sum(c * c for c in components))
    return [c / norm for c in components]


@pytest.fixture
def stub_embeddings(monkeypatch):
    """
    Map exact texts to fixed vectors.

    Patches embed_texts where graders imports it, so no network call is
    made and the assertions are about the formula only.
    """
    def _install(mapping: dict[str, list[float]]):
        def _fake(texts):
            missing = [t for t in texts if t not in mapping]
            if missing:
                raise AssertionError(f"unmapped text in test: {missing[:1]}")
            return [mapping[t] for t in texts]
        monkeypatch.setattr(
            "app.domain.documents.embeddings.embed_texts", _fake
        )
    return _install


# ============================================================
# The formula
# ============================================================
class TestSemanticFormula:
    def test_identical_meaning_scores_near_one(self, stub_embeddings):
        """
        Expected input:  response and chunk mapped to the same vector.
        Expected output: cosine 1.0 — the ceiling is reachable, which the
                         previous Jaccard implementation could not manage.
        """
        v = _unit(1, 0, 0)
        stub_embeddings({"respuesta": v, "contexto": v})
        assert grounding_score("respuesta", ["contexto"]) == 1.0

    def test_orthogonal_meaning_scores_zero(self, stub_embeddings):
        """
        Expected input:  response orthogonal to the only chunk.
        Expected output: 0.0 — no semantic support at all.
        """
        stub_embeddings({"respuesta": _unit(1, 0, 0), "contexto": _unit(0, 1, 0)})
        assert grounding_score("respuesta", ["contexto"]) == 0.0

    def test_takes_the_maximum_not_the_mean(self, stub_embeddings):
        """
        Expected input:  one strongly supporting chunk among three
                         unrelated ones.
        Expected output: the score of the supporting chunk.

        WHY MAX: a grounded answer usually rests on one or two of the
        five chunks retrieved. Averaging would punish the answer for
        whatever else retrieval happened to return.
        """
        stub_embeddings({
            "respuesta": _unit(1, 0, 0),
            "match": _unit(1, 0, 0),
            "otro1": _unit(0, 1, 0),
            "otro2": _unit(0, 0, 1),
        })
        assert grounding_score("respuesta", ["otro1", "match", "otro2"]) == 1.0

    def test_more_context_never_lowers_the_score(self, stub_embeddings):
        """
        Expected input:  the same answer, with extra unrelated chunks added.
        Expected output: score does not drop.

        This is the property the original Jaccard bug violated: retrieving
        MORE relevant context grew the union and pushed the score DOWN, so
        genuine retrieval improvements looked like quality regressions.
        """
        stub_embeddings({
            "respuesta": _unit(1, 0, 0),
            "match": _unit(1, 0, 0),
            "extra": _unit(0, 1, 0),
        })
        small = grounding_score("respuesta", ["match"])
        large = grounding_score("respuesta", ["match", "extra", "extra"])
        assert large >= small

    def test_score_stays_within_bounds(self, stub_embeddings):
        """Negative cosine (opposed vectors) is clamped to 0.0, not returned."""
        stub_embeddings({"respuesta": _unit(1, 0, 0), "contexto": _unit(-1, 0, 0)})
        score = grounding_score("respuesta", ["contexto"])
        assert 0.0 <= score <= 1.0

    def test_precomputed_chunk_embeddings_are_used(self, monkeypatch):
        """
        Expected input:  chunk vectors supplied by the caller.
        Expected output: only the RESPONSE is embedded — chunk embeddings
                         already exist in document_chunks.embedding, and
                         re-embedding them per evaluation is pure waste.
        """
        calls = []

        def _fake(texts):
            calls.append(list(texts))
            return [_unit(1, 0, 0) for _ in texts]

        monkeypatch.setattr("app.domain.documents.embeddings.embed_texts", _fake)
        score = grounding_score(
            "respuesta", ["contexto"], chunk_embeddings=[_unit(1, 0, 0)]
        )
        assert score == 1.0
        assert calls == [["respuesta"]], "chunks must not be re-embedded"

    def test_mismatched_embedding_count_falls_back(self, monkeypatch):
        """A caller passing the wrong number of vectors must not corrupt
        the score silently — it degrades to the lexical path instead."""
        monkeypatch.setattr(
            "app.domain.documents.embeddings.embed_texts",
            lambda texts: [_unit(1, 0, 0) for _ in texts],
        )
        score = grounding_score(
            "la política de seguridad", ["la política de seguridad"],
            chunk_embeddings=[],
        )
        assert 0.0 <= score <= 1.0


# ============================================================
# Guards and the degraded path
# ============================================================
class TestGuardsAndDegradation:
    def test_empty_inputs_score_zero_not_crash(self):
        assert grounding_score("", ["some context"]) == 0.0
        assert grounding_score("some response", []) == 0.0

    def test_embedding_failure_falls_back_to_lexical(self, monkeypatch):
        """
        Expected input:  the embedding API raising.
        Expected output: a lexical containment value instead of an
                         exception. Evaluation must never take the
                         response pipeline down, and a scoring outage
                         should read LOW against the 0.62 gate rather
                         than silently passing everything.
        """
        def _boom(texts):
            raise RuntimeError("embedding backend down")

        monkeypatch.setattr("app.domain.documents.embeddings.embed_texts", _boom)
        score = grounding_score(
            "la organización debe documentar los riesgos",
            ["la organización debe documentar los riesgos de IA"],
        )
        assert 0.0 < score <= 1.0


# ============================================================
# Calibration against the real embedding model
# ============================================================
class TestSemanticCalibration:
    """
    The bands this asserts come from the n=35 calibration set in
    scripts/calibrate_grounding_metric.py:

        real answer, ES sources    mean 0.790  [0.70-0.87]
        real answer, EN sources    mean 0.638  [0.50-0.74]
        fabricated but plausible   mean 0.597  [0.39-0.73]
        off-topic (control)        mean 0.148  [0.11-0.24]

    Thresholds here are the band edges, not the means, so normal model
    drift does not turn this red.
    """

    def test_grounding_score_semantic_calibration(self):
        spanish_answer = (
            "La organización debe definir y documentar un proceso de evaluación "
            "del riesgo de la IA alineado con su política y objetivos, que "
            "identifique y analice los riesgos y determine sus niveles."
        )
        spanish_context = [
            "La organización debe establecer un proceso de evaluación del riesgo "
            "de la IA que se alinee con la política de IA y los objetivos de la "
            "IA, identifique los riesgos y analice sus consecuencias y "
            "probabilidad, y determine los niveles de riesgo."
        ]
        english_context = [
            "The organization shall establish an AI risk assessment process that "
            "aligns with the AI policy and AI objectives, identifies risks, "
            "analyses their consequences and likelihood, and determines risk "
            "levels."
        ]
        off_topic_answer = (
            "La receta tradicional requiere harina, huevos y mantequilla. Se "
            "mezcla todo hasta obtener una masa homogénea y se hornea a 180 "
            "grados durante 25 minutos."
        )

        same_language = grounding_score(spanish_answer, spanish_context)
        cross_language = grounding_score(spanish_answer, english_context)
        off_topic = grounding_score(off_topic_answer, spanish_context)

        assert same_language > 0.70, f"same-language grounding too low: {same_language}"
        # The whole point of the change: evidence in the other language is
        # still visible. The lexical metric scored this case ~0.07.
        assert cross_language > 0.50, f"cross-language grounding too low: {cross_language}"
        assert off_topic < 0.20, f"off-topic scored too high: {off_topic}"
        assert cross_language > off_topic

    def test_threshold_separates_the_calibration_bands(self):
        """
        GROUNDING_THRESHOLD must sit between the fabricated band and the
        real-answer bands — that is what makes it a gate rather than a
        number.
        """
        assert 0.35 < GROUNDING_THRESHOLD < 0.70
        assert GROUNDING_THRESHOLD == 0.62




# ============================================================
# Fix 2 — the persisted evaluation must reuse the judge's verdict
#
# THE BUG: ai_service ran the LLM judge once inside
# _run_governance_pipeline, then called evaluate_trace() with
# use_llm_judge=False to avoid paying twice. evaluate_trace then RECOMPUTED
# the governance block with no judge, so context_adherence fell back to the
# lexical grounding proxy and hallucination to a threshold on it. The real
# claim-level verdict the caller already held was thrown away — and the
# persisted pass/fail decision was built on the proxy.
#
# Observable symptom in production logs: context_adh was always exactly equal
# to grounding, and gated at >= 0.7, which the proxy can never reach.
# ============================================================
from unittest.mock import MagicMock

from app.domain.evals.service import EvalService


def _judge_verdict() -> dict:
    """A governance block as evaluate_response() returns it WITH a live judge."""
    return {
        "retrieval_precision": 0.8,
        "groundedness_score": 0.55,
        # The distinguishing value: a real claims ratio, not the grounding proxy.
        "context_adherence_score": 0.8,
        "citation_score": 0.0,
        "completeness_score": 0.5,
        "instruction_adherence_score": 1.0,
        "hallucination": False,
        "should_have_said_idk": False,
        "claims_total": 5,
        "claims_supported": 4,
        "judge_ok": True,
        "judge_raw": {"claims": []},
    }


@pytest.fixture
def eval_service():
    """EvalService over a mocked DB — the insert is irrelevant here."""
    db = MagicMock()
    db.table.return_value.insert.return_value.execute.return_value = MagicMock(
        data=[{"id": "eval-1"}]
    )
    return EvalService(db)


class TestPrecomputedGovernanceReuse:
    def test_precomputed_verdict_is_persisted_verbatim(self, eval_service, monkeypatch):
        """
        Expected input:  a governance block from a judged evaluation.
        Expected output: those values are persisted, and evaluate_response is
                         never called again (no second judge cost, and no
                         downgrade to the proxy).
        """
        import app.domain.evals.service as svc

        called = {"n": 0}
        monkeypatch.setattr(
            svc, "evaluate_response",
            lambda **kw: called.__setitem__("n", called["n"] + 1) or {},
        )

        result = eval_service.evaluate_trace(
            trace_id="t1",
            response_text="una respuesta fundamentada en el contexto",
            chunk_texts=["una respuesta fundamentada en el contexto recuperado"],
            query="q",
            precomputed_governance=_judge_verdict(),
        )

        assert called["n"] == 0, "evaluate_response must not be re-run"
        gov = result.governance
        assert gov["context_adherence_score"] == 0.8
        assert gov["claims_total"] == 5 and gov["claims_supported"] == 4
        assert gov["judge_ok"] is True

    def test_context_adherence_is_not_the_grounding_proxy(self, eval_service):
        """
        The exact production symptom: context_adherence must no longer collapse
        onto the grounding score when a judge verdict exists.
        """
        result = eval_service.evaluate_trace(
            trace_id="t2",
            response_text="documentar la trazabilidad de los datos",
            chunk_texts=["documentar la trazabilidad de los datos"],
            query="q",
            precomputed_governance=_judge_verdict(),
        )
        assert result.grounding_score == 1.0          # containment
        assert result.governance["context_adherence_score"] == 0.8  # claims ratio
        assert result.governance["context_adherence_score"] != result.grounding_score

    def test_falls_back_to_fresh_computation_when_absent(self, eval_service, monkeypatch):
        """
        None and {} (the latter being what ai_service holds after a failed
        inline evaluation) must both fall through to a fresh computation with
        the judge enabled, rather than persisting an empty block.
        """
        import app.domain.evals.service as svc

        seen: list[bool] = []

        def fake_eval(**kw):
            seen.append(kw["use_llm_judge"])
            return _judge_verdict()

        monkeypatch.setattr(svc, "evaluate_response", fake_eval)

        for absent in (None, {}):
            eval_service.evaluate_trace(
                trace_id="t3",
                response_text="respuesta",
                chunk_texts=["contexto"],
                query="q",
                precomputed_governance=absent,
                use_llm_judge=True,
            )

        assert seen == [True, True], "judge must run when no verdict was handed over"
