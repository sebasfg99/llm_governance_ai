# ============================================================
# app/tests/test_grounding_metric.py
#
# Regression tests for the grounding metric fix.
#
# THE BUG THESE LOCK DOWN:
# grounding_score() used to divide the response∩context overlap by the UNION
# of both word sets (Jaccard). That made the score a function of how much
# context was retrieved rather than of how grounded the answer was, with two
# consequences:
#
#   1. Unreachable threshold — with a 78-word answer over a 516-word context
#      the arithmetic maximum was 78/594 = 0.1416, below the
#      GROUNDING_THRESHOLD of 0.15 the caller compares against. No response
#      could pass, regardless of quality.
#   2. Inverted incentive — retrieving MORE relevant context grew the union
#      and pushed the score DOWN, so real retrieval improvements (e.g. the
#      agent-scoped diversity cap) registered as quality regressions.
#
# The fix is containment: overlap / |response words|. These tests assert the
# property that matters — independence from context size — rather than
# hard-coding scores that would drift with the stop-word list.
# ============================================================

import pytest

from app.domain.evals.graders import grounding_score
from app.domain.evals.service import GROUNDING_THRESHOLD


class TestContainmentFormula:
    def test_fully_grounded_response_scores_one(self):
        """
        Expected input:  every meaningful response word appears in the context.
        Expected output: exactly 1.0. Under the old Jaccard formula this same
                         input scored |R|/|R∪C|, i.e. arbitrarily low.
        """
        context = ["la organización debe documentar la trazabilidad de los datos"]
        assert grounding_score("documentar la trazabilidad de los datos", context) == 1.0

    def test_completely_ungrounded_response_scores_zero(self):
        context = ["algorithmic transparency requirements for AI systems"]
        assert grounding_score("gallo pinto arroz frijoles culantro", context) == 0.0

    def test_score_is_independent_of_context_size(self):
        """
        THE core property. The same response over the same relevant chunk must
        score identically whether or not unrelated context accompanies it.

        This is the regression guard for the inverted incentive: under Jaccard,
        appending padding chunks strictly lowered the score.
        """
        response = "la organización debe documentar la trazabilidad"
        relevant = "la organización debe documentar la trazabilidad de los datos"
        padding = [
            "unrelated payment card industry cryptographic protocol inventory",
            "vendor governance third party risk assessment questionnaire",
            "annex bibliography ISO IEC technical report overview terminology",
        ]

        small = grounding_score(response, [relevant])
        large = grounding_score(response, [relevant] + padding)

        assert small == large == 1.0

    def test_more_context_never_lowers_the_score(self):
        """Monotonicity: adding context can only keep or raise containment,
        never lower it, because the denominator no longer moves."""
        response = "transparencia explicabilidad y trazabilidad de los datos"
        base = ["transparencia y explicabilidad"]
        extra = base + ["trazabilidad de los datos de entrenamiento"]

        assert grounding_score(response, extra) >= grounding_score(response, base)

    def test_partial_overlap_is_proportional_to_the_response(self):
        """Half the response words present -> ~0.5, regardless of context bulk."""
        response = "alpha bravo charlie delta"
        score = grounding_score(response, ["alpha bravo " + "padding " * 200])
        assert score == pytest.approx(0.5, abs=0.01)


class TestThresholdIsReachable:
    def test_perfect_grounding_clears_the_threshold(self):
        """
        The bug in one assertion: a perfectly grounded answer must be able to
        pass. Under Jaccard with a realistic context this was arithmetically
        impossible.
        """
        context = ["la organización debe documentar la trazabilidad de los datos " * 40]
        score = grounding_score("documentar la trazabilidad de los datos", context)
        assert score == 1.0
        assert score >= GROUNDING_THRESHOLD

    def test_threshold_sits_between_the_calibration_bands(self):
        """
        Guards the recalibration: the threshold must separate the labelled
        grounded set (lowest observed 0.4545) from the ungrounded set (highest
        observed 0.3704). A future edit that moves it outside that gap — e.g.
        back to a Jaccard-era value — should fail here.
        """
        assert 0.3704 < GROUNDING_THRESHOLD < 0.4545

    def test_empty_inputs_score_zero_not_crash(self):
        assert grounding_score("", ["some context"]) == 0.0
        assert grounding_score("some response", []) == 0.0
        # A response made entirely of stop words has no meaningful words.
        assert grounding_score("the a an of to", ["some context"]) == 0.0


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
