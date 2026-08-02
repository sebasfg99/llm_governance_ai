# ============================================================
# app/tests/test_agent_tags_retrieval.py
#
# WHAT THIS FILE COVERS:
# The business-agent (`agent_tags`) filter added on top of the existing RAG
# pipeline, plus its automatic broadening to the "general" corpus.
#
# Two independent behaviours are under test:
#
#   1. PLUMBING — does the `p_agent_tags` argument actually reach the four
#      Postgres RPCs, and is it *absent* when no agent filter was requested?
#      The absence half matters as much as the presence half: existing callers
#      (risk/service.py, policies/service.py) must keep producing a payload
#      that a pre-migration-009 database still accepts.
#
#   2. FALLBACK — given a weak agent-scoped result, does retrieve() re-run the
#      search with agent_tags + ["general"], and does it correctly decline to
#      do so when the result is good, when the caller disabled the fallback,
#      or when "general" is already in the requested set?
#
# TEST STRATEGY:
# No database and no OpenAI calls. A FakeDB records every RPC invocation and
# replays canned rows, while the two external dependencies of the pipeline
# (`embed_single` for embeddings and `generate_bilingual_queries` for query
# expansion) are patched at their import site inside the service module.
# `rerank_chunks` is likewise replaced by a deterministic similarity sort so
# assertions describe the agent-tag logic rather than the reranker's weights.
# ============================================================

from __future__ import annotations

import pytest

from app.core.config import settings
from app.core.errors import UpstreamError
from app.domain.retrieval.ranking import ChunkResult
from app.domain.retrieval.service import RetrievalService


# ============================================================
# Test doubles
# ============================================================
class _FakeExecute:
    """Mimics the object returned by supabase's `db.rpc(...)`."""

    def __init__(self, rows: list[dict], error: Exception | None = None) -> None:
        self._rows = rows
        self._error = error

    def execute(self):
        # The real client raises from .execute(), not from .rpc(), so the
        # failure has to be produced here for the service's try/except around
        # `.execute()` to be exercised the way production does.
        if self._error is not None:
            raise self._error
        return type("Result", (), {"data": self._rows})()


class FakeDB:
    """
    Minimal stand-in for the Supabase client.

    Expected input:
        rows_by_rpc: maps an RPC name to the rows it should return. Any RPC
                     not listed returns no rows.
        error:       when set, EVERY rpc call raises it, which is how the
                     "migration 009 not applied" scenario is simulated.

    Behaviour:
        Every call is appended to `self.calls` as (rpc_name, args_dict) so a
        test can assert on the exact payload the service built.
    """

    def __init__(
        self,
        rows_by_rpc: dict[str, list[dict]] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.rows_by_rpc = rows_by_rpc or {}
        self.error = error
        self.calls: list[tuple[str, dict]] = []

    def rpc(self, name: str, args: dict):
        self.calls.append((name, args))
        return _FakeExecute(self.rows_by_rpc.get(name, []), self.error)

    # --- convenience accessors used by the assertions -------------------
    def args_for(self, rpc_name: str) -> list[dict]:
        """Every payload sent to `rpc_name`, in call order."""
        return [args for name, args in self.calls if name == rpc_name]

    def agent_tag_args(self, rpc_name: str) -> list[list[str] | None]:
        """
        The p_agent_tags value of each call to `rpc_name`.

        Returns None for a call that omitted the key entirely, which is the
        distinction the plumbing tests care about.
        """
        return [args.get("p_agent_tags") for args in self.args_for(rpc_name)]

    def industry_args(self, rpc_name: str) -> list[list[str] | None]:
        """The p_industry_codes value of each call to `rpc_name`."""
        return [args.get("p_industry_codes") for args in self.args_for(rpc_name)]


def make_row(
    chunk_id: str,
    similarity: float,
    document_id: str | None = None,
    industry_codes: list[str] | None = None,
) -> dict:
    """
    Build one raw DB row as the match_* RPCs return it.

    Each row gets its own document_id by default so the per-document diversity
    cap (settings.retrieval_max_chunks_per_document, normally 1) never merges
    or reorders rows behind the test's back.

    `industry_codes` is included from migration 010 onward; it defaults to
    ["ALL"] here so rows are universally visible unless a test says otherwise.
    """
    return {
        "id": chunk_id,
        "document_id": document_id or f"doc-{chunk_id}",
        "tenant_id": "global",
        "content": f"content of {chunk_id}",
        "chunk_index": 0,
        "similarity": similarity,
        "section_ref": None,
        "industry_codes": industry_codes if industry_codes is not None else ["ALL"],
    }


# ============================================================
# Fixtures
# ============================================================
@pytest.fixture(autouse=True)
def _stub_pipeline_externals(monkeypatch):
    """
    Neutralise everything in retrieve() that is not the agent-tag logic.

    - generate_bilingual_queries → a single variant, so each pass performs
      exactly one round of RPC calls and call counts stay readable.
    - embed_single → a constant vector, and it counts its own invocations so
      a test can prove the fallback pass reuses embeddings instead of paying
      for a second embedding round-trip.
    - rerank_chunks → a plain descending-similarity sort. The real reranker
      mixes in keyword features; pinning it here keeps "top-1 score" exactly
      predictable, which is what the fallback threshold is compared against.
    """
    calls = {"embed": 0}

    def fake_embed_single(text: str) -> list[float]:
        calls["embed"] += 1
        return [0.1, 0.2, 0.3]

    monkeypatch.setattr(
        "app.domain.retrieval.service.embed_single", fake_embed_single
    )
    monkeypatch.setattr(
        "app.domain.retrieval.service.generate_bilingual_queries",
        lambda topic: [topic],
    )
    monkeypatch.setattr(
        "app.domain.retrieval.service.rerank_chunks",
        lambda chunks, query: sorted(chunks, key=lambda c: c.similarity, reverse=True),
    )
    return calls


@pytest.fixture
def service_factory():
    """
    Build a RetrievalService around a FakeDB.

    RetrievalService.__init__ only stores the client, so a plain construction
    is enough — no patching of the module-level supabase factory required.
    """

    def _factory(db: FakeDB) -> RetrievalService:
        return RetrievalService(db)

    return _factory


# ============================================================
# 1. Plumbing — vector-only path (match_chunks / match_global_chunks)
# ============================================================
class TestVectorRpcPlumbing:
    @pytest.mark.parametrize("agent_tags", [None, []])
    def test_p_agent_tags_omitted_when_no_filter(self, service_factory, agent_tags):
        """
        Expected input:  agent_tags None or [] (what risk/policies always send).
        Expected output: neither RPC payload contains the p_agent_tags key, so
                         the request is byte-compatible with a database where
                         migration 009 has not been applied.
        """
        db = FakeDB({"match_chunks": [make_row("c1", 0.9)]})
        service_factory(db).retrieve(
            tenant_id="acme", query="test", agent_tags=agent_tags
        )

        assert "p_agent_tags" not in db.args_for("match_chunks")[0]
        assert "p_agent_tags" not in db.args_for("match_global_chunks")[0]

    def test_p_agent_tags_sent_when_filter_present(self, service_factory):
        """
        Expected input:  agent_tags=["normativa"].
        Expected output: both the tenant and the global vector RPC receive
                         p_agent_tags=["normativa"].
        """
        db = FakeDB({"match_chunks": [make_row("c1", 0.9)]})
        service_factory(db).retrieve(
            tenant_id="acme", query="test", agent_tags=["normativa"]
        )

        assert db.agent_tag_args("match_chunks") == [["normativa"]]
        assert db.agent_tag_args("match_global_chunks") == [["normativa"]]

    def test_rpc_failure_with_agent_tags_raises_with_migration_hint(
        self, service_factory
    ):
        """
        Expected input:  agent filter requested, but the RPC rejects the call
                         (what a database missing migration 009 does).
        Expected output: UpstreamError naming migration 009 — never a silent
                         degradation to an unfiltered search, which would
                         return the whole corpus while the caller believes it
                         is reading one agent's slice.
        """
        db = FakeDB(error=RuntimeError("function match_chunks(...) does not exist"))
        with pytest.raises(UpstreamError) as excinfo:
            service_factory(db).retrieve(
                tenant_id="acme", query="test", agent_tags=["normativa"]
            )

        assert "009_agent_tags.sql" in str(excinfo.value)

    def test_global_rpc_failure_without_agent_tags_stays_non_fatal(
        self, service_factory
    ):
        """
        Expected input:  no agent filter, and the global library RPC fails.
        Expected output: the pre-existing non-fatal policy is preserved — the
                         tenant results still come back. (Here the tenant RPC
                         fails too, so the assertion is that the error raised
                         is the tenant one, with no migration-009 hint.)
        """
        db = FakeDB(error=RuntimeError("boom"))
        with pytest.raises(UpstreamError) as excinfo:
            service_factory(db).retrieve(tenant_id="acme", query="test")

        assert "009_agent_tags.sql" not in str(excinfo.value)


# ============================================================
# 2. Plumbing — hybrid path (match_chunks_hybrid / match_global_chunks_hybrid)
# ============================================================
class TestHybridRpcPlumbing:
    @pytest.mark.parametrize("agent_tags", [None, []])
    def test_p_agent_tags_omitted_when_no_filter(self, service_factory, agent_tags):
        db = FakeDB({"match_chunks_hybrid": [make_row("c1", 0.9)]})
        service_factory(db).retrieve(
            tenant_id="acme", query="test", use_hybrid=True, agent_tags=agent_tags
        )

        assert "p_agent_tags" not in db.args_for("match_chunks_hybrid")[0]
        assert "p_agent_tags" not in db.args_for("match_global_chunks_hybrid")[0]

    def test_p_agent_tags_sent_when_filter_present(self, service_factory):
        db = FakeDB({"match_chunks_hybrid": [make_row("c1", 0.9)]})
        service_factory(db).retrieve(
            tenant_id="acme",
            query="test",
            use_hybrid=True,
            agent_tags=["normativa"],
        )

        assert db.agent_tag_args("match_chunks_hybrid") == [["normativa"]]
        assert db.agent_tag_args("match_global_chunks_hybrid") == [["normativa"]]

    def test_hybrid_failure_falls_back_to_vector_keeping_agent_filter(
        self, service_factory, monkeypatch
    ):
        """
        Expected input:  hybrid enabled with an agent filter, and the hybrid
                         RPCs are unavailable (migration 006 not applied).
        Expected output: BOTH the tenant and the global search degrade to their
                         vector-only RPC, and both still carry p_agent_tags.
                         Losing the filter on the fallback path would quietly
                         widen the search back to the full corpus; failing hard
                         instead of degrading would make migration 006 a
                         prerequisite for agent queries, which it is not.
        """

        class SelectiveFailDB(FakeDB):
            def rpc(self, name: str, args: dict):
                self.calls.append((name, args))
                if name.endswith("_hybrid"):
                    return _FakeExecute([], RuntimeError("hybrid rpc missing"))
                return _FakeExecute(self.rows_by_rpc.get(name, []))

        db = SelectiveFailDB({"match_chunks": [make_row("c1", 0.9)]})
        service_factory(db).retrieve(
            tenant_id="acme",
            query="test",
            use_hybrid=True,
            agent_tags=["normativa"],
        )

        assert db.agent_tag_args("match_chunks") == [["normativa"]]
        assert db.agent_tag_args("match_global_chunks") == [["normativa"]]


# ============================================================
# 3. The "general" fallback
# ============================================================
class TestGeneralFallback:
    def test_fallback_triggers_on_low_top1_score(self, service_factory, monkeypatch):
        """
        Expected input:  agent_tags=["normativa"], and the agent-scoped pass
                         returns a top-1 score below agent_fallback_threshold.
        Expected logic:  retrieve() re-runs the pipeline with the UNION
                         ["normativa", "general"] — the agent's own tag must
                         remain in play, so a relevant agent chunk cannot be
                         displaced by generic material.
        Expected output: two rounds of RPC calls, the second one carrying the
                         broadened tag set, and the returned chunks are the
                         second pass's.
        """
        # Threshold above the pass-1 score, below the pass-2 score.
        monkeypatch.setattr(settings, "agent_fallback_threshold", 0.80)
        monkeypatch.setattr(settings, "similarity_threshold", 0.10)

        class TwoPassDB(FakeDB):
            """Returns a weak row while scoped, a strong row once broadened."""

            def rpc(self, name: str, args: dict):
                self.calls.append((name, args))
                if name != "match_chunks":
                    return _FakeExecute([])
                tags = args.get("p_agent_tags") or []
                rows = (
                    [make_row("strong", 0.95)]
                    if "general" in tags
                    else [make_row("weak", 0.50)]
                )
                return _FakeExecute(rows)

        db = TwoPassDB()
        result = service_factory(db).retrieve(
            tenant_id="acme", query="test", agent_tags=["normativa"]
        )

        assert db.agent_tag_args("match_chunks") == [
            ["normativa"],
            ["normativa", "general"],
        ]
        assert [c.chunk_id for c in result] == ["strong"]

    def test_fallback_not_triggered_on_good_coverage(
        self, service_factory, monkeypatch
    ):
        """
        Expected input:  a top-1 score at or above the threshold.
        Expected output: exactly one pass; the agent's own corpus answered the
                         question and is not diluted with general material.
        """
        monkeypatch.setattr(settings, "agent_fallback_threshold", 0.45)
        monkeypatch.setattr(settings, "similarity_threshold", 0.10)

        db = FakeDB({"match_chunks": [make_row("c1", 0.90)]})
        result = service_factory(db).retrieve(
            tenant_id="acme", query="test", agent_tags=["normativa"]
        )

        assert db.agent_tag_args("match_chunks") == [["normativa"]]
        assert [c.chunk_id for c in result] == ["c1"]

    def test_empty_pass1_triggers_fallback(self, service_factory, monkeypatch):
        """
        Expected input:  the agent-scoped pass returns nothing at all.
        Expected output: that is the worst possible coverage (scored 0.0), so
                         the fallback must fire.
        """
        monkeypatch.setattr(settings, "agent_fallback_threshold", 0.45)

        db = FakeDB()  # every RPC returns zero rows
        service_factory(db).retrieve(
            tenant_id="acme", query="test", agent_tags=["normativa"]
        )

        assert db.agent_tag_args("match_chunks") == [
            ["normativa"],
            ["normativa", "general"],
        ]

    def test_fallback_disabled_by_flag(self, service_factory, monkeypatch):
        """
        Expected input:  a weak result but enable_general_fallback=False.
        Expected output: one pass only. This is the per-agent switch the design
                         requires for agents whose own corpus is dense enough
                         that broadening would only add noise.
        """
        monkeypatch.setattr(settings, "agent_fallback_threshold", 0.99)

        db = FakeDB({"match_chunks": [make_row("c1", 0.50)]})
        service_factory(db).retrieve(
            tenant_id="acme",
            query="test",
            agent_tags=["normativa"],
            enable_general_fallback=False,
        )

        assert db.agent_tag_args("match_chunks") == [["normativa"]]

    def test_no_retry_when_general_already_requested(
        self, service_factory, monkeypatch
    ):
        """
        Expected input:  agent_tags already contains "general".
        Expected output: no second pass. Broadening would produce the identical
                         tag set and therefore the identical query — pure waste.
        """
        monkeypatch.setattr(settings, "agent_fallback_threshold", 0.99)

        db = FakeDB({"match_chunks": [make_row("c1", 0.50)]})
        service_factory(db).retrieve(
            tenant_id="acme", query="test", agent_tags=["normativa", "general"]
        )

        assert db.agent_tag_args("match_chunks") == [["normativa", "general"]]

    def test_embeddings_computed_once_across_both_passes(
        self, service_factory, monkeypatch, _stub_pipeline_externals
    ):
        """
        Expected input:  a fallback-triggering query with one query variant.
        Expected output: embed_single is called exactly once. The two passes
                         send identical query text, so re-embedding for pass 2
                         would be a duplicated paid API call for an identical
                         vector.
        """
        monkeypatch.setattr(settings, "agent_fallback_threshold", 0.99)

        db = FakeDB({"match_chunks": [make_row("c1", 0.50)]})
        service_factory(db).retrieve(
            tenant_id="acme", query="test", agent_tags=["normativa"]
        )

        # Two passes ran...
        assert len(db.agent_tag_args("match_chunks")) == 2
        # ...on a single embedding.
        assert _stub_pipeline_externals["embed"] == 1

    def test_no_agent_tags_means_single_unfiltered_pass(self, service_factory, monkeypatch):
        """
        Control case: without agent_tags the fallback machinery is inert, even
        when the score is far below agent_fallback_threshold. This is the
        regression guard for every pre-existing caller.
        """
        monkeypatch.setattr(settings, "agent_fallback_threshold", 0.99)
        monkeypatch.setattr(settings, "similarity_threshold", 0.10)

        db = FakeDB({"match_chunks": [make_row("c1", 0.50)]})
        result = service_factory(db).retrieve(tenant_id="acme", query="test")

        assert db.agent_tag_args("match_chunks") == [None]
        assert [c.chunk_id for c in result] == ["c1"]


# ============================================================
# 4. Candidate-pool narrowing
# ============================================================
def test_agent_filter_narrows_candidate_pool(service_factory, monkeypatch):
    """
    The checklist item "the filter effectively reduces retrieval candidates",
    expressed as a unit test: with the same query, an agent-scoped run must see
    no more candidates than an unscoped one, because the SQL predicate
    `agent_tags && p_agent_tags` can only remove rows.
    """
    monkeypatch.setattr(settings, "similarity_threshold", 0.10)
    # High enough that the scoped pass would normally broaden; disabled below
    # so the comparison stays between exactly one scoped and one unscoped pass.
    monkeypatch.setattr(settings, "agent_fallback_threshold", 0.0)

    class NarrowingDB(FakeDB):
        """Two chunks in the corpus; only one carries the 'normativa' tag."""

        def rpc(self, name: str, args: dict):
            self.calls.append((name, args))
            if name != "match_chunks":
                return _FakeExecute([])
            if args.get("p_agent_tags"):
                return _FakeExecute([make_row("tagged", 0.90)])
            return _FakeExecute([make_row("tagged", 0.90), make_row("untagged", 0.80)])

    unscoped = NarrowingDB()
    unscoped_result = service_factory(unscoped).retrieve(
        tenant_id="acme", query="test", top_k=10
    )

    scoped = NarrowingDB()
    scoped_result = service_factory(scoped).retrieve(
        tenant_id="acme", query="test", top_k=10, agent_tags=["normativa"]
    )

    assert len(scoped_result) < len(unscoped_result)
    assert {c.chunk_id for c in scoped_result} == {"tagged"}


# ============================================================
# 5. Industry filter (migration 010)
#
# Same design as the agent filter: the predicate runs in SQL, never as a
# Python post-filter. These tests exist because the previous, Python-side
# implementation passed its own unit tests while filtering nothing in
# production — so "the parameter is sent" is asserted separately from "the
# wrong-sector chunk does not come back".
# ============================================================
class TestIndustryRpcPlumbing:
    def test_p_industry_codes_omitted_when_no_industry(self, service_factory):
        """
        Expected input:  no industry (what risk/policies always send).
        Expected output: the key is absent from both vector RPC payloads, so
                         the request stays valid against a pre-010 database.
        """
        db = FakeDB({"match_chunks": [make_row("c1", 0.9)]})
        service_factory(db).retrieve(tenant_id="acme", query="test")

        assert "p_industry_codes" not in db.args_for("match_chunks")[0]
        assert "p_industry_codes" not in db.args_for("match_global_chunks")[0]

    def test_p_industry_codes_sent_as_single_element_array(self, service_factory):
        """
        Expected input:  industry="bank" (a scalar).
        Expected output: the RPCs receive ["bank"] — the column is a TEXT[]
                         and the predicate is array overlap.
        """
        db = FakeDB({"match_chunks": [make_row("c1", 0.9)]})
        service_factory(db).retrieve(tenant_id="acme", query="test", industry="bank")

        assert db.industry_args("match_chunks") == [["bank"]]
        assert db.industry_args("match_global_chunks") == [["bank"]]

    def test_p_industry_codes_sent_on_hybrid_path(self, service_factory):
        db = FakeDB({"match_chunks_hybrid": [make_row("c1", 0.9)]})
        service_factory(db).retrieve(
            tenant_id="acme", query="test", industry="bank", use_hybrid=True
        )

        assert db.industry_args("match_chunks_hybrid") == [["bank"]]
        assert db.industry_args("match_global_chunks_hybrid") == [["bank"]]

    def test_hybrid_failure_falls_back_to_vector_keeping_industry_filter(
        self, service_factory
    ):
        """
        The hybrid→vector degradation must stay symmetric for BOTH branches,
        exactly as it does for agent_tags. Dropping the industry filter while
        degrading would silently widen the search back to the whole corpus.
        """

        class SelectiveFailDB(FakeDB):
            def rpc(self, name: str, args: dict):
                self.calls.append((name, args))
                if name.endswith("_hybrid"):
                    return _FakeExecute([], RuntimeError("hybrid rpc missing"))
                return _FakeExecute(self.rows_by_rpc.get(name, []))

        db = SelectiveFailDB({"match_chunks": [make_row("c1", 0.9)]})
        service_factory(db).retrieve(
            tenant_id="acme", query="test", industry="bank", use_hybrid=True
        )

        assert db.industry_args("match_chunks") == [["bank"]]
        assert db.industry_args("match_global_chunks") == [["bank"]]

    def test_rpc_failure_with_industry_raises_with_migration_hint(
        self, service_factory
    ):
        """
        A pre-010 database rejects the p_industry_codes argument. That must
        surface as a loud error naming migration 010 — degrading to an
        unfiltered search is what produced the original bug.
        """
        db = FakeDB(error=RuntimeError("function match_chunks(...) does not exist"))
        with pytest.raises(UpstreamError) as excinfo:
            service_factory(db).retrieve(
                tenant_id="acme", query="test", industry="bank"
            )

        assert "010_industry_filter_fix.sql" in str(excinfo.value)

    def test_both_filters_sent_together(self, service_factory):
        """Agent and industry filters are independent and compose."""
        db = FakeDB({"match_chunks": [make_row("c1", 0.9)]})
        service_factory(db).retrieve(
            tenant_id="acme",
            query="test",
            industry="bank",
            agent_tags=["normativa"],
            enable_general_fallback=False,
        )

        args = db.args_for("match_chunks")[0]
        assert args["p_agent_tags"] == ["normativa"]
        assert args["p_industry_codes"] == ["bank"]


class TestIndustryExclusion:
    """
    Real exclusion, not just parameter passing. The FakeDB below honours
    p_industry_codes the way migration 010 does, so these assert on the
    chunks that actually come back.
    """

    class _IndustryAwareDB(FakeDB):
        """
        Two-document corpus:
          iso        → industry_codes ['ALL']              (Layer 1-2)
          sugef_bank → industry_codes ['bank']             (Layer 3)

        The RPC applies the permissive rule from migration 010: 'ALL' is
        always visible, a sector-tagged chunk only to its own sectors.
        """

        CORPUS = [
            ("iso", 0.90, ["ALL"]),
            ("sugef_bank", 0.85, ["bank"]),
        ]

        def rpc(self, name: str, args: dict):
            self.calls.append((name, args))
            if name != "match_chunks":
                return _FakeExecute([])
            requested = args.get("p_industry_codes")
            rows = [
                make_row(cid, sim, industry_codes=codes)
                for cid, sim, codes in self.CORPUS
                if not requested
                or "ALL" in codes
                or set(codes) & set(requested)
            ]
            return _FakeExecute(rows)

    def test_wrong_sector_chunk_is_excluded(self, service_factory, monkeypatch):
        """
        Expected input:  a cooperative institution querying a corpus that
                         contains a bank-only SUGEF document.
        Expected output: the bank-only chunk does NOT come back. This is the
                         assertion the old post-filter could never have made
                         truthfully.
        """
        monkeypatch.setattr(settings, "similarity_threshold", 0.10)

        db = self._IndustryAwareDB()
        result = service_factory(db).retrieve(
            tenant_id="acme", query="test", top_k=10, industry="cooperative"
        )

        assert [c.chunk_id for c in result] == ["iso"]

    def test_matching_sector_chunk_is_included(self, service_factory, monkeypatch):
        monkeypatch.setattr(settings, "similarity_threshold", 0.10)

        db = self._IndustryAwareDB()
        result = service_factory(db).retrieve(
            tenant_id="acme", query="test", top_k=10, industry="bank"
        )

        assert {c.chunk_id for c in result} == {"iso", "sugef_bank"}

    def test_no_industry_returns_full_corpus(self, service_factory, monkeypatch):
        """
        The fix is additive: without an industry filter the previous behaviour
        (everything is returned) is preserved exactly.
        """
        monkeypatch.setattr(settings, "similarity_threshold", 0.10)

        db = self._IndustryAwareDB()
        result = service_factory(db).retrieve(
            tenant_id="acme", query="test", top_k=10
        )

        assert {c.chunk_id for c in result} == {"iso", "sugef_bank"}

    def test_industry_codes_survive_to_chunk_results(
        self, service_factory, monkeypatch
    ):
        """
        The column must reach ChunkResult. Not mapping it is precisely what
        made the old post-filter a no-op, and it is what lets
        _warn_on_industry_mismatch detect a filter that failed to apply.
        """
        monkeypatch.setattr(settings, "similarity_threshold", 0.10)

        db = self._IndustryAwareDB()
        result = service_factory(db).retrieve(
            tenant_id="acme", query="test", top_k=10, industry="bank"
        )

        assert all(c.industry_codes is not None for c in result)

    def test_industry_codes_survive_the_hybrid_path(self, service_factory, monkeypatch):
        """
        Same guarantee on the hybrid path, where reciprocal_rank_fusion builds
        the ChunkResult objects instead of RetrievalService.
        """
        monkeypatch.setattr(settings, "similarity_threshold", 0.10)

        db = FakeDB(
            {
                "match_chunks_hybrid": [
                    {
                        **make_row("c1", 0.9, industry_codes=["bank"]),
                        "vector_rank": 1,
                        "keyword_rank": 1,
                    }
                ]
            }
        )
        result = service_factory(db).retrieve(
            tenant_id="acme", query="test", top_k=10, industry="bank", use_hybrid=True
        )

        assert [c.industry_codes for c in result] == [["bank"]]


# ============================================================
# 6. Diversity cap: strict for broad search, permissive when agent-scoped
#
# retrieval_max_chunks_per_document=1 protects a heterogeneous whole-corpus
# search from being monopolised by one long document. Once agent_tags narrows
# the pool to a single domain, that same cap starves the answer of depth: a
# question specifically about ISO 42001 could only ever receive ONE ISO chunk,
# with the remaining top-k slots going to adjacent frameworks. That is what
# produced grounding=0.045 / hallucination=True on the Normativa validation run.
# ============================================================
class TestDiversityCapRegimes:
    @staticmethod
    def _skewed_corpus() -> list[dict]:
        """
        One dominant document plus enough distinct documents to fill top_k.

        The distinct-document padding matters: apply_per_document_cap does not
        DROP overflow chunks, it moves them behind the capped ones. With too few
        documents the overflow would flow straight back into the free top-k
        slots and both cap values would look identical — which is precisely the
        shape real retrieval has (dozens of candidate documents), so the test
        must reproduce it to be meaningful.
        """
        rows = [
            make_row(f"dominant-{i}", 0.99 - i * 0.01, document_id="doc-dominant")
            for i in range(4)
        ]
        rows += [
            make_row(f"other-{name}", sim, document_id=f"doc-{name}")
            for name, sim in zip("bcdefg", [0.95, 0.94, 0.93, 0.92, 0.91, 0.90])
        ]
        return rows

    @staticmethod
    def _count_from_dominant(chunks) -> int:
        return sum(1 for c in chunks if c.document_id == "doc-dominant")

    def test_agent_scoped_pass_uses_permissive_cap(self, service_factory, monkeypatch):
        """
        Expected input:  agent_tags set, one document holding the 4 best chunks.
        Expected output: up to agent_max_chunks_per_document (3) of the top-k
                         slots come from that document, so the LLM gets real
                         depth on the relevant source.
        """
        monkeypatch.setattr(settings, "similarity_threshold", 0.10)
        monkeypatch.setattr(settings, "agent_max_chunks_per_document", 3)
        monkeypatch.setattr(settings, "retrieval_max_chunks_per_document", 1)

        db = FakeDB({"match_chunks": self._skewed_corpus()})
        result = service_factory(db).retrieve(
            tenant_id="acme",
            query="test",
            top_k=5,
            agent_tags=["normativa"],
            enable_general_fallback=False,
        )

        assert len(result) == 5
        assert self._count_from_dominant(result) == 3

    def test_unscoped_pass_keeps_strict_cap(self, service_factory, monkeypatch):
        """
        The regression guard for every existing caller (generic chat,
        risk/service.py, policies/service.py): without agent_tags the cap must
        stay at 1, giving k distinct documents exactly as before this change.
        """
        monkeypatch.setattr(settings, "similarity_threshold", 0.10)
        monkeypatch.setattr(settings, "agent_max_chunks_per_document", 3)
        monkeypatch.setattr(settings, "retrieval_max_chunks_per_document", 1)

        db = FakeDB({"match_chunks": self._skewed_corpus()})
        result = service_factory(db).retrieve(
            tenant_id="acme", query="test", top_k=5
        )

        assert len(result) == 5
        assert self._count_from_dominant(result) == 1
        # 5 slots, 5 different documents.
        assert len({c.document_id for c in result}) == 5

    @pytest.mark.parametrize("cap,expected", [(1, 1), (2, 2), (4, 4)])
    def test_permissive_cap_value_is_honoured(
        self, service_factory, monkeypatch, cap, expected
    ):
        """The agent-scoped branch reads agent_max_chunks_per_document, not the
        general setting — including when it is configured back down to 1."""
        monkeypatch.setattr(settings, "similarity_threshold", 0.10)
        monkeypatch.setattr(settings, "agent_max_chunks_per_document", cap)
        monkeypatch.setattr(settings, "retrieval_max_chunks_per_document", 1)

        db = FakeDB({"match_chunks": self._skewed_corpus()})
        result = service_factory(db).retrieve(
            tenant_id="acme",
            query="test",
            top_k=5,
            agent_tags=["normativa"],
            enable_general_fallback=False,
        )

        assert self._count_from_dominant(result) == expected

    def test_fallback_pass_also_uses_permissive_cap(self, service_factory, monkeypatch):
        """
        Pass 2 receives the broadened tag list, which is still non-empty, so it
        must stay in the agent-scoped regime. Reverting to cap=1 on the fallback
        would silently undo the fix for exactly the low-coverage queries that
        need depth most.
        """
        monkeypatch.setattr(settings, "similarity_threshold", 0.10)
        monkeypatch.setattr(settings, "agent_max_chunks_per_document", 3)
        monkeypatch.setattr(settings, "retrieval_max_chunks_per_document", 1)
        # Force the fallback so the returned chunks come from pass 2.
        monkeypatch.setattr(settings, "agent_fallback_threshold", 0.999)

        db = FakeDB({"match_chunks": self._skewed_corpus()})
        result = service_factory(db).retrieve(
            tenant_id="acme", query="test", top_k=5, agent_tags=["normativa"]
        )

        assert db.agent_tag_args("match_chunks") == [
            ["normativa"],
            ["normativa", "general"],
        ]
        assert self._count_from_dominant(result) == 3
