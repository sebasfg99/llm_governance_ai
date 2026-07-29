# ============================================================
# app/tests/test_retrieval_industry.py
#
# Industry visibility rules for Layer-3 sector documents.
#
# HISTORY — WHY THIS FILE CHANGED SHAPE:
# These tests used to exercise `RetrievalService._filter_by_industry`, a
# Python post-filter. That post-filter passed every test here and still let
# wrong-sector content through in production, because the RPCs never returned
# the `industry_codes` column: the field arrived as None for every chunk and
# the permissive "no codes means applies to everyone" branch accepted
# everything. The tests were green because they constructed chunks WITH codes;
# reality never did.
#
# Migration 010_industry_filter_fix.sql moved the filtering into SQL. What
# remains in Python is the rule itself (`chunk_matches_industry`), used to
# verify after the fact that SQL did its job. These tests now pin that rule
# down; end-to-end plumbing of `p_industry_codes` is covered in
# test_agent_tags_retrieval.py.
# ============================================================

import pytest

from app.domain.retrieval.ranking import ChunkResult
from app.domain.retrieval.service import RetrievalService


def make_chunk(chunk_id: str, industry_codes: list[str] | None) -> ChunkResult:
    return ChunkResult(
        chunk_id=chunk_id,
        document_id="doc-1",
        tenant_id="global",
        content="test content",
        chunk_index=0,
        similarity=0.9,
        section_ref=None,
        industry_codes=industry_codes,
    )


@pytest.fixture
def service():
    # Both the predicate and the sanity check are pure — no DB needed.
    return RetrievalService.__new__(RetrievalService)


class TestChunkMatchesIndustry:
    """
    The visibility rule, which is the Python mirror of the SQL predicate in
    migration 010. Permissive by design: cross-sector material must never be
    excluded, only sector-restricted material may be.
    """

    def test_all_tag_always_visible(self, service):
        assert service.chunk_matches_industry(["ALL"], "cooperative") is True

    def test_none_codes_treated_as_all(self, service):
        # Safety net for rows that predate migration 008.
        assert service.chunk_matches_industry(None, "cooperative") is True

    def test_empty_codes_treated_as_all(self, service):
        assert service.chunk_matches_industry([], "cooperative") is True

    def test_matching_industry_visible(self, service):
        assert (
            service.chunk_matches_industry(["bank", "cooperative"], "cooperative")
            is True
        )

    def test_non_matching_industry_hidden(self, service):
        assert service.chunk_matches_industry(["bank"], "cooperative") is False

    def test_no_industry_requested_shows_everything(self, service):
        assert service.chunk_matches_industry(["bank"], None) is True

    def test_insurance_excluded_from_sector_docs(self, service):
        assert (
            service.chunk_matches_industry(["bank", "cooperative"], "insurance")
            is False
        )

    def test_fintech_excluded_from_sector_docs(self, service):
        assert (
            service.chunk_matches_industry(["bank", "cooperative"], "fintech")
            is False
        )

    def test_layered_corpus_for_cooperative(self, service):
        """
        The realistic mix: Layer 1-2 frameworks are always visible, Layer 3
        sector documents only to the sectors they name.
        """
        corpus = {
            "iso": ["ALL"],                        # Layer 1-2 → visible
            "nist": None,                          # Layer 1-2 → visible
            "sugef": ["bank", "cooperative"],      # Layer 3 → visible (matches)
            "conassif_bank": ["bank"],             # Layer 3 → hidden
        }
        visible = {
            name
            for name, codes in corpus.items()
            if service.chunk_matches_industry(codes, "cooperative")
        }
        assert visible == {"iso", "nist", "sugef"}


class TestWarnOnIndustryMismatch:
    """
    The sanity check must REPORT without filtering. Anything that drops chunks
    here would re-introduce a second source of truth alongside SQL, which is
    exactly how the original bug hid for so long.
    """

    def test_warns_when_sql_filter_did_not_apply(self, service, caplog):
        """
        Expected input:  a chunk tagged bank-only came back for a cooperative.
        Expected output: a WARNING naming migration 010 — and the chunk list is
                         untouched, since this method returns None by design.
        """
        chunks = [make_chunk("bank_only", ["bank"])]
        with caplog.at_level("WARNING"):
            assert service._warn_on_industry_mismatch(chunks, "cooperative") is None

        assert "Industry filter mismatch" in caplog.text
        assert "migration 010" in caplog.text

    def test_silent_when_filter_worked(self, service, caplog):
        chunks = [make_chunk("iso", ["ALL"]), make_chunk("sugef", ["cooperative"])]
        with caplog.at_level("WARNING"):
            service._warn_on_industry_mismatch(chunks, "cooperative")

        assert "Industry filter mismatch" not in caplog.text

    def test_silent_when_no_industry_requested(self, service, caplog):
        chunks = [make_chunk("bank_only", ["bank"])]
        with caplog.at_level("WARNING"):
            service._warn_on_industry_mismatch(chunks, None)

        assert "Industry filter mismatch" not in caplog.text

    def test_absent_column_is_not_reported_as_mismatch(self, service, caplog):
        """
        industry_codes=None means the database predates migration 010 and is
        not returning the column at all. That is a different situation from
        the filter returning wrong rows, and reporting it here would produce a
        warning on every single query against an un-migrated database.
        """
        chunks = [make_chunk("unknown", None)]
        with caplog.at_level("WARNING"):
            service._warn_on_industry_mismatch(chunks, "cooperative")

        assert "Industry filter mismatch" not in caplog.text
