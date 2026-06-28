"""Unit tests for industry post-filter in RetrievalService."""
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
    # _filter_by_industry is a pure method — no DB needed.
    return RetrievalService.__new__(RetrievalService)


class TestFilterByIndustry:
    def test_all_tag_always_included(self, service):
        chunk = make_chunk("c1", ["ALL"])
        result = service._filter_by_industry([chunk], "cooperative")
        assert len(result) == 1

    def test_none_codes_treated_as_all(self, service):
        chunk = make_chunk("c1", None)
        result = service._filter_by_industry([chunk], "cooperative")
        assert len(result) == 1

    def test_empty_codes_treated_as_all(self, service):
        chunk = make_chunk("c1", [])
        result = service._filter_by_industry([chunk], "cooperative")
        assert len(result) == 1

    def test_matching_industry_included(self, service):
        chunk = make_chunk("c1", ["bank", "cooperative"])
        result = service._filter_by_industry([chunk], "cooperative")
        assert len(result) == 1

    def test_non_matching_industry_excluded(self, service):
        chunk = make_chunk("c1", ["bank"])
        result = service._filter_by_industry([chunk], "cooperative")
        assert len(result) == 0

    def test_no_industry_filter_returns_all(self, service):
        chunks = [
            make_chunk("c1", ["bank"]),
            make_chunk("c2", ["cooperative"]),
            make_chunk("c3", ["ALL"]),
            make_chunk("c4", None),
        ]
        result = service._filter_by_industry(chunks, None)
        assert len(result) == 4

    def test_mixed_pool_cooperative(self, service):
        chunks = [
            make_chunk("iso",    ["ALL"]),         # Layer 1-2 → always included
            make_chunk("nist",   None),             # Layer 1-2 → always included
            make_chunk("sugef",  ["bank", "cooperative"]),  # Layer 3 → included (matches)
            make_chunk("conassif_bank", ["bank"]),  # Layer 3 → excluded (bank only)
        ]
        result = service._filter_by_industry(chunks, "cooperative")
        ids = {c.chunk_id for c in result}
        assert "iso" in ids
        assert "nist" in ids
        assert "sugef" in ids
        assert "conassif_bank" not in ids

    def test_bank_sees_bank_docs(self, service):
        chunks = [
            make_chunk("sugef",  ["bank", "cooperative"]),
            make_chunk("bank_only", ["bank"]),
        ]
        result = service._filter_by_industry(chunks, "bank")
        assert len(result) == 2

    def test_insurance_excluded_from_sector_docs(self, service):
        chunk = make_chunk("sugef", ["bank", "cooperative"])
        result = service._filter_by_industry([chunk], "insurance")
        assert len(result) == 0

    def test_fintech_excluded_from_sector_docs(self, service):
        chunk = make_chunk("sugef", ["bank", "cooperative"])
        result = service._filter_by_industry([chunk], "fintech")
        assert len(result) == 0

    def test_order_preserved(self, service):
        chunks = [
            make_chunk("a", ["ALL"]),
            make_chunk("b", ["bank", "cooperative"]),
            make_chunk("c", ["ALL"]),
        ]
        result = service._filter_by_industry(chunks, "cooperative")
        assert [c.chunk_id for c in result] == ["a", "b", "c"]
