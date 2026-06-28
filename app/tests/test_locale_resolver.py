"""Unit tests for app.domain.locale.resolver."""
import pytest
from app.domain.locale.resolver import resolve_country_slug, resolve_industry_code


class TestResolveCountrySlug:
    def test_valid_iso_code_cr(self):
        assert resolve_country_slug("CR") == "costa-rica"

    def test_valid_iso_code_pa(self):
        assert resolve_country_slug("PA") == "panama"

    def test_valid_iso_code_do(self):
        assert resolve_country_slug("DO") == "republica-dominicana"

    def test_valid_iso_code_sv(self):
        assert resolve_country_slug("SV") == "el-salvador"

    def test_valid_iso_code_gt(self):
        assert resolve_country_slug("GT") == "guatemala"

    def test_valid_iso_code_hn(self):
        assert resolve_country_slug("HN") == "honduras"

    def test_valid_iso_code_ni(self):
        assert resolve_country_slug("NI") == "nicaragua"

    def test_lowercase_code(self):
        assert resolve_country_slug("cr") == "costa-rica"

    def test_all_returns_none(self):
        assert resolve_country_slug("ALL") is None

    def test_global_returns_none(self):
        assert resolve_country_slug("GLOBAL") is None

    def test_empty_string_returns_none(self):
        assert resolve_country_slug("") is None

    def test_none_returns_none(self):
        assert resolve_country_slug(None) is None

    def test_unknown_code_returns_none(self):
        assert resolve_country_slug("US") is None

    def test_slug_not_valid_input(self):
        # Slugs like "costa-rica" are not ISO codes → None
        assert resolve_country_slug("costa-rica") is None

    def test_full_name_not_valid_input(self):
        # Full names are not ISO codes → None
        assert resolve_country_slug("Costa Rica") is None


class TestResolveIndustryCode:
    def test_canonical_code_bank(self):
        assert resolve_industry_code("bank") == "bank"

    def test_canonical_code_cooperative(self):
        assert resolve_industry_code("cooperative") == "cooperative"

    def test_canonical_code_insurance(self):
        assert resolve_industry_code("insurance") == "insurance"

    def test_canonical_code_fintech(self):
        assert resolve_industry_code("fintech") == "fintech"

    def test_canonical_code_other(self):
        assert resolve_industry_code("other") == "other"

    def test_spanish_label_banco(self):
        assert resolve_industry_code("Banco") == "bank"

    def test_spanish_label_cooperativa(self):
        assert resolve_industry_code("Cooperativa") == "cooperative"

    def test_spanish_label_aseguradora(self):
        assert resolve_industry_code("Aseguradora") == "insurance"

    def test_spanish_label_otro(self):
        assert resolve_industry_code("Otro") == "other"

    def test_case_insensitive(self):
        assert resolve_industry_code("BANK") == "bank"
        assert resolve_industry_code("COOPERATIVE") == "cooperative"

    def test_none_returns_none(self):
        assert resolve_industry_code(None) is None

    def test_empty_string_returns_none(self):
        assert resolve_industry_code("") is None

    def test_unknown_value_returns_none(self):
        assert resolve_industry_code("invalid") is None

    def test_whitespace_stripped(self):
        assert resolve_industry_code("  bank  ") == "bank"
