# ============================================================
# app/domain/locale/resolver.py — Country + industry resolution
#
# Mirrors supabase/functions/_shared/locale-resolver.ts.
# Both must be kept in sync when adding new countries or
# industry codes.
# ============================================================

# ISO 3166-1 alpha-2 → URL slug. Must match src/lib/country-slug-map.ts.
COUNTRY_CODE_TO_SLUG: dict[str, str] = {
    "CR": "costa-rica",
    "PA": "panama",
    "DO": "republica-dominicana",
    "SV": "el-salvador",
    "GT": "guatemala",
    "HN": "honduras",
    "NI": "nicaragua",
}

VALID_INDUSTRY_CODES: frozenset[str] = frozenset(
    {"bank", "cooperative", "insurance", "fintech", "other"}
)

# Spanish/English labels → canonical code
_INDUSTRY_LABEL_MAP: dict[str, str] = {
    "banco": "bank",
    "bank": "bank",
    "cooperativa": "cooperative",
    "cooperative": "cooperative",
    "aseguradora": "insurance",
    "insurance": "insurance",
    "fintech": "fintech",
    "otro": "other",
    "other": "other",
}

# Values that mean "no specific country" — must never be used as a country slug
_REJECT_COUNTRY_VALUES: frozenset[str] = frozenset({"ALL", "GLOBAL", ""})


def resolve_country_slug(value: str | None) -> str | None:
    """
    Normalize an ISO country code or full name to a URL slug.

    Returns None when value is missing, globally-scoped ("ALL"), or unrecognized.
    Never raises — unknown input returns None and should be logged by the caller.

    Examples:
        "CR"          → "costa-rica"
        "PA"          → "panama"
        "ALL"         → None  (global scope, not a valid institution country)
        ""            → None
        None          → None
        "costa-rica"  → None  (slugs are not valid input — pass the ISO code)
    """
    if not value:
        return None
    upper = value.strip().upper()
    if upper in _REJECT_COUNTRY_VALUES:
        return None
    return COUNTRY_CODE_TO_SLUG.get(upper)


def resolve_industry_code(value: str | None) -> str | None:
    """
    Normalize an industry label or code to a canonical code.

    Returns None when value is missing or unrecognized.
    Never raises.

    Examples:
        "cooperative"  → "cooperative"
        "Cooperativa"  → "cooperative"
        "bank"         → "bank"
        "Banco"        → "bank"
        "invalid"      → None
        None           → None
    """
    if not value:
        return None
    normalized = value.strip().lower()
    return _INDUSTRY_LABEL_MAP.get(normalized)
