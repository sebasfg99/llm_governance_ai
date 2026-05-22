# ============================================================
# app/domain/llm/pii_detector.py — Lightweight PII detector
#
# WHY THIS FILE EXISTS:
# Response-level guardrails must block answers that leak
# personally identifiable information that the LLM may have
# extracted (or invented) from the retrieved context. We use
# regex patterns covering the regions our customers operate
# in (CR, MX, US-LATAM) so we have zero external dependencies.
#
# WHAT THIS DETECTS:
#   - Email addresses
#   - International + Costa Rica + Mexico phone numbers
#   - Costa Rica c\u00e9dula (national ID)
#   - Mexico RFC (taxpayer ID)
#   - Mexico CURP (population registry ID)
#   - Generic credit card numbers (Luhn-checkable shapes)
#   - IBAN-like account numbers
#
# REDACTION:
# When found, the matches are replaced with a placeholder
# token like "[REDACTED:EMAIL]" so the response remains
# readable but the secret is gone.
#
# LIMITATIONS:
# Regex-only PII detection misses contextual leaks (e.g.,
# "the client's home address is X" where X is not formatted
# as a postal code). Use this as a defensive layer, not the
# only defense.
# ============================================================

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class PIIMatch:
    """One PII finding with type and the matched text."""
    pii_type: str
    matched_text: str


# ============================================================
# Patterns
#
# Each tuple is (pii_type, pattern). pii_type is the label that
# will appear in the redaction placeholder. Order matters: more
# specific patterns should come first so the labels are accurate.
# ============================================================
_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    # Email — RFC 5322 simplified
    (
        "EMAIL",
        re.compile(
            r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        ),
    ),
    # Mexico CURP (18 chars: 4 letters + 6 digits + H/M + 5 letters + 2 chars)
    (
        "CURP",
        re.compile(
            r"\b[A-Z]{4}\d{6}[HM][A-Z]{5}[A-Z0-9]\d\b",
        ),
    ),
    # Mexico RFC (12 or 13 chars: 3-4 letters + 6 digits + 3 alnum)
    (
        "RFC",
        re.compile(
            r"\b[A-Z\u00d1&]{3,4}\d{6}[A-Z0-9]{3}\b",
        ),
    ),
    # Costa Rica c\u00e9dula formatted as 1-1234-5678 or 101230456
    (
        "CR_CEDULA",
        re.compile(
            r"\b\d-\d{4}-\d{4}\b|\b[1-9]\d{8}\b",
        ),
    ),
    # International phone with + country code (+506-, +52-, +1-, +34-, +57-)
    (
        "PHONE",
        re.compile(
            r"\+(?:506|52|1|34|57|54|55|56)[\s\-]?\d{2,4}[\s\-]?\d{3,4}[\s\-]?\d{3,4}",
        ),
    ),
    # Generic 8-12 digit phone (CR style with 8 digits, MX/US 10)
    (
        "PHONE",
        re.compile(
            r"\b\d{4}[\s\-]?\d{4}\b|\b\(\d{3}\)\s?\d{3}[\s\-]?\d{4}\b|\b\d{3}[\s\-]\d{3}[\s\-]\d{4}\b",
        ),
    ),
    # Credit-card-like (13-16 digits in groups). The Luhn algorithm is
    # applied separately by `_passes_luhn` to avoid flagging order IDs.
    (
        "CREDIT_CARD",
        re.compile(
            r"\b(?:\d[\s\-]?){13,16}\b",
        ),
    ),
    # IBAN — country code + check digits + up to 30 alnum
    (
        "IBAN",
        re.compile(
            r"\b[A-Z]{2}\d{2}[A-Z0-9]{10,30}\b",
        ),
    ),
]


def _passes_luhn(digits: str) -> bool:
    """Validate a string of digits with the Luhn checksum used by
    credit cards. Used to suppress false positives on long IDs."""
    nums = [int(c) for c in digits if c.isdigit()]
    if len(nums) < 13 or len(nums) > 19:
        return False
    checksum = 0
    parity = len(nums) % 2
    for i, n in enumerate(nums):
        if i % 2 == parity:
            n *= 2
            if n > 9:
                n -= 9
        checksum += n
    return checksum % 10 == 0


def detect_pii(text: str) -> list[PIIMatch]:
    """
    Scan a string for known PII patterns.

    Args:
        text: The text to scan (typically an LLM response).

    Returns:
        A list of PIIMatch entries. Empty when nothing is found.
    """
    if not text:
        return []
    findings: list[PIIMatch] = []
    seen: set[tuple[str, str]] = set()  # de-dupe (type, span) pairs

    for pii_type, pattern in _PATTERNS:
        for match in pattern.finditer(text):
            matched = match.group(0)
            if pii_type == "CREDIT_CARD" and not _passes_luhn(matched):
                # Long digit string that fails Luhn → almost certainly
                # not a card. Skip to avoid flagging order numbers.
                continue
            key = (pii_type, matched)
            if key in seen:
                continue
            seen.add(key)
            findings.append(PIIMatch(pii_type=pii_type, matched_text=matched))
    return findings


def redact_pii(text: str, findings: list[PIIMatch] | None = None) -> str:
    """
    Replace every PII match with `[REDACTED:<TYPE>]`.

    The function is safe to call repeatedly — it scans on every
    invocation. Pass an explicit `findings` list when you already
    have one to avoid the rescan.
    """
    if not text:
        return text
    findings = findings if findings is not None else detect_pii(text)
    redacted = text
    # Sort by length desc so a longer match is replaced before any
    # shorter substring that might also match.
    for finding in sorted(findings, key=lambda f: -len(f.matched_text)):
        redacted = redacted.replace(
            finding.matched_text, f"[REDACTED:{finding.pii_type}]"
        )
    return redacted
