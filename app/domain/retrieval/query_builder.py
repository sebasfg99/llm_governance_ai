# ============================================================
# app/domain/retrieval/query_builder.py — Server-side retrieval
# query construction.
#
# WHY THIS FILE EXISTS:
# The user query serves two different purposes in the pipeline:
#
#   1. Retrieval (RAG): embedding quality depends on a rich,
#      specific string that captures the regulatory domain and
#      project context. Short queries produce diffuse embeddings;
#      long form-fill queries dilute the semantic signal.
#
#   2. Generation (LLM): the user's original wording must be
#      preserved verbatim so the LLM can address their actual
#      question.
#
# This module provides pure, deterministic functions that build
# an enriched retrieval_query from the user's question and the
# project profile — without touching the prompt or the API
# contract.
#
# All functions here are side-effect-free and fully unit-testable
# without mocking any external service.
# ============================================================

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.domain.projects.models import Project

# ------------------------------------------------------------------
# Framework detection patterns
# Each entry: (compiled regex, canonical tag)
# Patterns are tried in order; first match wins per framework.
# ------------------------------------------------------------------
_FRAMEWORK_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"iso[/ ]?iec[/ ]?42001|iso[/ ]?42001", re.IGNORECASE), "iso_42001"),
    (re.compile(r"nist\s+ai\s+rmf|nist\s+airmf|nist", re.IGNORECASE), "nist_airmf"),
    (re.compile(r"eu\s+ai\s+act|ai\s+act", re.IGNORECASE), "eu_ai_act"),
    (re.compile(r"owasp\s+llm\s+top\s*10|owasp", re.IGNORECASE), "owasp_llm_top10"),
    (re.compile(r"gdpr", re.IGNORECASE), "gdpr"),
    (re.compile(r"pci\s+dss|pci(?!\s*\w)", re.IGNORECASE), "pci_dss"),
]


def infer_framework_filter(
    query: str,
    explicit: list[str] | None,
) -> list[str] | None:
    """
    Resolve the framework filter to use for retrieval.

    If the caller already provided an explicit list, it is returned
    unchanged — user intent always wins.

    Otherwise the query text is scanned for framework mentions (both
    Spanish and English, case-insensitive).  The canonical tag for
    each detected framework is collected into a deduplicated list.
    Returns None if nothing is detected so that RetrievalService
    applies no filter (broad search).

    Args:
        query:    The user question (after guardrails).
        explicit: framework_filter as received in the request body,
                  may be None or an empty list.

    Returns:
        The explicit list if provided and non-empty, otherwise the
        inferred list (possibly None).
    """
    if explicit:
        return explicit

    detected: list[str] = []
    seen: set[str] = set()
    for pattern, tag in _FRAMEWORK_PATTERNS:
        if tag not in seen and pattern.search(query):
            detected.append(tag)
            seen.add(tag)

    return detected if detected else None


def extract_user_question(query: str) -> str:
    """
    Extract the core question from a potentially long query string.

    Long queries often arrive from UI forms where the user fills in
    project fields and appends a question at the end.  The embedded
    question is the most semantically dense part and produces better
    embeddings than the full form text.

    Heuristics (applied in order, first match wins):

    1. If the text contains newlines, inspect the last non-empty
       line.  If it ends with '?', return it — it is the explicit
       closing question.
    2. If any line starts with '¿', return the *last* such line.
    3. Return the full query stripped of leading/trailing whitespace.

    This function never calls an LLM.  It is deterministic and fast.

    Args:
        query: The raw user query string.

    Returns:
        The extracted question, or the full query if no heuristic
        matches.
    """
    query = query.strip()
    if not query:
        return query

    lines = query.splitlines()
    non_empty = [ln.strip() for ln in lines if ln.strip()]

    if len(non_empty) > 1:
        last = non_empty[-1]
        if last.endswith("?"):
            return last

        inverted_q_lines = [ln for ln in non_empty if ln.startswith("¿")]
        if inverted_q_lines:
            return inverted_q_lines[-1]

    return query


def build_retrieval_query(
    user_question: str,
    project: "Project | None",
) -> str:
    """
    Construct a rich retrieval query string for vector embedding.

    Combines the user question with a compact summary of the project
    profile so that the embedding captures both the regulatory topic
    and the project context.  The result is a single line — multi-
    line strings degrade embedding quality.

    Only non-null, non-empty profile fields are included.  The total
    project metadata contribution is kept short (≤~200 tokens) to
    avoid diluting the question signal.

    The format is:

        {question} | Proyecto: {name} | Tipo IA: {ai_type} |
        Datos: {data_used} | Sector: {industry} |
        Autonomía: {autonomy_level} | Proveedores: {existing_vendor}

    Args:
        user_question: The question extracted by extract_user_question.
        project:       The loaded Project instance, or None.

    Returns:
        The enriched retrieval query string.
    """
    if not project:
        return user_question

    profile = project.project_profile
    fragments: list[str] = []

    if project.name:
        fragments.append(f"Proyecto: {project.name}")

    if profile.ai_type:
        fragments.append(f"Tipo IA: {profile.ai_type}")

    if profile.data_used:
        data_str = (
            ", ".join(str(d) for d in profile.data_used)
            if isinstance(profile.data_used, list)
            else str(profile.data_used)
        )
        if data_str:
            fragments.append(f"Datos: {data_str}")

    if profile.industry:
        fragments.append(f"Sector: {profile.industry}")

    if profile.autonomy_level:
        fragments.append(f"Autonomía: {profile.autonomy_level}")

    if profile.existing_vendor:
        fragments.append(f"Proveedores: {profile.existing_vendor}")

    if profile.project_description:
        # Collapse newlines so the retrieval query stays on a single line.
        desc = " ".join(profile.project_description.split())[:150].strip()
        if desc:
            fragments.append(f"Descripción: {desc}")

    if not fragments:
        return user_question

    context_str = " | ".join(fragments)
    return f"{user_question} | {context_str}"


def split_top_k_across_frameworks(
    top_k: int,
    frameworks: list[str],
) -> list[tuple[str, int]]:
    """
    Distribute top_k slots evenly across multiple inferred frameworks.

    When a query mentions both ISO 42001 and NIST, a single broad
    retrieval call tends to return many chunks from the dominant
    framework (usually ISO) and starve the secondary one.  Splitting
    the budget guarantees each framework contributes to the context.

    Args:
        top_k:     Total chunks to retrieve.
        frameworks: Canonical framework tags (e.g. iso_42001, nist_airmf).

    Returns:
        List of (framework_tag, per_framework_top_k) pairs.  Remainder
        slots are assigned to the first frameworks in the list.
    """
    if not frameworks:
        return []

    base = max(1, top_k // len(frameworks))
    remainder = top_k - base * len(frameworks)
    allocation: list[tuple[str, int]] = []
    for i, fw in enumerate(frameworks):
        slots = base + (1 if i < remainder else 0)
        allocation.append((fw, slots))
    return allocation
