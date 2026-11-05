#!/usr/bin/env python3
"""
Create datasets/gold_standard.json from the template with a 5-query pilot
annotation pass. Uses full chunk text from Supabase + keyword heuristics.

This is a bootstrap pilot — human review is still recommended before
treating metrics as authoritative.
"""

from __future__ import annotations

import json
import os
import re
import sys
import unicodedata
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.supabase import get_supabase_client

TEMPLATE_PATH = "datasets/gold_standard_template.json"
OUTPUT_PATH = "datasets/gold_standard.json"

# Diverse pilot set: ISO ES, NIST GOVERN/MAP, ISO EN, ISO evidence, legal/credit
PILOT_TRACE_IDS = {
    "9cbc05bc-3c05-444f-8965-a5fd5042274a",
    "e29bc33c-ea17-4b33-916d-a450a14cc189",
    "4b59e756-20be-44d5-a2d2-a96de1ad42bc",
    "daa2635d-4d64-4ac4-a7ae-b618173d8d34",
    "aea70abb-b346-4a89-bc8f-0fa64415cc58",
}


def _normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.lower())
    return "".join(c for c in text if not unicodedata.combining(c))


def _score_relevance(query: str, content: str, section_ref: str | None) -> int:
    """Return 0/1/2 relevance using keyword overlap heuristics."""
    q = _normalize(query)
    c = _normalize(content)
    section = _normalize(section_ref or "")

    # Strong signals for query 1: ISO 42001 governance committee
    if "comite de gobernanza" in q or "comité de gobernanza" in query.lower():
        if any(k in c for k in ("comite", "comité", "gobernanza", "alta direccion", "roles y responsabilidades")):
            return 2
        if "iso 42001" in c or "42001" in c:
            return 1
        return 0

    # NIST GOVERN + MAP
    if "govern" in q and "map" in q and "nist" in q:
        if "govern" in c and "map" in c:
            return 2
        if "govern" in c or "map" in c:
            return 2 if "appendix d" in section or "appendix c" in section else 1
        if "nist" in c or "ai rmf" in c:
            return 1
        return 0

    # English ISO 42001 requirements
    if "iso 42001" in q and "governance" in q:
        if "42001" in c or "iso 42001" in c:
            return 2 if any(k in c for k in ("requirement", "governance", "management system", "requisito")) else 1
        if "govern" in c or "oversight" in c:
            return 1
        return 0

    # ISO 42001 evidence / documentation
    if "evidencias" in q or "evidence" in q:
        if any(k in c for k in ("evidencia", "evidence", "documentacion", "documentación", "registro", "informacion documentada")):
            return 2
        if "42001" in c or "iso 42001" in c:
            return 1
        if "pci" in c or "payment card" in c:
            return 0
        return 0

    # Legal requirements for AI in credit
    if "credito" in q or "crédito" in query.lower():
        if any(k in c for k in ("credito", "crédito", "credit", "financier", "prestamo", "préstamo", "scoring")):
            return 2
        if any(k in c for k in ("decision automatizada", "automated decision", "alto riesgo", "high-risk", "high risk")):
            return 2
        if "gdpr" in c or "proteccion de datos" in c or "protección de datos" in c:
            return 1
        return 0

    # Impact analysis in AI governance
    if "analisis de impacto" in q or "análisis de impacto" in query.lower():
        if any(k in c for k in ("impacto", "impact", "evaluacion de impacto", "evaluación de impacto", "fria")):
            return 2
        if "42001" in c:
            return 1
        return 0

    # Fallback: token overlap
    q_tokens = set(re.findall(r"[a-z0-9]{4,}", q))
    c_tokens = set(re.findall(r"[a-z0-9]{4,}", c))
    if not q_tokens:
        return 0
    overlap = len(q_tokens & c_tokens) / len(q_tokens)
    if overlap >= 0.35:
        return 2
    if overlap >= 0.15:
        return 1
    return 0


def fetch_chunk_contents(db: Any, chunk_ids: list[str]) -> dict[str, dict]:
    if not chunk_ids:
        return {}
    result = (
        db.table("document_chunks")
        .select("id, content, section_ref")
        .in_("id", chunk_ids)
        .execute()
    )
    return {row["id"]: row for row in (result.data or [])}


def main() -> None:
    with open(TEMPLATE_PATH, encoding="utf-8") as f:
        template = json.load(f)

    entries = template["entries"]
    db = get_supabase_client()

    annotated_count = 0
    for entry in entries:
        trace_id = entry.get("trace_id")
        if trace_id not in PILOT_TRACE_IDS:
            continue

        chunks = entry.get("retrieved_chunks") or []
        if not chunks:
            continue

        chunk_ids = [c["chunk_id"] for c in chunks]
        meta = fetch_chunk_contents(db, chunk_ids)
        query = entry["user_query"]

        for chunk in chunks:
            cid = chunk["chunk_id"]
            row = meta.get(cid, {})
            content = row.get("content") or chunk.get("content_preview") or ""
            section = row.get("section_ref") or chunk.get("section_ref")
            chunk["relevance"] = _score_relevance(query, content, section)

        annotated_count += 1
        print(f"Annotated: {query[:70]}...")
        for chunk in chunks:
            print(f"  {chunk['chunk_id'][:8]}... relevance={chunk['relevance']}")

    out = {
        "_annotation_instructions": template.get("_annotation_instructions", ""),
        "_pilot_note": (
            "Pilot annotations for 5 queries (automated heuristic pass). "
            "Review before using as authoritative baseline."
        ),
        "entries": entries,
    }

    os.makedirs(os.path.dirname(OUTPUT_PATH) or ".", exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {OUTPUT_PATH} ({annotated_count} pilot entries annotated)")


if __name__ == "__main__":
    main()
