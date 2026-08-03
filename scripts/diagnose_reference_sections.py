#!/usr/bin/env python3
# ============================================================
# scripts/diagnose_reference_sections.py
#
# Step 1 diagnostic for the "bibliography contaminates retrieval"
# problem: which signal identifies a reference/bibliography chunk
# most reliably?
#
# Candidate signals evaluated here:
#   A) HEADER  — an explicit heading ("BIBLIOGRAFIA", "Referencias
#      normativas", "References", ...) somewhere in the chunk.
#   B) DENSITY — the fraction of the chunk taken up by standard
#      citation identifiers (ISO/IEC 24027:2021, NIST SP 800-53,
#      EN 301 549, ...) plus the shortage of connected prose.
#
# Run with:
#   PYTHONPATH=. .venv/bin/python scripts/diagnose_reference_sections.py
# ============================================================

from __future__ import annotations

import os
import re
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.supabase import get_supabase_client

SEPARATOR = "=" * 78


def _print(title: str) -> None:
    print()
    print(SEPARATOR)
    print(f"  {title}")
    print(SEPARATOR)


# ------------------------------------------------------------
# Signal A — explicit bibliography heading
#
# Matched case-insensitively anywhere in the chunk. In the ISO
# 42001 replica the heading sits MID-chunk (idx=69), so anchoring
# to the start of the chunk would miss it.
# ------------------------------------------------------------
_HEADER_RE = re.compile(
    r"(?:^|\s)("
    r"BIBLIOGRAF[IÍ]A|BIBLIOGRAPHY|"
    r"REFERENCIAS(?:\s+(?:NORMATIVAS|BIBLIOGR[AÁ]FICAS))?|"
    r"REFERENCES|NORMATIVE\s+REFERENCES|"
    r"DOCUMENTOS?\s+DE\s+REFERENCIA|"
    r"NORMAS?\s+(?:DE\s+)?REFERENCIA|"
    r"OBRAS?\s+CITADAS|WORKS\s+CITED"
    r")(?:\s|$|:)",
    re.IGNORECASE,
)

# ------------------------------------------------------------
# Signal B — density of standard-citation identifiers
# ------------------------------------------------------------
_CITATION_RE = re.compile(
    r"ISO(?:/IEC)?(?:/IEEE)?\s+(?:TR\s+|TS\s+|PAS\s+|Guide\s+)?\d{3,6}(?:-\d+)?(?::\d{4})?"
    r"|IEC\s+\d{3,6}(?:-\d+)?(?::\d{4})?"
    r"|NIST\s+(?:SP\s+)?\d{3,4}(?:-\d+)*(?:[A-Za-z])?"
    r"|NIST\.[A-Z]{2}\.\d+"
    r"|EN\s+\d{3,6}(?::\d{4})?"
    r"|RFC\s+\d{3,5}"
    r"|IEEE\s+\d{3,4}(?:\.\d+)?"
    r"|ANSI\s+[A-Z0-9./-]{3,}"
    r"|BS\s+\d{3,5}",
    re.IGNORECASE,
)

# Connectives that only appear in running prose, never in a list of
# standard titles. Used as the negative half of the density signal.
_PROSE_RE = re.compile(
    r"\b(?:deber[íi]a|deben?|debe|puede[n]?|podr[íi]a|se\s+debe|"
    r"la\s+organizaci[oó]n|el\s+sistema|por\s+ejemplo|es\s+decir|"
    r"con\s+el\s+fin|para\s+que|cuando|si\s+el|shall|should|may\s+be|"
    r"the\s+organization|in\s+order\s+to|such\s+as|for\s+example)\b",
    re.IGNORECASE,
)


def citation_density(text: str) -> float:
    """Fraction of the chunk's characters consumed by citation IDs."""
    if not text:
        return 0.0
    covered = sum(m.end() - m.start() for m in _CITATION_RE.finditer(text))
    return covered / len(text)


def citation_count(text: str) -> int:
    return len(_CITATION_RE.findall(text or ""))


def prose_count(text: str) -> int:
    return len(_PROSE_RE.findall(text or ""))


def main() -> None:
    db = get_supabase_client()

    # ---------- 0. The four labelled chunks from the diagnosis ----------
    _print("0. Labelled chunks (ISO 42001 replica) — how do the signals score?")
    iso_doc = "a93bc396-6c66-4296-aa8a-0f178998cac0"
    labels = {56: "SUBSTANTIVE", 64: "SUBSTANTIVE", 68: "SUBSTANTIVE",
              69: "BIBLIOGRAPHY (mixed)", 70: "BIBLIOGRAPHY (tail)"}
    rows = (
        db.table("document_chunks")
        .select("chunk_index, section_ref, content")
        .eq("document_id", iso_doc)
        .in_("chunk_index", list(labels))
        .order("chunk_index")
        .execute()
        .data
        or []
    )
    print(f"  {'idx':>4}  {'label':22}  {'header?':8}  {'cites':>5}  "
          f"{'density':>7}  {'prose':>5}")
    for r in rows:
        c = r["content"] or ""
        idx = r["chunk_index"]
        hdr = "YES" if _HEADER_RE.search(c) else "no"
        print(f"  {idx:>4}  {labels[idx]:22}  {hdr:8}  {citation_count(c):>5}  "
              f"{citation_density(c):>7.3f}  {prose_count(c):>5}")

    # ---------- 1. Corpus-wide header scan ----------
    _print("1. Which documents carry an explicit bibliography heading?")
    docs = (
        db.table("documents")
        .select("id, filename, framework, chunk_count")
        .execute()
        .data
        or []
    )
    by_doc = {d["id"]: d for d in docs}

    all_chunks: list[dict] = []
    page = 0
    while True:
        batch = (
            db.table("document_chunks")
            .select("document_id, chunk_index, section_ref, content")
            .range(page * 1000, page * 1000 + 999)
            .execute()
            .data
            or []
        )
        all_chunks.extend(batch)
        if len(batch) < 1000:
            break
        page += 1
    print(f"  scanned {len(all_chunks)} chunks across {len(docs)} documents")

    header_hits: dict[str, list[dict]] = {}
    heading_forms: Counter[str] = Counter()
    for c in all_chunks:
        m = _HEADER_RE.search(c["content"] or "")
        if m:
            header_hits.setdefault(c["document_id"], []).append(c)
            heading_forms[m.group(1).strip().upper()] += 1

    print(f"\n  documents with >=1 heading hit: {len(header_hits)} / {len(docs)}")
    print("\n  heading forms found:")
    for form, n in heading_forms.most_common():
        print(f"    {n:5d}  {form}")

    # ---------- 2. Where does the heading sit inside the document? ----------
    _print("2. Heading position within each document (relative chunk index)")
    print("  A heading in the last ~10% of a document is a real end-of-doc")
    print("  bibliography. One near the start is usually the ISO-style")
    print("  'Referencias normativas' clause, which is SHORT and is followed")
    print("  by substantive content — so 'everything after the heading' is")
    print("  NOT a safe rule.\n")
    for doc_id, hits in sorted(
        header_hits.items(), key=lambda kv: by_doc.get(kv[0], {}).get("filename", "")
    ):
        d = by_doc.get(doc_id, {})
        total = d.get("chunk_count") or 0
        positions = sorted(h["chunk_index"] for h in hits)
        rel = [f"{(p / total * 100):.0f}%" if total else "?" for p in positions]
        print(f"  {d.get('filename','?')[:52]:52} n={total:>4}  "
              f"idx={positions}  ({', '.join(rel)})")

    # ---------- 3. Density signal, corpus-wide ----------
    _print("3. Citation-density signal — top 25 chunks corpus-wide")
    scored = []
    for c in all_chunks:
        text = c["content"] or ""
        scored.append((citation_density(text), citation_count(text),
                       prose_count(text), c))
    scored.sort(key=lambda t: -t[0])
    print(f"  {'dens':>6}  {'cites':>5}  {'prose':>5}  {'hdr':>4}  {'idx':>4}  doc / preview")
    for dens, cites, prose, c in scored[:25]:
        d = by_doc.get(c["document_id"], {})
        hdr = "YES" if _HEADER_RE.search(c["content"] or "") else "-"
        preview = (c["content"] or "")[:60].replace("\n", " ")
        print(f"  {dens:>6.3f}  {cites:>5}  {prose:>5}  {hdr:>4}  "
              f"{c['chunk_index']:>4}  {d.get('filename','?')[:26]:26} {preview}")

    # ---------- 4. False-positive probe for the density signal ----------
    _print("4. Density false positives: SUBSTANTIVE chunks that cite many standards")
    print("  These are chunks with a high citation count but plenty of prose —")
    print("  i.e. real content that merely references standards. A density-only")
    print("  rule would wrongly demote them.\n")
    fp = [t for t in scored if t[1] >= 3 and t[2] >= 3]
    fp.sort(key=lambda t: -t[1])
    for dens, cites, prose, c in fp[:12]:
        d = by_doc.get(c["document_id"], {})
        preview = (c["content"] or "")[:100].replace("\n", " ")
        print(f"  dens={dens:.3f} cites={cites} prose={prose}  "
              f"[{d.get('filename','?')[:24]}] {preview}...")


if __name__ == "__main__":
    main()
