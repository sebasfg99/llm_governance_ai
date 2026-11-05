#!/usr/bin/env python3
# ============================================================
# scripts/build_eval_dataset.py — Build RAG retrieval eval dataset
#
# PURPOSE:
# Export real queries from the `traces` table and produce two files:
#   1. datasets/eval_queries.json       — queries with retrieval params, ready for eval
#   2. datasets/gold_standard_template.json — annotation template with chunk previews
#
# The annotator fills in `relevance` for each chunk (2=RELEVANT, 1=PARTIAL, 0=NOT_RELEVANT),
# then saves the result as datasets/gold_standard.json for use by eval_retrieval.py.
#
# METRIC DISTINCTION (important — do not conflate):
#   - `retrieval_precision` in the governance pipeline = fraction of chunks with
#     similarity >= threshold (stored in the `evaluations` table). This is a proxy
#     metric computed automatically post-response.
#   - Precision@k / Recall@k / NDCG / MRR computed by eval_retrieval.py = IR benchmark
#     metrics measured against this manually annotated gold standard. These require
#     human judgement and are only as good as the annotation.
#
# RETRIEVAL PARAMS:
# The `traces` table does NOT store country, framework_filter, or include_global.
# This script reconstructs them via:
#   (a) Inference from retrieved chunk metadata (document_chunks.framework / .country)
#   (b) Documented system defaults when inference yields nothing
#   (c) Notebook payloads (full params) when real traces are insufficient
#
# HOW TO RUN:
#   PYTHONPATH=. .venv/bin/python scripts/build_eval_dataset.py
#   PYTHONPATH=. .venv/bin/python scripts/build_eval_dataset.py --limit 50 --min-traces 30
#
# PREREQUISITES:
#   .env with SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY
# ============================================================

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import sys
import warnings
from collections import Counter
from datetime import datetime
from math import ceil
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.logging import get_logger
from app.db.supabase import get_supabase_client

logger = get_logger(__name__)

# ============================================================
# Constants
# ============================================================
RAG_USE_CASES = ("chatbot", "executive_summary")

# Default retrieval params when none can be inferred from traces.
# These mirror AIService defaults at time of writing.
DEFAULT_TOP_K = 5
DEFAULT_INCLUDE_GLOBAL = True
DEFAULT_COUNTRY = None
DEFAULT_FRAMEWORK_FILTER = None

# Query length strata (character count of user_query)
STRATUM_SHORT_MAX = 50
STRATUM_LONG_MIN = 150


# ============================================================
# Chunk ID normalization — reused from scripts/verify_grounding.py
# ============================================================
def normalize_chunk_ids(raw_ids: Any) -> list[str]:
    """
    Normalize retrieved_chunk_ids which may be a list of strings,
    a list of dicts with 'chunk_id' or 'id' keys, or a JSON string.

    Invalid UUIDs are dropped so PostgREST `.in_("id", ...)` does not fail
    when traces store malformed retrieved_chunk_ids.
    """
    if raw_ids is None:
        return []

    if isinstance(raw_ids, str):
        text = raw_ids.strip()
        if not text:
            return []
        try:
            raw_ids = json.loads(text)
        except json.JSONDecodeError:
            return []

    if not isinstance(raw_ids, list):
        return []

    chunk_ids: list[str] = []
    for entry in raw_ids:
        cid: str | None = None
        if isinstance(entry, str):
            cid = entry
        elif isinstance(entry, dict):
            cid = entry.get("chunk_id") or entry.get("id")
        if not cid:
            continue
        cid = str(cid).strip()
        # Basic UUID shape check (8-4-4-4-12 hex)
        if len(cid) == 36 and cid.count("-") == 4:
            chunk_ids.append(cid)
    return chunk_ids


# ============================================================
# Notebook payload parser — runtime .ipynb parsing
# ============================================================
def load_notebook_payloads(path: str, use_cases: tuple[str, ...]) -> list[dict]:
    """
    Parse a Jupyter notebook at runtime and extract request payloads.

    Looks for code cells containing `payload = {...}` assignments and uses
    ast.literal_eval to safely parse the Python dict. Only returns payloads
    whose use_case is in `use_cases` (RAG use cases only).
    """
    if not path or not str(path).strip():
        return []
    if not os.path.exists(path):
        logger.warning(f"Notebook not found at {path} — skipping notebook payloads.")
        return []

    try:
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
    except Exception as e:
        logger.warning(f"Could not parse notebook {path}: {e}")
        return []

    payloads: list[dict] = []
    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        src = "".join(cell.get("source", []))
        m = re.search(r"payload\s*=\s*(\{.*?\})", src, re.DOTALL)
        if not m:
            continue
        try:
            p = ast.literal_eval(m.group(1))
        except Exception:
            continue
        if not isinstance(p, dict):
            continue
        if "query" not in p or "use_case" not in p:
            continue
        if p["use_case"] not in use_cases:
            continue
        payloads.append(p)

    logger.info(f"Loaded {len(payloads)} usable payloads from notebook {path}")
    return payloads


def _notebook_payload_to_entry(p: dict) -> dict:
    """Convert a notebook payload dict to the eval_queries entry format."""
    return {
        "trace_id": None,
        "user_query": p["query"],
        "tenant_id": p.get("tenant_id", "unknown"),
        "use_case": p["use_case"],
        "created_at": None,
        "params_source": "payload_notebook",
        "retrieval_params": {
            "top_k": p.get("top_k", DEFAULT_TOP_K),
            "include_global": p.get("include_global", DEFAULT_INCLUDE_GLOBAL),
            "country": p.get("country", DEFAULT_COUNTRY),
            "framework_filter": p.get("framework_filter", DEFAULT_FRAMEWORK_FILTER),
        },
        "retrieved_chunk_ids": [],
        "retrieved_content": None,
        "project_id": p.get("project_id"),
        "thread_id": p.get("thread_id"),
    }


# ============================================================
# Retrieval param inference from document_chunks metadata
# ============================================================
def infer_params_from_chunks(
    db: Any,
    chunk_ids: list[str],
) -> dict:
    """
    Query document_chunks to infer framework_filter and country.

    document_chunks propagates framework and country from the parent
    document at ingestion time, so we can read them here without a join.

    Returns a dict with keys: framework_filter, country, params_source.
    """
    if not chunk_ids:
        return {
            "framework_filter": DEFAULT_FRAMEWORK_FILTER,
            "country": DEFAULT_COUNTRY,
            "params_source": "trace_defaults",
        }

    try:
        result = (
            db.table("document_chunks")
            .select("id, framework, country")
            .in_("id", chunk_ids)
            .execute()
        )
        rows = result.data or []
    except Exception as e:
        logger.warning(f"Could not query document_chunks for param inference: {e}")
        return {
            "framework_filter": DEFAULT_FRAMEWORK_FILTER,
            "country": DEFAULT_COUNTRY,
            "params_source": "trace_defaults",
        }

    frameworks = [r["framework"] for r in rows if r.get("framework")]
    countries = [r["country"] for r in rows if r.get("country")]

    inferred_framework = None
    inferred_country = None

    if frameworks:
        most_common_fw = Counter(frameworks).most_common(1)[0][0]
        inferred_framework = [most_common_fw]

    if countries:
        inferred_country = Counter(countries).most_common(1)[0][0]

    params_source = (
        "inferred_from_chunks"
        if (inferred_framework or inferred_country)
        else "trace_defaults"
    )

    return {
        "framework_filter": inferred_framework,
        "country": inferred_country,
        "params_source": params_source,
    }


# ============================================================
# Stratification
# ============================================================
def _query_length_stratum(query: str) -> str:
    n = len(query)
    if n < STRATUM_SHORT_MAX:
        return "short"
    if n < STRATUM_LONG_MIN:
        return "medium"
    return "long"


def stratify(entries: list[dict], limit: int) -> list[dict]:
    """
    Best-effort stratification by query_length × tenant_id × inferred_framework.

    Selects up to ceil(limit / n_strata) entries per stratum. Empty strata
    emit a warning so the caller can supplement with notebook payloads.
    """
    # Build strata
    strata: dict[tuple, list[dict]] = {}
    for e in entries:
        ql = _query_length_stratum(e["user_query"])
        tenant = e["tenant_id"]
        fw = (
            e["retrieval_params"]["framework_filter"][0]
            if e["retrieval_params"]["framework_filter"]
            else "none"
        )
        key = (ql, tenant, fw)
        strata.setdefault(key, []).append(e)

    if not strata:
        return []

    per_stratum = max(1, ceil(limit / len(strata)))
    selected: list[dict] = []
    for key, bucket in strata.items():
        if not bucket:
            warnings.warn(
                f"Stratum {key} is empty — supplement with notebook payloads.",
                UserWarning,
                stacklevel=2,
            )
        selected.extend(bucket[:per_stratum])

    return selected[:limit]


# ============================================================
# Chunk lookup for annotation template
# ============================================================
def fetch_chunk_previews(db: Any, chunk_ids: list[str]) -> dict[str, dict]:
    """
    Fetch section_ref and content for annotation context.
    Returns a dict keyed by chunk_id.
    """
    if not chunk_ids:
        return {}
    try:
        result = (
            db.table("document_chunks")
            .select("id, section_ref, content")
            .in_("id", chunk_ids)
            .execute()
        )
        return {r["id"]: r for r in (result.data or [])}
    except Exception as e:
        logger.warning(f"Could not fetch chunk previews: {e}")
        return {}


# ============================================================
# Main build function
# ============================================================
def build_eval_dataset(
    *,
    limit: int,
    min_traces: int,
    use_cases: tuple[str, ...],
    output_dir: str,
    notebook_path: str,
) -> dict[str, Any]:
    db = get_supabase_client()

    os.makedirs(output_dir, exist_ok=True)

    # ---- Step 1: Pull traces ----
    logger.info(
        f"Pulling traces: use_cases={use_cases} limit_pool={limit * 3}"
    )
    try:
        result = (
            db.table("traces")
            .select(
                "id, user_query, tenant_id, use_case, created_at, "
                "retrieved_chunk_ids, retrieved_content, response_text, "
                "project_id, thread_id"
            )
            .in_("use_case", list(use_cases))
            .order("created_at", desc=True)
            .limit(limit * 3)
            .execute()
        )
        raw_traces = result.data or []
    except Exception as e:
        logger.error(f"Failed to pull traces: {e}")
        raw_traces = []

    logger.info(f"Fetched {len(raw_traces)} raw traces from DB")

    # ---- Step 2 + 3: Normalize chunk IDs + infer params ----
    entries: list[dict] = []
    for t in raw_traces:
        chunk_ids = normalize_chunk_ids(t.get("retrieved_chunk_ids"))
        inferred = infer_params_from_chunks(db, chunk_ids)

        entry = {
            "trace_id": str(t["id"]),
            "user_query": t["user_query"],
            "tenant_id": t["tenant_id"],
            "use_case": t["use_case"],
            "created_at": t.get("created_at"),
            "params_source": inferred["params_source"],
            "retrieval_params": {
                "top_k": DEFAULT_TOP_K,
                "include_global": DEFAULT_INCLUDE_GLOBAL,
                "country": inferred["country"],
                "framework_filter": inferred["framework_filter"],
            },
            "retrieved_chunk_ids": chunk_ids,
            "retrieved_content": t.get("retrieved_content"),
            "project_id": t.get("project_id"),
            "thread_id": t.get("thread_id"),
        }
        entries.append(entry)

    # ---- Step 4: Stratify ----
    selected = stratify(entries, limit)
    logger.info(f"After stratification: {len(selected)} entries selected")

    # ---- Step 5: Supplement with notebook payloads if needed ----
    if len(selected) < min_traces:
        shortage = min_traces - len(selected)
        logger.warning(
            f"Only {len(selected)} trace-based entries; need {min_traces}. "
            f"Supplementing with up to {shortage} notebook payloads."
        )
        nb_payloads = load_notebook_payloads(notebook_path, use_cases)
        for p in nb_payloads[:shortage]:
            selected.append(_notebook_payload_to_entry(p))
        logger.info(f"Total after notebook supplement: {len(selected)} entries")

    if not selected:
        logger.error(
            "No entries produced (0 traces + 0 notebook payloads). "
            "Check use_case filter and database connectivity."
        )
        return {"eval_queries": 0, "gold_template": 0, "output_dir": output_dir}

    # ---- Step 6: Fetch chunk previews for annotation template ----
    all_chunk_ids: list[str] = []
    for e in selected:
        all_chunk_ids.extend(e["retrieved_chunk_ids"])
    all_chunk_ids = list(dict.fromkeys(all_chunk_ids))  # deduplicate preserving order

    chunk_meta = fetch_chunk_previews(db, all_chunk_ids)

    # ---- Step 7: Write eval_queries.json ----
    eval_path = os.path.join(output_dir, "eval_queries.json")
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(selected, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Wrote {len(selected)} entries → {eval_path}")

    # ---- Step 8: Write gold_standard_template.json ----
    # Annotation scale:
    #   relevance: 2 = RELEVANT (chunk directly answers the query)
    #              1 = PARTIAL  (chunk is related but not the main answer)
    #              0 = NOT_RELEVANT (chunk does not help answer the query)
    #   Fill in the `relevance` field for each chunk, then save as gold_standard.json
    template_entries: list[dict] = []
    for e in selected:
        retrieved_chunks = []
        for cid in e["retrieved_chunk_ids"]:
            meta = chunk_meta.get(cid)
            content_preview = (
                (meta["content"] or "")[:200] if meta else "[chunk not found]"
            )
            retrieved_chunks.append(
                {
                    "chunk_id": cid,
                    "section_ref": meta["section_ref"] if meta else None,
                    "content_preview": content_preview,
                    "relevance": None,  # annotator fills this in: 0, 1, or 2
                }
            )

        template_entries.append(
            {
                "trace_id": e["trace_id"],
                "user_query": e["user_query"],
                "tenant_id": e["tenant_id"],
                "use_case": e["use_case"],
                "params_source": e["params_source"],
                "retrieval_params": e["retrieval_params"],
                "thread_context": None,  # optional: fill in if query depends on conversation history
                "retrieved_chunks": retrieved_chunks,
            }
        )

    template_path = os.path.join(output_dir, "gold_standard_template.json")
    annotation_guide = {
        "_annotation_instructions": (
            "Fill in 'relevance' for each chunk. Scale: "
            "2=RELEVANT (directly answers the query), "
            "1=PARTIAL (related but not the main answer), "
            "0=NOT_RELEVANT (does not help). "
            "Entries with all-null relevance are skipped by eval_retrieval.py. "
            "Save your annotations as datasets/gold_standard.json."
        ),
        "entries": template_entries,
    }
    with open(template_path, "w", encoding="utf-8") as f:
        json.dump(annotation_guide, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Wrote annotation template → {template_path}")

    return {
        "eval_queries": len(selected),
        "gold_template": len(template_entries),
        "output_dir": output_dir,
    }


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build RAG retrieval evaluation dataset from production traces"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Target number of eval entries (default: 50). "
             "Script pulls limit*3 traces from DB for stratification.",
    )
    parser.add_argument(
        "--min-traces",
        type=int,
        default=30,
        help="Minimum entries required before supplementing with notebook payloads (default: 30).",
    )
    parser.add_argument(
        "--use-cases",
        type=str,
        default="chatbot,executive_summary",
        help="Comma-separated use cases to include (default: chatbot,executive_summary). "
             "risk_intake and similar non-RAG use cases are excluded.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Directory for output files (default: datasets/).",
    )
    parser.add_argument(
        "--notebook",
        type=str,
        default="",
        help="Optional .ipynb with `payload = {...}` cells to supplement traces when "
        "count < --min-traces. Default: none (use production traces only; see datasets/eval_queries.json).",
    )
    args = parser.parse_args()

    use_cases = tuple(uc.strip() for uc in args.use_cases.split(",") if uc.strip())

    stats = build_eval_dataset(
        limit=args.limit,
        min_traces=args.min_traces,
        use_cases=use_cases,
        output_dir=args.output_dir,
        notebook_path=args.notebook,
    )

    print(json.dumps(stats, indent=2, ensure_ascii=False))
