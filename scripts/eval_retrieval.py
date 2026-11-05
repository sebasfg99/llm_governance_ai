#!/usr/bin/env python3
# ============================================================
# scripts/eval_retrieval.py — IR benchmark evaluation for RAG retrieval
#
# PURPOSE:
# Read a manually annotated gold standard and measure how well
# RetrievalService finds the annotated relevant chunks for each query.
#
# METRIC DISTINCTION (important — do not conflate):
#   - `retrieval_precision` in the GOVERNANCE PIPELINE = fraction of retrieved
#     chunks with similarity >= threshold (0.45). Computed automatically
#     post-response and stored in the `evaluations` table. It does NOT require
#     human annotation and does NOT measure recall or ranking quality.
#   - The metrics computed HERE (Precision@k, Recall@k, NDCG@k, MRR) are
#     IR BENCHMARK METRICS measured against manually annotated gold standard.
#     They require human judgement and measure retrieval quality holistically.
#     These are the metrics relevant for deciding whether to switch to hybrid
#     search or a reranker.
#
# COST NOTE:
# Each query in the gold standard triggers one OpenAI embedding call
# (text-embedding-3-small). Approximate cost: ~$0.0002 per 1K tokens.
# For 30 queries of ~50 tokens each: < $0.001 total. Still, be aware
# that repeated runs accumulate cost.
#
# SIMILARITY THRESHOLD NOTE:
# RetrievalService applies settings.similarity_threshold (default 0.45)
# and may return fewer than top_k chunks. A query returning 0 chunks
# scores 0 on all metrics. This is expected and meaningful — it means
# the retriever found no evidence above the quality bar.
#
# HOW TO RUN:
#   # Baseline (vector-only):
#   PYTHONPATH=. .venv/bin/python scripts/eval_retrieval.py \
#     --gold-standard datasets/gold_standard.json
#
#   # Hybrid:
#   PYTHONPATH=. .venv/bin/python scripts/eval_retrieval.py \
#     --gold-standard datasets/gold_standard.json --use-hybrid
#
# PREREQUISITES:
#   - datasets/gold_standard.json with at least some annotated entries
#     (entries where all relevance values are null are skipped)
#   - .env with SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, OPENAI_API_KEY
#   - For --use-hybrid: migration 006_rag_governance.sql must be applied
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.logging import get_logger
from app.db.supabase import get_supabase_client
from app.domain.retrieval.service import RetrievalService

logger = get_logger(__name__)


# ============================================================
# IR metric implementations
# ============================================================

def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of top-k retrieved chunks that are relevant (score >= 1)."""
    top_k = retrieved[:k]
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant chunks found in top-k retrieved."""
    if not relevant:
        return 0.0
    top_k = retrieved[:k]
    hits = sum(1 for cid in top_k if cid in relevant)
    return hits / len(relevant)


def mrr(retrieved: list[str], relevant: set[str]) -> float:
    """Mean Reciprocal Rank: 1/rank of first relevant chunk (0 if none found)."""
    for rank, cid in enumerate(retrieved, start=1):
        if cid in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(
    retrieved: list[str],
    relevance_scores: dict[str, int],
    k: int,
) -> float:
    """
    Normalized Discounted Cumulative Gain at k.

    Uses actual relevance scores (2=RELEVANT, 1=PARTIAL, 0=not in gold).
    DCG = sum(score / log2(rank + 1)) for rank 1..k.
    IDCG = DCG of ideal ranking (best possible scores first).
    NDCG = DCG / IDCG.
    """
    def dcg(items: list[str], scores: dict[str, int], k: int) -> float:
        total = 0.0
        for rank, cid in enumerate(items[:k], start=1):
            gain = scores.get(cid, 0)
            total += gain / math.log2(rank + 1)
        return total

    actual_dcg = dcg(retrieved, relevance_scores, k)

    # Ideal: sort all annotated chunks by score descending
    ideal_order = sorted(relevance_scores.keys(), key=lambda c: relevance_scores[c], reverse=True)
    ideal_dcg = dcg(ideal_order, relevance_scores, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


# ============================================================
# Gold standard loader
# ============================================================

def load_gold_standard(path: str) -> list[dict]:
    """
    Load and validate the annotated gold standard.

    Skips entries where all chunk relevance values are still None (unannotated).
    Accepts both the flat list format and the wrapped format with _annotation_instructions.
    """
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if isinstance(raw, dict) and "entries" in raw:
        entries = raw["entries"]
    elif isinstance(raw, list):
        entries = raw
    else:
        raise ValueError(f"Unrecognised gold standard format in {path}")

    annotated = []
    skipped = 0
    for e in entries:
        chunks = e.get("retrieved_chunks", [])
        if not chunks:
            skipped += 1
            continue
        if all(c.get("relevance") is None for c in chunks):
            skipped += 1
            continue
        annotated.append(e)

    if skipped:
        logger.info(f"Skipped {skipped} unannotated entries in gold standard")

    if not annotated:
        raise ValueError(
            f"No annotated entries found in {path}. "
            "Fill in 'relevance' values (0/1/2) and save as gold_standard.json."
        )

    logger.info(f"Loaded {len(annotated)} annotated entries from {path}")
    return annotated


# ============================================================
# Per-query evaluation
# ============================================================

def evaluate_query(
    entry: dict,
    top_k: int,
    use_hybrid: bool,
    db: Any,
) -> dict:
    """
    Run RetrievalService for a single gold standard entry and compute metrics.

    Returns a dict with query metadata + all metric scores.
    """
    params = entry.get("retrieval_params", {})
    tenant_id = entry.get("tenant_id", "global")
    query = entry["user_query"]

    try:
        chunks = RetrievalService(db).retrieve(
            tenant_id=tenant_id,
            query=query,
            top_k=top_k,
            include_global=params.get("include_global", True),
            country=params.get("country"),
            framework_filter=params.get("framework_filter"),
            use_hybrid=use_hybrid,
        )
        retrieved_ids = [c.chunk_id for c in chunks]
    except Exception as e:
        logger.warning(f"RetrievalService failed for query '{query[:60]}...': {e}")
        retrieved_ids = []

    # Build relevance sets from gold standard annotation
    # relevant = score >= 1 (PARTIAL or RELEVANT)
    gold_chunks = entry.get("retrieved_chunks", [])
    relevance_scores: dict[str, int] = {}
    for c in gold_chunks:
        score = c.get("relevance")
        if score is not None:
            relevance_scores[c["chunk_id"]] = int(score)

    relevant_ids = {cid for cid, score in relevance_scores.items() if score >= 1}

    recall_k = max(top_k, 10)  # always compute recall at >= 10 to match spec

    return {
        "trace_id": entry.get("trace_id"),
        "user_query": query[:120],
        "tenant_id": tenant_id,
        "use_case": entry.get("use_case"),
        "params_source": entry.get("params_source"),
        "use_hybrid": use_hybrid,
        "retrieved_count": len(retrieved_ids),
        "gold_relevant_count": len(relevant_ids),
        f"precision_at_{top_k}": precision_at_k(retrieved_ids, relevant_ids, top_k),
        f"recall_at_{recall_k}": recall_at_k(retrieved_ids, relevant_ids, recall_k),
        f"ndcg_at_{recall_k}": ndcg_at_k(retrieved_ids, relevance_scores, recall_k),
        "mrr": mrr(retrieved_ids, relevant_ids),
        "retrieved_ids": retrieved_ids,
    }


# ============================================================
# Main eval function
# ============================================================

def run_eval(
    *,
    gold_standard_path: str,
    top_k: int,
    use_hybrid: bool,
    output_dir: str,
) -> dict[str, Any]:
    entries = load_gold_standard(gold_standard_path)
    db = get_supabase_client()

    mode = "hybrid" if use_hybrid else "vector"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(
        f"Starting eval: mode={mode} top_k={top_k} queries={len(entries)} "
        f"(each query calls OpenAI embeddings — check cost before large runs)"
    )

    results: list[dict] = []
    for i, entry in enumerate(entries, start=1):
        logger.info(f"[{i}/{len(entries)}] {entry['user_query'][:60]}...")
        result = evaluate_query(entry, top_k=top_k, use_hybrid=use_hybrid, db=db)
        results.append(result)

    if not results:
        logger.error("No results produced.")
        return {}

    # ---- Aggregate metrics ----
    recall_k = max(top_k, 10)
    metric_keys = [
        f"precision_at_{top_k}",
        f"recall_at_{recall_k}",
        f"ndcg_at_{recall_k}",
        "mrr",
    ]
    aggregates = {
        key: sum(r[key] for r in results) / len(results)
        for key in metric_keys
    }
    aggregates["mode"] = mode
    aggregates["top_k"] = top_k
    aggregates["n_queries"] = len(results)

    # ---- Write CSV ----
    csv_path = os.path.join(output_dir, f"eval_results_{mode}_{timestamp}.csv")
    fieldnames = [
        "trace_id", "user_query", "tenant_id", "use_case", "params_source",
        "use_hybrid", "retrieved_count", "gold_relevant_count",
        f"precision_at_{top_k}", f"recall_at_{recall_k}",
        f"ndcg_at_{recall_k}", "mrr",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        # Aggregate summary row first
        writer.writerow(
            {
                "trace_id": "AGGREGATE",
                "user_query": f"n={len(results)}",
                "tenant_id": "",
                "use_case": "",
                "params_source": "",
                "use_hybrid": use_hybrid,
                "retrieved_count": "",
                "gold_relevant_count": "",
                **{k: f"{aggregates[k]:.4f}" for k in metric_keys},
            }
        )
        for r in results:
            writer.writerow(
                {k: (f"{r[k]:.4f}" if isinstance(r.get(k), float) else r.get(k)) for k in fieldnames}
            )

    logger.info(f"Wrote CSV → {csv_path}")

    # ---- Write detail JSON ----
    detail_path = os.path.join(output_dir, f"eval_results_{mode}_{timestamp}_detail.json")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(
            {"aggregates": aggregates, "per_query": results},
            f,
            indent=2,
            ensure_ascii=False,
            default=str,
        )
    logger.info(f"Wrote detail JSON → {detail_path}")

    # ---- Print summary ----
    print(f"\n{'='*60}")
    print(f"  RAG Retrieval Eval — mode={mode.upper()}  (n={len(results)} queries)")
    print(f"{'='*60}")
    for key in metric_keys:
        print(f"  {key:<25} {aggregates[key]:.4f}")
    print(f"{'='*60}\n")
    print(f"  CSV:    {csv_path}")
    print(f"  Detail: {detail_path}")

    return aggregates


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate RAG retrieval quality against annotated gold standard"
    )
    parser.add_argument(
        "--gold-standard",
        type=str,
        required=True,
        help="Path to annotated gold standard JSON (e.g. datasets/gold_standard.json).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per query (default: 5). "
             "Precision@k uses this value; Recall and NDCG use max(top_k, 10).",
    )
    parser.add_argument(
        "--use-hybrid",
        action="store_true",
        help="Use hybrid search (vector + keyword RRF). "
             "Requires migration 006_rag_governance.sql to be applied.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Directory for output files (default: datasets/).",
    )
    args = parser.parse_args()

    run_eval(
        gold_standard_path=args.gold_standard,
        top_k=args.top_k,
        use_hybrid=args.use_hybrid,
        output_dir=args.output_dir,
    )
