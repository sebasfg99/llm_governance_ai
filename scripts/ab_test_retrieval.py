#!/usr/bin/env python3
# ============================================================
# scripts/ab_test_retrieval.py — A/B test: Vector-only vs Hybrid retrieval
#
# PURPOSE:
# Compare two retrieval configurations on the same annotated gold standard:
#   System A: use_hybrid=False  (vector-only, current production default)
#   System B: use_hybrid=True   (hybrid vector + keyword via RRF)
#
# CrossEncoder reranking is NOT included here — it is not yet implemented.
# Current reranking in app/domain/retrieval/ranking.py is heuristic
# (similarity + keyword overlap + section priority + length quality).
# A future script will A/B test reranker vs hybrid baseline (Week 2b).
#
# METRIC DISTINCTION:
#   Metrics here (Precision@k, Recall@k, NDCG@k, MRR) are IR benchmark
#   metrics against the annotated gold standard — NOT the governance-pipeline
#   `retrieval_precision` metric (which measures similarity >= threshold).
#
# COST NOTE:
# Each query triggers TWO OpenAI embedding calls (one per system).
# For 30 queries: < $0.002 total (text-embedding-3-small).
#
# HYBRID PREREQUISITE:
# migration 006_rag_governance.sql must be applied for match_chunks_hybrid RPC.
# If not applied, System B falls back gracefully to vector-only with a warning
# and the A/B comparison is skipped.
#
# HOW TO RUN:
#   PYTHONPATH=. .venv/bin/python scripts/ab_test_retrieval.py \
#     --gold-standard datasets/gold_standard.json
#
# EXPECTED OUTPUT:
#   datasets/ab_test_results.csv
#   Console summary table with absolute and % deltas per metric
#
# PREREQUISITES:
#   .env with SUPABASE_URL, SUPABASE_SERVICE_ROLE_KEY, OPENAI_API_KEY
# ============================================================

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from typing import Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.core.logging import get_logger
from app.db.supabase import get_supabase_client
from app.core.errors import UpstreamError

# Reuse metric functions and gold standard loader from eval_retrieval
from scripts.eval_retrieval import (
    evaluate_query,
    load_gold_standard,
)

logger = get_logger(__name__)


def _check_hybrid_available(db: Any, entry: dict, top_k: int) -> bool:
    """
    Run a single query with use_hybrid=True to verify migration 006 is applied.
    Returns True if hybrid is available, False otherwise.
    """
    try:
        from app.domain.retrieval.service import RetrievalService
        params = entry.get("retrieval_params", {})
        RetrievalService(db).retrieve(
            tenant_id=entry.get("tenant_id", "global"),
            query=entry["user_query"],
            top_k=top_k,
            include_global=params.get("include_global", True),
            country=params.get("country"),
            framework_filter=params.get("framework_filter"),
            use_hybrid=True,
        )
        return True
    except UpstreamError as e:
        logger.warning(f"Hybrid search not available (migration 006 not applied?): {e}")
        return False
    except Exception as e:
        logger.warning(f"Unexpected error probing hybrid: {e}")
        return False


def run_ab_test(
    *,
    gold_standard_path: str,
    top_k: int,
    output_dir: str,
) -> dict[str, Any]:
    entries = load_gold_standard(gold_standard_path)
    db = get_supabase_client()
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    recall_k = max(top_k, 10)
    metric_keys = [
        f"precision_at_{top_k}",
        f"recall_at_{recall_k}",
        f"ndcg_at_{recall_k}",
        "mrr",
    ]

    # ---- Probe hybrid availability ----
    hybrid_available = _check_hybrid_available(db, entries[0], top_k)
    if not hybrid_available:
        print(
            "\nWARNING: Hybrid search is not available (match_chunks_hybrid RPC missing).\n"
            "Apply migration 006_rag_governance.sql and re-run.\n"
            "Showing System A (vector-only) results only.\n"
        )

    # ---- Run System A: vector-only ----
    logger.info(f"Running System A (vector-only) on {len(entries)} queries...")
    results_a: list[dict] = []
    for i, entry in enumerate(entries, start=1):
        logger.info(f"  A [{i}/{len(entries)}] {entry['user_query'][:60]}...")
        results_a.append(evaluate_query(entry, top_k=top_k, use_hybrid=False, db=db))

    agg_a = {k: sum(r[k] for r in results_a) / len(results_a) for k in metric_keys}

    # ---- Run System B: hybrid ----
    results_b: list[dict] = []
    agg_b: dict[str, float] = {}

    if hybrid_available:
        logger.info(f"Running System B (hybrid) on {len(entries)} queries...")
        for i, entry in enumerate(entries, start=1):
            logger.info(f"  B [{i}/{len(entries)}] {entry['user_query'][:60]}...")
            results_b.append(evaluate_query(entry, top_k=top_k, use_hybrid=True, db=db))
        agg_b = {k: sum(r[k] for r in results_b) / len(results_b) for k in metric_keys}

    # ---- Write CSV ----
    csv_path = os.path.join(output_dir, f"ab_test_results_{timestamp}.csv")
    rows: list[dict] = []
    for key in metric_keys:
        a_val = agg_a[key]
        b_val = agg_b.get(key)
        if b_val is not None:
            delta_abs = b_val - a_val
            delta_pct = (delta_abs / a_val * 100) if a_val > 0 else float("inf")
            rows.append(
                {
                    "metric": key,
                    "system_a_vector": f"{a_val:.4f}",
                    "system_b_hybrid": f"{b_val:.4f}",
                    "delta_abs": f"{delta_abs:+.4f}",
                    "delta_pct": f"{delta_pct:+.1f}%",
                }
            )
        else:
            rows.append(
                {
                    "metric": key,
                    "system_a_vector": f"{a_val:.4f}",
                    "system_b_hybrid": "N/A (migration 006 not applied)",
                    "delta_abs": "N/A",
                    "delta_pct": "N/A",
                }
            )

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["metric", "system_a_vector", "system_b_hybrid", "delta_abs", "delta_pct"],
        )
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Wrote A/B results → {csv_path}")

    # ---- Print summary table ----
    print(f"\n{'='*72}")
    print(f"  RAG A/B Test — Vector-only vs Hybrid  (n={len(entries)} queries, top_k={top_k})")
    print(f"{'='*72}")
    header = f"  {'Metric':<28} {'System A (vector)':>16} {'System B (hybrid)':>16} {'Delta':>10} {'Delta%':>8}"
    print(header)
    print(f"  {'-'*68}")
    for row in rows:
        print(
            f"  {row['metric']:<28} {row['system_a_vector']:>16} "
            f"{row['system_b_hybrid']:>16} {row['delta_abs']:>10} {row['delta_pct']:>8}"
        )
    print(f"{'='*72}")
    print(f"\n  Results saved to: {csv_path}")

    if hybrid_available and agg_b:
        improvements = [k for k in metric_keys if agg_b.get(k, 0) > agg_a[k]]
        if improvements:
            print(
                f"\n  RECOMMENDATION: Hybrid improves {len(improvements)}/{len(metric_keys)} metrics. "
                f"Consider wiring use_hybrid=settings.retrieval_use_hybrid in AIService (Week 1c)."
            )
        else:
            print(
                "\n  RESULT: Vector-only performs >= Hybrid on all metrics. "
                "Keep use_hybrid=False in production until further tuning."
            )

    return {
        "system_a": agg_a,
        "system_b": agg_b or None,
        "csv_path": csv_path,
        "n_queries": len(entries),
    }


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="A/B test: Vector-only vs Hybrid retrieval on annotated gold standard"
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
        help="Number of chunks to retrieve per query per system (default: 5).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets",
        help="Directory for output files (default: datasets/).",
    )
    args = parser.parse_args()

    result = run_ab_test(
        gold_standard_path=args.gold_standard,
        top_k=args.top_k,
        output_dir=args.output_dir,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
