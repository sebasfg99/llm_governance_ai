#!/usr/bin/env python3
# ============================================================
# scripts/export_finetune_data.py — Export fine-tuning dataset
#
# WHY THIS SCRIPT EXISTS:
# After the system has been running and collecting traces +
# evaluations + feedback, you can export the high-quality
# responses as a fine-tuning dataset for OpenAI.
#
# Fine-tuning teaches the model HOW to respond (style, format,
# tone) — NOT what to know. Use traces where:
#   - eval.passed = true (good quality heuristic graders)
#   - feedback.rating >= 4 (user-approved responses)
#
# OUTPUT FORMAT: OpenAI fine-tune JSONL
# Each line is a JSON object:
# {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}
#
# HOW TO RUN:
#   PYTHONPATH=. python scripts/export_finetune_data.py \
#     --use-case default_rag \
#     --min-rating 4 \
#     --output finetune_export.jsonl
# ============================================================

import sys
import os
import json
import argparse
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.supabase import get_supabase_client
from app.core.logging import get_logger

logger = get_logger(__name__)


def export_finetune_data(
    use_case: str | None,
    min_rating: int,
    output_path: str,
    limit: int = 1000,
) -> None:
    """
    Export approved traces as OpenAI fine-tuning JSONL.

    Filters traces by:
      1. eval.passed = true (heuristic quality check passed)
      2. feedback.rating >= min_rating (user approved)

    Args:
        use_case:    Filter to specific use case, or None for all.
        min_rating:  Minimum feedback rating (1-5). Default 4.
        output_path: Path to write the .jsonl output file.
        limit:       Max number of examples to export.
    """
    db = get_supabase_client()

    logger.info(
        f"Exporting fine-tuning data: "
        f"use_case={use_case or 'all'}, "
        f"min_rating={min_rating}, "
        f"output={output_path}"
    )

    # --- Step 1: Get traces with passing evaluations ---
    # Join traces with evaluations where passed = true
    trace_query = (
        db.table("traces")
        .select("id, user_query, prompt_text, response_text, use_case, model_provider")
        .limit(limit * 2)  # Fetch extra, filter down after joining feedback
    )

    if use_case:
        trace_query = trace_query.eq("use_case", use_case)

    traces_result = trace_query.execute()
    traces = {t["id"]: t for t in (traces_result.data or [])}

    if not traces:
        logger.info("No traces found matching criteria.")
        return

    # --- Step 2: Get passing evaluations ---
    evals_result = (
        db.table("evaluations")
        .select("trace_id, passed")
        .eq("passed", True)
        .in_("trace_id", list(traces.keys()))
        .execute()
    )
    passing_trace_ids = {e["trace_id"] for e in (evals_result.data or [])}

    # --- Step 3: Get feedback with sufficient rating ---
    feedback_result = (
        db.table("feedback")
        .select("trace_id, rating")
        .gte("rating", min_rating)
        .in_("trace_id", list(traces.keys()))
        .execute()
    )
    approved_trace_ids = {f["trace_id"] for f in (feedback_result.data or [])}

    # --- Step 4: Find traces that pass BOTH filters ---
    qualified_ids = passing_trace_ids & approved_trace_ids
    qualified_traces = [
        traces[tid] for tid in qualified_ids
        if tid in traces
    ][:limit]

    if not qualified_traces:
        logger.info(
            "No traces found that pass both evaluation AND feedback filters. "
            f"Passing evals: {len(passing_trace_ids)}, "
            f"Approved feedback: {len(approved_trace_ids)}"
        )
        return

    logger.info(f"Found {len(qualified_traces)} qualified examples for export")

    # --- Step 5: Format as OpenAI fine-tune JSONL ---
    examples = []
    for trace in qualified_traces:
        # OpenAI fine-tuning format: messages array with system, user, assistant
        example = {
            "messages": [
                {
                    "role": "system",
                    # Use the full prompt text (already includes system instructions)
                    # In a production setup, you might extract just the system portion
                    "content": "You are an expert document analyst.",
                },
                {
                    "role": "user",
                    "content": trace["user_query"],
                },
                {
                    "role": "assistant",
                    "content": trace["response_text"],
                },
            ],
            # Metadata (not part of the JSONL format, kept for reference)
            "_metadata": {
                "trace_id": trace["id"],
                "use_case": trace["use_case"],
                "model": trace["model_provider"],
            },
        }
        examples.append(example)

    # --- Step 6: Write to JSONL file ---
    # JSONL = one JSON object per line (required by OpenAI fine-tuning API)
    with open(output_path, "w", encoding="utf-8") as f:
        for example in examples:
            # Write only the messages (not the metadata) to the fine-tune file
            fine_tune_entry = {"messages": example["messages"]}
            f.write(json.dumps(fine_tune_entry, ensure_ascii=False) + "\n")

    logger.info(
        f"\nExport complete!"
        f"\n  File: {output_path}"
        f"\n  Examples: {len(examples)}"
        f"\n  Use case filter: {use_case or 'all'}"
        f"\n  Min rating: {min_rating}"
        f"\n\nTo start fine-tuning with OpenAI:"
        f"\n  openai api fine_tuning.jobs.create -t {output_path} -m gpt-4o-mini"
    )

    # Print basic stats
    use_case_dist: dict[str, int] = {}
    for ex in examples:
        uc = ex["_metadata"]["use_case"]
        use_case_dist[uc] = use_case_dist.get(uc, 0) + 1

    logger.info(f"Use case distribution: {use_case_dist}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Export fine-tuning dataset from approved traces"
    )
    parser.add_argument(
        "--use-case",
        type=str,
        default=None,
        help="Filter to a specific use case (e.g., 'default_rag'). Omit for all.",
    )
    parser.add_argument(
        "--min-rating",
        type=int,
        default=4,
        help="Minimum feedback rating to include (1-5, default: 4)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=f"finetune_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        help="Output .jsonl file path",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum number of examples to export (default: 1000)",
    )
    args = parser.parse_args()

    export_finetune_data(
        use_case=args.use_case,
        min_rating=args.min_rating,
        output_path=args.output,
        limit=args.limit,
    )
