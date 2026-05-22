# ============================================================
# app/domain/evals/llm_judge.py — LLM-as-judge for RAG evaluation
#
# WHY THIS FILE EXISTS:
# Heuristic graders (Jaccard overlap, length checks) are cheap
# but unreliable for governance use cases. An LLM judge can:
#   - Decompose a response into atomic claims
#   - Decide which claims are supported by retrieved chunks
#   - Detect hallucinated facts that pure overlap misses
#   - Detect when the model should have abstained ("IDK")
#
# WHY gpt-4o-mini:
# Same provider as the production model (no extra deps), low
# latency, low cost (~$0.0002 per call). For 10k calls/day that
# is roughly $2/day for the judge layer.
#
# OUTPUT CONTRACT:
# The judge returns a strict JSON object that the evaluator
# consumes. Every parse failure falls back to a safe default
# (no claims found, no hallucination flagged) so the pipeline
# never blocks on a malformed judge response.
# ============================================================

from __future__ import annotations

import json
import re
from typing import Any

import openai as openai_sdk

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


# Module-level singleton — the SDK client is thread-safe and cheap to keep
# around. We mirror the pattern in app/domain/documents/embeddings.py.
_judge_client: openai_sdk.OpenAI | None = None


def _get_judge_client() -> openai_sdk.OpenAI:
    global _judge_client
    if _judge_client is None:
        _judge_client = openai_sdk.OpenAI(api_key=settings.openai_api_key)
    return _judge_client


# ============================================================
# Judge prompt
#
# The prompt is intentionally explicit about the JSON shape so
# the model produces parseable output without retries. We pass
# the chunks numbered [1], [2], ... so the judge can attribute
# each claim to a chunk index.
# ============================================================
_JUDGE_SYSTEM_PROMPT = """You are an evaluator for a Retrieval-Augmented Generation (RAG) system in a regulated AI governance platform. Your job is to audit one (query, retrieved_chunks, answer) tuple.

You MUST output a single JSON object with this exact shape and nothing else:

{
  "claims": [
    {
      "text": "<one atomic factual claim from the answer>",
      "supported": <true|false>,
      "supporting_chunk_indexes": [<int>, ...],
      "is_hallucination": <true|false>
    }
  ],
  "should_have_said_idk": <true|false>,
  "reasoning": "<2-3 sentences max, plain text, in Spanish>"
}

Definitions:
- "claim": an independently verifiable factual statement. Greetings, hedges ("creo que..."), and meta-commentary are NOT claims.
- "supported": TRUE if the claim is directly entailed by at least one retrieved chunk, OR if it is a faithful paraphrase or conceptual equivalent of content in at least one chunk. Synonyms, reformulations, and language translations (ES↔EN) of the same factual content count as support. Common sense, the model's training data, and the user query alone do NOT count as support.
- "is_hallucination": TRUE when the claim is stated as fact AND is not supported by any chunk (neither literally nor by semantic equivalence). FALSE if the answer explicitly hedges or attributes uncertainty, or if the claim is a reasonable synthesis/paraphrase of the retrieved content.
- "should_have_said_idk": TRUE when the retrieved chunks do not contain enough information to answer, AND the answer fails to acknowledge that gap.

Rules:
- Do NOT include markdown, code fences, or text outside the JSON.
- Keep "claims" empty if the answer is purely conversational (e.g., "Hola, ¿en qué te ayudo?").
- If the answer correctly says it does not know, set "should_have_said_idk" = false (it abstained correctly).
"""


# ============================================================
# Public entrypoint
# ============================================================
def judge_response(
    *,
    query: str,
    response: str,
    chunks: list[str],
    instructions: str | None = None,
    model: str = "gpt-4o-mini",
) -> dict[str, Any]:
    """
    Run the LLM judge on one RAG response.

    Args:
        query:        The user query.
        response:     The LLM's answer to evaluate.
        chunks:       The retrieved chunk contents in the order they
                      appeared in the prompt. The judge will reference
                      them by 1-based index.
        instructions: Optional system-level instructions that the LLM
                      was supposed to follow (used only for context).
        model:        OpenAI model name. Defaults to gpt-4o-mini.

    Returns:
        A dict with keys:
          - "claims":               list of claim dicts (see prompt).
          - "claims_total":         len(claims)
          - "claims_supported":     count of supported claims
          - "hallucination":        bool — any claim flagged as hallucination
          - "should_have_said_idk": bool
          - "reasoning":            short Spanish explanation
          - "raw":                  the raw judge JSON (for audit)
          - "ok":                   True if the judge ran successfully
    """
    if not response or not response.strip():
        return _empty_judge_result(reason="empty_response")

    chunks_block = _format_chunks(chunks)
    instructions_block = (
        f"\n\nSystem instructions the answerer was supposed to follow:\n{instructions.strip()}"
        if instructions
        else ""
    )
    user_prompt = (
        f"User query:\n{query.strip()}\n\n"
        f"Retrieved chunks:\n{chunks_block}\n\n"
        f"Answer to audit:\n{response.strip()}"
        f"{instructions_block}\n\n"
        "Return the JSON now."
    )

    try:
        completion = _get_judge_client().chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=900,
            # response_format forces valid JSON for compatible models. We
            # still defend against malformed output below for robustness.
            response_format={"type": "json_object"},
        )
    except Exception as e:
        logger.warning(f"LLM judge call failed: {e}")
        return _empty_judge_result(reason=f"judge_error: {e}")

    raw_text = (completion.choices[0].message.content or "").strip()
    parsed = _safe_parse_json(raw_text)
    if parsed is None:
        logger.warning(
            f"LLM judge returned unparseable JSON. First 200 chars: {raw_text[:200]!r}"
        )
        return _empty_judge_result(reason="json_parse_error", raw=raw_text)

    return _normalize_judge_output(parsed)


# ============================================================
# Helpers
# ============================================================
def _format_chunks(chunks: list[str]) -> str:
    """Render chunks as a numbered block for the judge prompt."""
    if not chunks:
        return "(no chunks were retrieved)"
    lines: list[str] = []
    for i, c in enumerate(chunks, start=1):
        text = (c or "").strip()
        if len(text) > 1500:
            text = text[:1500] + "..."
        lines.append(f"[{i}] {text}")
    return "\n\n".join(lines)


def _safe_parse_json(text: str) -> dict | None:
    """
    Parse the judge output, tolerating leading/trailing fences.
    Returns None if parsing fails — callers should fall back to
    the empty result.
    """
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None


def _normalize_judge_output(parsed: dict) -> dict[str, Any]:
    """
    Coerce the parsed JSON into the strict shape expected by callers.
    Missing or wrong-type fields fall back to safe defaults so a
    sloppy judge response never crashes the pipeline.
    """
    raw_claims = parsed.get("claims", [])
    claims: list[dict[str, Any]] = []
    if isinstance(raw_claims, list):
        for c in raw_claims:
            if not isinstance(c, dict):
                continue
            claims.append(
                {
                    "text": str(c.get("text", "")).strip(),
                    "supported": bool(c.get("supported", False)),
                    "supporting_chunk_indexes": [
                        int(x)
                        for x in (c.get("supporting_chunk_indexes") or [])
                        if isinstance(x, (int, float))
                    ],
                    "is_hallucination": bool(c.get("is_hallucination", False)),
                }
            )

    claims_total = len(claims)
    claims_supported = sum(1 for c in claims if c["supported"])
    hallucination = any(c["is_hallucination"] for c in claims)

    return {
        "claims": claims,
        "claims_total": claims_total,
        "claims_supported": claims_supported,
        "hallucination": hallucination,
        "should_have_said_idk": bool(parsed.get("should_have_said_idk", False)),
        "reasoning": str(parsed.get("reasoning", "")).strip(),
        "raw": parsed,
        "ok": True,
    }


def _empty_judge_result(*, reason: str, raw: str | None = None) -> dict[str, Any]:
    """
    Safe default when the judge cannot run or produce parseable JSON.
    The pipeline degrades gracefully: zero claims, no hallucination
    flagged, and the heuristic graders take over for scoring.
    """
    return {
        "claims": [],
        "claims_total": 0,
        "claims_supported": 0,
        "hallucination": False,
        "should_have_said_idk": False,
        "reasoning": f"judge_unavailable: {reason}",
        "raw": {"error": reason, "raw_text": raw} if raw else {"error": reason},
        "ok": False,
    }
