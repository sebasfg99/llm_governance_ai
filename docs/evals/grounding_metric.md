# The grounding metric: from Jaccard to containment

> ## ⚠️ SUPERSEDED (2026-08) — see [`docs/rag/grounding_metric_v2.md`](../rag/grounding_metric_v2.md)
>
> This document describes the **lexical** era of `grounding_score()` and the
> `0.40` threshold. Both are gone: the metric is now embedding cosine and the
> threshold is `0.62`, because word overlap turned out to be *inverted* on a
> bilingual corpus — a fabricated answer scored 0.462 while a correct answer
> resting on English sources scored 0.242.
>
> Kept because the Jaccard → containment fix described below is still the
> reason the threshold moved off 0.15, and because the reasoning about why a
> length-dependent denominator is wrong still applies.
>
> **Every score in this document is on the old scale and is not comparable to
> anything produced after `evaluator_version = "v3.semantic"`.**

Status: **superseded**. Historically: `grounding_score()` measured containment,
`GROUNDING_THRESHOLD` was recalibrated to `0.40`, and the persisted evaluation
reuses the LLM judge's verdict instead of a lexical proxy (that last part still
holds).

---

## The bug

`grounding_score()` (`app/domain/evals/graders.py`) computed **Jaccard**
similarity:

```python
score = len(response_words & context_words) / len(response_words | context_words)
```

while its own docstring described **containment** — *"the proportion of words
in the response that also appear in the retrieved chunks"*. The code did not do
what it said it did.

Because the union sits in the denominator, the score was a function of **how
much context was retrieved**, not of how grounded the answer was. Two
consequences, both measured on the Normativa validation run:

**1. The threshold was arithmetically unreachable.** Case A had 78 unique
response words over 516 unique context words:

| | value |
|---|---|
| reported Jaccard | 0.0780 |
| **theoretical maximum Jaccard** | **0.1416** |
| `GROUNDING_THRESHOLD` | **0.15** |
| containment (what the docstring described) | 0.5513 |

No response could ever pass, regardless of quality.

**2. The incentive was inverted.** Retrieving *more* relevant context grew the
union and pushed the score *down*. The agent-scoped diversity cap fix — which
genuinely improved retrieval by bringing 3 ISO chunks instead of 1 — therefore
registered as a quality regression.

## The fix

```python
score = len(response_words & context_words) / len(response_words)
```

Containment is independent of context size, bounded in [0, 1], reaches 1.0 on a
fully grounded answer, and matches the documented intent.

### Why not just raise the threshold

Recalibrating a Jaccard threshold would have preserved the inverted incentive:
any future retrieval improvement that adds relevant context would still depress
the score. The problem was not a mis-tuned threshold on the right metric — it
was the wrong metric.

## Calibration

Eight labelled responses scored against their real retrieved context:

| case | score | label |
|---|---|---|
| verbatim extract from context | 1.0000 | grounded |
| paraphrase + framing | 0.9310 | grounded |
| real Case A answer | 0.5513 | grounded |
| real Case C answer | 0.4545 | grounded — **lowest positive** |
| — **threshold 0.40** — | | |
| real Case B1 abstention | 0.3750 | abstention (correctly low) |
| invented reg. reusing domain vocabulary | 0.3704 | ungrounded — **highest negative** |
| invented regulation ("artículo 47 bis") | 0.2667 | ungrounded |
| off-topic (recipe) | 0.2105 | ungrounded |

`GROUNDING_THRESHOLD = 0.40` sits inside the observed gap
`[0.3704, 0.4545]`.

**The margin is genuinely narrow.** A fluent invention that reuses the domain's
vocabulary (0.3704) scores only just below a real paraphrase (0.4545). This is
inherent to lexical overlap: it measures *vocabulary provenance*, not truth.
Treat it as a coarse gate. The LLM judge is the authority.

The hallucination fallback in `evaluator.py` was recalibrated in the same pass:
`grounding < 0.05` became `< 0.30` (`_HALLUCINATION_FALLBACK_GROUNDING`). Under
Jaccard scores were compressed toward zero, so 0.05 was meaningful; under
containment it was dead code. 0.30 flags the clearly-ungrounded cases while
staying below a legitimate abstention, deliberately more conservative than the
pass threshold because a false hallucination flag causes the guardrails to
rewrite the user-visible answer.

## The second bug: the judge's verdict was thrown away

`ai_service` ran the LLM judge once in `_run_governance_pipeline`, then called
`evaluate_trace(..., use_llm_judge=governance is None)` — i.e. `False` — to
avoid paying for the judge twice. But `evaluate_trace` then **recomputed** the
whole governance block with no judge available, so `evaluator.py` fell through
to:

```python
elif judge_result.get("ok") is False:
    context_adherence = grounding      # the lexical proxy
```

The persisted row, and the `passed` decision built on it, discarded the real
claim-level verdict the caller already held. The observable symptom was
`context_adherence` always being *exactly equal* to `grounding`, while gated at
`>= 0.7` — a value the proxy cannot reach.

Fixed by threading the verdict through: `evaluate_trace()` takes
`precomputed_governance`, persists it verbatim when present, and only computes
fresh (with the judge enabled) when it is absent or empty. No extra judge cost.

### Before / after on Case A

| | before | after |
|---|---|---|
| `grounding_score` | 0.0780 | **0.5455** |
| `context_adherence_score` | 0.0780 (= proxy) | **0.0000** (4 claims, 0 supported) |
| `claims_total` | 0 | **4** |
| `hallucination` | from `grounding < 0.05` | **from the judge** |
| `passed` | False (unreachable) | False (genuine) |

`passed` is still `False`, and that is now **correct rather than an artifact**:
the judge examined 4 claims and supported none of them, because the retrieved
context was ISO 42001's *bibliography*, not its transparency clauses. The metric
now reports the real problem instead of masking it.

That retrieval problem — front-matter and bibliography chunks outranking
substantive clauses — is the next task. See
[`../rag/industry_codes.md`](../rag/industry_codes.md) for the
diagnose-before-fixing precedent.

## Known inconsistency (pre-existing, not addressed here)

`grounding_score` (0.5455) and `groundedness_score` (0.4935) differ slightly on
the same trace. The inline governance block scores the **raw** LLM text, while
`evaluate_trace` scores `governance["safe_response"]` — the post-guardrails text,
which may carry a disclaimer or an abstention rewrite. Both are "correct" for
their own input; they simply are not the same input.
