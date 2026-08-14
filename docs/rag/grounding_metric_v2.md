# Grounding metric v3: from word overlap to embedding cosine

Status: implemented. `grounding_score()` is semantic as of `evaluator_version = "v3.semantic"`.

> **File name note.** This document is `grounding_metric_v2.md` because that is
> the name the task specified, but the version it describes is **v3**. The
> evaluator was already on `v2.governance` (set by migration `006`), so the
> semantic change is the third generation, not the second.

---

## Why the metric changed

`grounding_score()` measured **word overlap** between the answer and the
retrieved chunks — the share of the answer's meaningful words that also appear
in the context.

On a bilingual corpus that metric is not merely weak. It is **inverted**.

Measured over 35 samples (`scripts/calibrate_grounding_metric.py`):

| sample group | n | **lexical** mean [min–max] | **semantic** mean [min–max] |
|---|---|---|---|
| A) real answer, Spanish sources | 10 | 0.717 [0.47–0.83] | 0.790 [0.70–0.87] |
| B) real answer, English sources | 10 | **0.242** [0.04–0.47] | 0.638 [0.50–0.74] |
| C) fabricated but plausible | 10 | **0.462** [0.24–0.62] | 0.597 [0.39–0.73] |
| D) off-topic (control) | 5 | **0.293** [0.24–0.35] | 0.148 [0.11–0.24] |

Read row B against rows C and D:

- A **fabricated** answer scored **0.462** while a **correct** answer whose
  evidence happened to be in English scored **0.242** — nearly half.
- Even **off-topic text** (a cooking recipe, a football result) scored
  **0.293**, still above the correct answer.

At the old gate of 0.40 this meant the system rejected correct answers about
MITRE ATLAS while accepting invented regulations.

### The mechanism

Answers are always Spanish (the system prompt requires it). Much of the corpus
is English: OWASP, MITRE ATLAS / SAFEAI, the PCI DSS English edition — 701 of
the 2,338 chunks tagged `seguridad`.

Containment measured against **English-only chunks was ~0.07**. Those chunks
occupied top-k slots and contributed essentially nothing to the score: the
metric silently shrank the effective context to the same-language subset.

A hypothesis that looked obvious and turned out to be **wrong**: that the
English-only stop-word list was to blame. Adding Spanish stop words moves
English-context and Spanish-context samples down by about the same amount
(−0.13 vs −0.10) and makes the gate worse (7/10 → 5/10 passing). It was
measured and discarded — do not retry it.

Combining the two metrics was also measured and discarded:
`max(lexical, semantic)` gives exactly the same operating point as semantic
alone (80% / 20% / +0.60), because semantic dominates wherever it matters.

---

## What it is now

```
score = max over chunks of cosine( embed(response), embed(chunk) )
```

Maximum rather than mean: a grounded answer usually rests on one or two of the
five chunks retrieved, and averaging would punish it for whatever else
retrieval returned.

Embeddings come from the same model as the corpus (`text-embedding-3-small`),
which is multilingual — that is what lets the score read evidence in either
language.

### Bands (semantic scale — **not** comparable to the old one)

| range | reading |
|---|---|
| ≥ 0.70 | well grounded |
| 0.62–0.70 | grounded; 0.62 is `GROUNDING_THRESHOLD` |
| 0.35–0.62 | weak — plausible invention lives here |
| < 0.35 | off-topic; measured max for off-topic was 0.148 |

---

## Threshold: 0.62

Sweep over the 35-sample set, 0.01 steps:

| threshold | keeps real | admits ungrounded | separation |
|---|---|---|---|
| 0.55 | 95% | 53% | +0.42 |
| 0.60 | 85% | 27% | +0.58 |
| **0.62** | **80%** | **20%** | **+0.60** ← maximum |
| 0.65 | 70% | 13% | +0.57 |
| 0.70 | 60% | 7% | +0.53 |

For comparison, the old lexical gate at 0.40 kept only **60%** of real answers
while admitting **40%** of ungrounded ones. The new gate is better on **both**
axes.

### What it still does not do

At 0.62 the gate admits 20% of plausible fabrications and rejects 20% of
genuine answers. Groups B (0.638) and C (0.597) overlap, because semantic
similarity measures **topical closeness, not factual support**: an invented
clause about the right subject scores close to a real one.

This is a coarse gate and a free fallback. The **LLM judge**
(`app/domain/evals/llm_judge.py`) remains the actual hallucination check.

---

## `_HALLUCINATION_FALLBACK_GROUNDING = 0.35`

`app/domain/evals/evaluator.py`. Fires only when the judge failed to run,
marking `hallucination=True` when grounding falls below it.

Calibrated against the 5 off-topic control samples, which scored 0.112–0.245
(mean 0.148). 0.35 sits above that ceiling, so genuinely off-topic output is
still caught, while **0 of 20 real answers** fall below it — no false
positives, which matters because the response guardrails annotate the
user-visible answer when this fires.

Deliberately **below** the main gate of 0.62: this is a floor detector, not
the gate. A coarse proxy should under-flag, not over-flag.

---

## Versioning — read before querying `evaluations`

New rows are written with `evaluator_version = "v3.semantic"`. Historical rows
keep `"v2.governance"` and **were not rewritten**.

The two scales are not comparable. Any dashboard, trend or alert over
`evaluations.grounding_score` **must filter by `evaluator_version`**, or it
will mix word-overlap values with cosine values and read the change as a step
in quality.

```sql
SELECT evaluator_version, COUNT(*), AVG(grounding_score)
FROM evaluations
GROUP BY evaluator_version;
```

At the time of the change: 592 rows on `v2.governance`, growing from 0 on
`v3.semantic`.

---

## Cost and latency

One embedding call for the response, plus one per chunk unless the caller
supplies vectors.

`grounding_score()` is called **twice per query** — once by
`evaluator.evaluate_response()` and once by `EvalService.evaluate_trace()` —
over the same response and chunks. A process-level cache keyed on the text
hash (`graders._EMBED_CACHE`, bounded at 512 entries) makes the second call
free.

`grounding_score()` accepts an optional `chunk_embeddings` argument. Chunk
vectors already exist in `document_chunks.embedding`; passing them skips
re-embedding entirely. The current call sites do not yet pass them — that is
the obvious next optimisation.

Latency is logged at DEBUG in `evaluator.evaluate_response()`:

```
grounding evaluation: 214ms (5 chunk(s))
```

### Degraded path

If the embedding call fails, `grounding_score()` falls back to the pre-v3
lexical containment and logs a warning. The returned value is then on the
**old scale** and will read low against 0.62 — deliberately, so a scoring
outage fails closed rather than silently passing everything.

---

## Effect on the e2e validators

| validator | before (lexical @0.40) | after (semantic @0.62) |
|---|---|---|
| Normativa (4 queries) | 3–4 / 4 | **4/4, 4/4, 4/4** |
| Seguridad, grounding check (6 queries) | 3/6 | **4/6** in 2 of 3 runs |

**Query 3 (MITRE ATLAS, English sources) clears the grounding gate for the
first time** — 0.663 / 0.647 / 0.647 across three runs, against 0.36–0.39 on
the lexical metric. That query is the reason this work happened.

Two Seguridad queries still fail the gate. **Neither is caused by this
change, and neither is fixed by it.** Both diagnoses below replace earlier
ones that were wrong and were corrected against direct SQL verification.

### Q2: encryption applied to AI

**Score:** 0.572–0.576 (below the 0.62 gate)

**Diagnosis:** residual EN↔ES cross-lingual gap.

**Verified evidence:**

- The Seguridad corpus holds **441 matches across 25 documents** mentioning
  encriptación / cifrado / AES / TLS / encrypt.
- Retrieval correctly returns 3 chunks of `SAFEAI_Full_Report` that discuss
  encryption of AI data and models directly.
- The semantic metric **improved** this case (~0.36 lexical → 0.572), but
  translating technical concepts between the English source and the Spanish
  answer still lands under 0.62.

**This is NOT a corpus gap.** An earlier version of this document claimed it
was. Ingesting more encryption content would not help — the best available
content is already being retrieved.

**Structural fix:** dual embedding (translate EN chunks to ES at ingestion and
store separate vectors). Roadmap, not scheduled.

### Q5: OWASP bibliography

**Score:** 0.472–0.495 (below the 0.62 gate)

**Diagnosis:** a retrieval problem, not a grounding one.

**Verified evidence:**

- The query asked for the references cited by the *OWASP GenAI Solutions
  Reference Guide*.
- Retrieval returned **zero chunks from that document**. Of the five
  retrieved: 3 from `PCI-DSS-v4_0_1-LA` (irrelevant — web attacks, RASP
  tooling), 1 from `SAFEAI_Full_Report`, 1 from
  `LLM_AI_Security_and_Governance_Checklist` (a *different* OWASP document).
- Zero chunks flagged `is_reference_list=true` reached the top-5, and only
  **one chunk in the entire corpus** of the requested document carries that
  flag.

**Root cause — a calibration bug in the reranker.** Confirmed by direct
measurement (`scripts/diagnose_retrieval_owasp.py`), not inferred.

`rerank_chunks()` in `app/domain/retrieval/ranking.py` fuses four features
with nominal weights that assume comparable scales. They are not comparable.
Measured over the 46-candidate pool for this query:

| feature | observed range | nominal weight | **effective influence** |
|---|---|---|---|
| `section_priority` | 0.00 – 1.00 | 0.10 | **0.100** ← largest |
| `similarity` | 0.446 – 0.591 | 0.55 | **0.080** |
| `length_quality` | 0.40 – 1.00 | 0.05 | 0.030 |
| `keyword_overlap` | 0.000 – 0.025 | 0.30 | **0.008** ← inert |

The order of real influence is the inverse of the intended one. Cosine
similarity from `text-embedding-3-small` is compressed into a narrow band
(0.145 wide here), while `section_priority` uses its full 0–1 range. Without
normalisation, the feature declared as a minor nudge outranks the one declared
as dominant, and `weight_section` behaves as a sort key rather than a
tie-breaker.

`_section_priority()` returns 1.0 when the chunk has a structured
`section_ref` (legal markers: "Artículo", "Anexo", "Cláusula"), 0.5 when the
content merely mentions an authority word, 0.0 otherwise. That is a property
of the **document type**, not of relevance to the query. Coverage is
effectively binary per document: PCI-DSS 99%, CONASSIF 99%, SAFEAI 98% versus
OWASP-GenAI 0%, Agentic-AI-Threats 0%.

**Evidence:** in the 46-candidate pool, the best OWASP-GenAI chunk scores
similarity **0.585 — 2nd best in the whole pool** — but lands at position
**24** of the fused ranking because `section_priority=0.00`. PCI-DSS chunks
scoring 0.523, ten points lower, rise to position **1** on
`section_priority=1.00`. Positions 1–21 of the fused ranking all have
`section_priority ≥ 0.50`; the first chunk without it sits at position 23 and
is the pool's **top hit by similarity**.

**This is NOT a corpus-volume problem.** An earlier version of this document
claimed "PCI DSS dominates by sheer size — 533 chunks against 165 for OWASP".
The pool says the opposite: **PCI contributes 13 candidates, OWASP 23**. OWASP
has more representation, not less. It loses in the rerank, not in retrieval.

(An earlier version before that claimed a list of standard titles was
"semantically diffuse". Also wrong — grounding cannot evaluate what retrieval
never returned.)

**Why Normativa never exposed this:** that corpus has near-universal
`section_ref` coverage, so the bonus is uniform across candidates and does not
distort their relative order. The bias only appears when documents with and
without legal structure compete — precisely the Seguridad case.

**Fix proposed:** normalise features before applying weights. Simulated in
`scripts/simulate_ranking_normalization.py`; implementation is separate work.

A second, independent obstacle remains for this query: the only chunk of the
requested document flagged `is_reference_list` (#69) scores 0.4457 against a
pool cutoff of 0.4458 — it misses the candidate pool by 0.0001. Fixing the
reranker surfaces OWASP content but not the bibliography chunk itself.

---

## What the semantic metric fixes, and what it does not

Worth stating plainly, because two of the three remaining Seguridad failures
were initially misattributed to this metric.

**It fixes:**

- **The inversion.** Off-topic text scoring above a correct answer on English
  sources — the defect that motivated the change — is gone.
- **Q3 (MITRE, pure cross-lingual with correct retrieval):** 0.36 → 0.647.
  Clears the gate for the first time.
- **Normativa stability:** 3–4/4 variable → 4/4 consistent.

**It does not fix:**

- **A calibration bug in the reranker.** Q5 fails for this reason: an
  unnormalised `section_priority` outweighs similarity, so documents without
  legal section markers lose regardless of relevance. See the Q5 section.
- **Residual cross-lingual distance on technical concepts.** Q2 fails for this
  reason: the right evidence is retrieved, in English, and the translated
  answer lands just under the gate.
- **Incomplete metadata.** `is_reference_list` is applied to one chunk of the
  OWASP corpus, which is not enough for bibliography queries to work.

These need separate work in retrieval, ingestion, and — for the cross-lingual
residual — possibly dual embedding. No threshold change fixes them: at a gate
low enough to admit Q2 and Q5, the fabricated-answer band (0.597 mean) comes
in with them.

---

## Related

- `scripts/calibrate_grounding_metric.py` — the 35-sample calibration study
- `scripts/diagnose_grounding_crosslingual.py` — the diagnosis, including the
  refuted stop-word hypothesis
- `app/domain/evals/graders.py` — the implementation
- `app/domain/evals/service.py` — `GROUNDING_THRESHOLD`, `EVALUATOR_VERSION`
- `app/domain/evals/evaluator.py` — `_HALLUCINATION_FALLBACK_GROUNDING`

---

## Known debt: SAFEAI's bibliography lost its flag (2026-08)

The migration-012 reprocess re-cut every boundary in the 7 documents
with a detectable front-matter heading. `is_reference_list` is
recomputed by the chunker on the new boundaries, and corpus-wide the
count moved **21 → 19**:

- **ISO 42001**: `#69` and `#70` → only `#70`. A *gain*: `#70` is the
  bibliography, and the old `#69` was a carry-forward false positive
  (substantive prose).
- **SAFEAI**: `#108` → **none**. A genuine loss.

### Mechanism

`APPENDIX I. REFERENCES` is still detected in the raw text (offset
207030, 99.1% of the document). But the +1 chunk shift left `#108`
mostly occupied by `APPENDIX H. ACRONYMS`, with the references
starting near its end:

```
#108  ...APPENDIX H. ACRONYMS AI Artificial Intelligence AIBOM ...
      ... [6] The MITRE Corporation, "MITRE ATLAS," 20 12 2024.   <- tail
#109  [Accessed 2 Jan 2025]. [5] NIST, "NIST Special Publication 800-30 R1"...
```

`#108` no longer clears the classifier's "≥60% of the chunk is
post-heading" rule, so it is not flagged; `#109` carries no heading of
its own and relies on carry-forward from `#108`, which never fired.
The chain breaks.

This is the SAME mixed-chunk problem that `is_front_matter` was built
to solve, appearing at the bibliography boundary instead of the body
boundary. The claim in `scripts/reprocess_front_matter.py` that
"`is_reference_list` is recomputed on the new boundaries, so the flags
stay correct automatically" was too optimistic and is disproved by
this case.

### The fix, not done here

Apply the same treatment: a HARD chunk cut at the reference heading,
mirroring `find_body_start`. `find_reference_headings()` already
returns the offsets and `chunk_text_with_sections()` already projects
them to word indices for the classifier — the missing piece is
forcing a boundary there instead of only classifying across it. On
cleanly cut chunks the flag becomes correct by construction, exactly
as it now is for front matter.

Not done in this cycle because it requires another corpus reprocess
and a full validation pass, and the current state passes the agreed
criterion (Normativa ≥4/4 in 2 of 3 runs). Scope when picked up:
`chunking.py`, a reprocess of the affected documents, and re-running
both e2e validators.

### Impact while it stands

The direction matters and an earlier draft of this section had it
backwards. Losing the flag does not make the bibliography
*unreachable* — it makes it **undemoted**. The penalty exists to stop
a reference list from crowding out substantive content, so with no
flag SAFEAI's citation list competes on equal terms with real
content. That is the original bibliography failure mode, back in one
document.

Measured on three probe queries (2026-08-14):

| query | `#109` in candidate pool | in top-5 |
|---|---|---|
| tácticas de MITRE ATLAS sobre model evasion | no | no |
| controles del framework SAFE-AI de MITRE | **yes** | no |
| protección contra ataques adversarios | no | no |

So the chunk does reach the candidate pool, and nothing demotes it
once there — it stayed out of the top-5 on merit, not by design. The
exposure is latent rather than active.

Unrelated to the Seguridad `Control bibliografía` query, which targets
the OWASP guide (still flagged) and fails for a different reason: its
flagged chunk sits 0.0001 below the candidate-pool cutoff.
