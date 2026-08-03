# `is_reference_list`: stopping bibliographies from outranking content

Status: implemented by migration `011_reference_list_flag.sql`, the classifier
in `app/domain/documents/reference_sections.py`, and the penalty in
`rerank_chunks()`.

---

## The problem

A bibliography is a list of standard **titles**. Embedded, it lands
semantically closer to a question about those standards than the substantive
clause that mentions them once in passing.

Measured on the ISO 42001 replica:

| chunk | `section_ref` | what it is | rank by cosine similarity |
|---|---|---|---|
| `idx=64` | `ANEXO C` | the transparency & explicability clause that **answers the question** | **227th** |
| `idx=69` | `ANEXO D` | `BIBLIOGRAFÍA` + list of ISO/IEC standard names | 3rd |
| `idx=70` | `ANEXO D` | continuation of that list | 5th |

The answer built from that context supported **0 of 4 claims**.

The mechanism is easy to reproduce. Ask *"¿Cómo aborda ISO 42001 el sesgo en
los sistemas de IA y la toma de decisiones asistida?"* and the bibliography
takes vector ranks **1 and 3** — because the query matches the entry
`ISO/IEC TR 24027:2021 — Bias in AI systems and AI aided decision making`
almost word for word. The list of titles wins against the prose that actually
discusses bias.

---

## Why detection is not obvious

Three candidate signals were measured across the whole 3,598-chunk corpus
before choosing one.

### `section_ref` — useless

`idx=64`, `idx=69` and `idx=70` all report `ANEXO C`/`ANEXO D`. The chunker's
section regex has no notion of a bibliography, so the field cannot separate
them.

### A heading match alone — too many false positives

`referencias`/`references` appears constantly **inside running prose** in the
Spanish regulatory corpus:

> "Toda referencia en la reglamentación emitida por el CONASSIF…"
> "Para las referencias con fecha, solo se aplica la edición citada."

and the NIST AI RMF Playbook carries **72** per-subsection `References` blocks.
Flagging on the word alone demotes ~45% of that document.

### Citation density alone — also too many false positives

Ranking the corpus by "fraction of characters inside an ISO/NIST/RFC
identifier" puts **vendor comparison tables, tables of contents and diagram
captions above the real bibliographies**. The single densest non-bibliography
chunk (`OWASP-GenAI #67`, a vendor matrix) outscores every genuine reference
list.

---

## What actually works

The conjunction of three signals, applied **at chunking time**:

1. **A heading anchored to its own line** in the raw document text.
   Anchoring with `^…$` is what rejects both the inline-prose case and TOC
   entries (`— Bibliografía 63`, `Referencias normativas ……… 12`), because
   neither has the line *ending* at the heading.

   This is why classification lives in the chunker: `chunk_text_with_sections()`
   joins words with spaces, so by the time a chunk exists the line structure
   that distinguishes a heading from the same word in a sentence is gone.

2. **The reference material must dominate the chunk** (≥ 60% of its words).
   A chunk that is mostly substantive with a short reference block at the end
   keeps its full score.

3. **The text after the heading must not read as prose**, measured as
   function-word ratio. Connected Spanish/English prose runs 27–50% function
   words; lists of titles, author names and identifiers run 6–20%. Corpus
   means: **0.18** for reference regions vs **0.38** for substantive ones.

### Continuation chunks

A bibliography longer than one chunk produces a second chunk with **no heading
in it at all** — this is exactly `idx=70`. `ReferenceRegionClassifier` carries
state across chunks in document order to catch these.

Carry-forward is deliberately conservative: the region survives a chunk
boundary only if the chunk *opens* as a list (first 40 words) **and** is
non-prose overall. Averaging over the whole chunk alone was not enough — the
Playbook's bulleted-question style sits near the threshold, and whole-chunk
averaging leaked the flag into eight chunks of real content.

---

## The penalty: demote, never exclude

`rerank_chunks()` multiplies the **fused** score of a flagged chunk by
`settings.retrieval_reference_penalty` (default `0.25`).

Applied to the fused score, not to similarity alone — otherwise a bibliography
recovers the lost ground through the section and length terms, since it
carries a `section_ref` and sits in the ideal length band.

**Nothing is deleted and nothing is filtered in SQL.** *"¿Qué normas ISO cita
el Anexo D de ISO 42001?"* is a legitimate question whose only correct answer
**is** the bibliography. So `rerank_chunks()` waives the penalty entirely when
the query is itself reference-seeking (`_REFERENCE_QUERY_PATTERNS`).

That waiver vocabulary is deliberately narrow — matching bare `norma` or
`standard` would waive the penalty for most of this corpus's questions.

This is also why the flag is **returned** by the RPCs but never used as a SQL
predicate: only the reranker can see the query, and the query is what decides
whether the bibliography is noise or the answer.

---

## Results

Classification over the whole corpus: **21 of 3,598 chunks (0.6%)** flagged,
across 4 documents.

| document | flagged chunks |
|---|---|
| `ISO42001_Replica.pdf` | `69`, `70` |
| `SAFEAI_Full_Report.pdf` | `108` |
| `AI_RMF_Playbook.pdf` | 17 chunks (each ≥60% academic citations) |
| `OWASP-GenAI-…-Reference-Guide` | `69` |

Ranking, on four queries that reproduce the inversion — the bibliography is
evicted from the top-5 in every case and real content is promoted in:

| query | reference chunks in top-5 before | after |
|---|---|---|
| sesgo en sistemas de IA | `[70, 69]` | `[]` |
| calidad de datos, privacidad y seguridad | `[70]` | `[]` |
| marcos de gestión de riesgos y terminología | `[69, 70]` | `[]` |
| gestión del riesgo y gobernanza de TI | `[69]` | `[]` |

Control — the bibliography stays reachable for reference-seeking queries, and
the waiver is what keeps it there. Measuring this needs care: comparing
`rerank_chunks(...)` against `rerank_chunks(..., reference_penalty=1.0)` is
**tautological**, because the waiver already forces the multiplier to `1.0`
and both sides compute the same thing. Suppressing the waiver so the penalty
genuinely lands shows what it costs:

| query | rank with waiver | rank if the penalty applied |
|---|---|---|
| ¿Qué normas ISO cita el Anexo D? | `70`→1, `69`→4 | 29, 30 |
| Muéstrame la bibliografía | `70`→1, `69`→2 | 29, 30 |
| ¿Cuáles son las normas citadas? | `70`→6, `69`→22 | 29, 30 |

Without the waiver the bibliography sinks to the bottom of a 30-candidate
pool; with it, it wins the top slots on the two specific questions. (The third
query is vague — it names no document — so it ranks 6th on retrieval merit
alone; that is not something this fix controls.)

LLM judge, same model and prompt, context as the only variable:

| | claims supported | hallucination |
|---|---|---|
| before | 14 / 15 | 1 query flagged |
| after | 13 / 13 | none |

The clearest single case is *"¿Qué marcos de gestión de riesgos y terminología
de IA se relacionan con ISO 42001?"*: with the bibliography in context the
model read a list of standard names and asserted an unsupported relationship
about ISO 31000 (`hallucination=True`, 3/4 supported); without it, 2/2
supported and no hallucination.

---

## Operating it

Migration `011` defaults every existing row to `FALSE` — "ordinary content",
the safe default, so nothing is demoted without cause. The feature is inert
until the corpus is classified.

**Re-embedding is not required.** Nothing here changes chunk text: the same
document produces the same chunks in the same order, only the flag is missing.

```bash
# preview (works before the migration is applied)
PYTHONPATH=. .venv/bin/python scripts/backfill_reference_flags.py --dry-run

# write flags (requires migration 011)
PYTHONPATH=. .venv/bin/python scripts/backfill_reference_flags.py
```

The backfill re-parses and re-chunks each document in memory, **verifies the
recomputed text still matches what is stored**, then updates
`is_reference_list` alone. Embeddings are never read or written. A document
whose text no longer matches is skipped and reported rather than written with
flags belonging to different text.

To disable the behaviour without reverting anything, set
`RETRIEVAL_REFERENCE_PENALTY=1.0`.

### Related scripts

- `scripts/diagnose_reference_sections.py` — the Step 1 signal comparison
- `scripts/validate_reference_penalty.py` — classification, Case A, control
- `scripts/judge_reference_penalty.py` — LLM-judge before/after
