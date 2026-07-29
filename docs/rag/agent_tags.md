# `agent_tags`: business-agent tagging and retrieval fallback

Status: **fully implemented** — schema + ingestion (migration
`009_agent_tags.sql`, `scripts/seed_documents.py`) *and* the retrieval
fallback (`app/domain/retrieval/service.py`), exposed on `POST /ai/query` via
the `agent_tags` request field.

Still out of scope: the orchestrator that picks the agent automatically. For
now `agent_tags` is set manually in the request body.

## What is implemented today

### Tags

Every document (and every one of its chunks) carries an `agent_tags TEXT[]`
column indicating which business agent(s) its content serves:

| Tag           | Agent                                        |
|---------------|----------------------------------------------|
| `normativa`   | Regulatory / compliance frameworks           |
| `seguridad`   | Security (OWASP, MITRE ATLAS, PCI, cyber)    |
| `proveedores` | Third-party / vendor governance              |
| `privacidad`  | Data protection and privacy                  |
| `general`     | Fallback for unrecognized/missing frameworks |

A document can carry several tags at once (e.g. GDPR →
`{privacidad, normativa, proveedores}`).

### How tags are assigned

Statically, from the framework slug — no text classification involved. The
single source of truth is `FRAMEWORK_AGENT_MAP` +
`agent_tags_for_framework()` in `scripts/seed_documents.py`, mirroring the
existing `industry_codes_for_framework()` pattern. Documents whose framework
is `None` or unmapped get `["general"]` (never an empty list) so they remain
retrievable as a fallback instead of disappearing from every agent.

The seed script writes `agent_tags` on the `documents` row *before* calling
`process_document()`, which denormalizes it (together with `industry_codes`)
onto every `document_chunks` row. Running `seed_documents.py --force`
re-tags the whole corpus, documents and chunks alike.

### How to filter

All four retrieval RPCs (`match_chunks`, `match_global_chunks`,
`match_chunks_hybrid`, `match_global_chunks_hybrid`) accept an optional
`p_agent_tags TEXT[] DEFAULT NULL` parameter:

- `NULL` or empty array → no agent filtering (pre-existing callers are
  unaffected).
- Non-empty → only chunks satisfying `agent_tags && p_agent_tags` (array
  overlap) are returned, served by the GIN index `idx_chunks_agent_tags`.

## Retrieval fallback (implemented)

Implemented in `RetrievalService.retrieve()`
(`app/domain/retrieval/service.py`). When a specific agent (e.g.
`proveedores`) runs its retrieval:

1. **Pass 1**: the normal pipeline (query expansion → embeddings →
   hybrid/vector search → RRF → rerank → diversity cap → threshold) runs with
   `p_agent_tags = ARRAY['proveedores']`, keeping the existing country +
   industry filters (all of which are applied in SQL — see
   [`industry_codes.md`](industry_codes.md)).
2. **Evaluate**: the top-1 score of pass 1 is compared against
   `settings.agent_fallback_threshold`. An empty pass-1 result scores `0.0`,
   which is the worst possible coverage and always broadens.
3. **Fallback**: if the top-1 score is below the threshold, retrieval re-runs
   with the union `p_agent_tags = ARRAY['proveedores', 'general']` (the
   union, not `'general'` alone), same country + industry filters.
4. Threshold and diversity cap apply to pass 2 as well — both passes go
   through the same `_run_retrieval_pass()` helper, so this holds by
   construction rather than by convention.
5. Pass 2's result is returned. Its tag set is a strict superset of pass 1's,
   so it can never see fewer candidates; there is no "best of the two"
   decision to make.

### The two knobs

| Knob | Where | Default | Effect |
|---|---|---|---|
| `agent_fallback_threshold` | `app/core/config.py`, env `AGENT_FALLBACK_THRESHOLD` | `0.45` | Score below which pass 1 is considered poor coverage. |
| `enable_general_fallback` | `AIQueryRequest` field, per request | `true` | Set `false` to force strictly agent-scoped retrieval. |

`agent_fallback_threshold` is deliberately **separate** from
`similarity_threshold` even though they share a default. `retrieve()` already
discards everything below `similarity_threshold`, so reusing that value would
make "top-1 below threshold" exactly equivalent to "pass 1 returned nothing" —
a degenerate trigger. Raising it (e.g. `AGENT_FALLBACK_THRESHOLD=0.60`) makes
the fallback also broaden on weak-but-passing results. Reranking
(`app/domain/retrieval/ranking.py`) only reorders; it is not a second cutoff.
`governance_strict_context_threshold` (0.7) is a separate knob, for
strict-context generation.

The fallback is **parameterizable per agent** via `enable_general_fallback`:
an agent whose own corpus is dense enough does not want its results diluted.
It is also skipped automatically when `general` is already among the requested
tags, since retrying would issue the identical query.

### Behaviour when `agent_tags` is absent

Requests that omit `agent_tags` (or send `[]`) are completely unaffected: the
`p_agent_tags` key is **not even included** in the RPC payload, so the request
stays valid against a database where migration 009 has not been applied. This
is what keeps `risk/service.py` and `policies/service.py` working untouched.

### Error policy

When `agent_tags` **is** set, a failing retrieval RPC raises `UpstreamError`
with an explicit migration-009 hint instead of degrading to an unfiltered
search. Silently returning the whole corpus while the caller believes it is
reading one agent's slice would be a correctness bug that no log would reveal.

Degrading from the *hybrid* RPCs to the *vector-only* RPCs is still allowed
(migration 006 remains optional), and the agent filter is carried across that
degradation on both the tenant and the global branch.

### Reading the logs

`RetrievalService` logs at INFO on every agent-scoped call:

```
Agent-tag pass 1: tags=['normativa'], candidates=12, returned=5, top1_score=0.612
Agent-tag fallback NOT triggered (top1=0.612, threshold=0.45, enabled=True, already_general=False, tags=['normativa'])
```

or, when coverage is poor:

```
Agent-tag pass 1: tags=['normativa'], candidates=3, returned=0, top1_score=0.000
Agent-tag fallback TRIGGERED: top1=0.000 < 0.45 → retrying with tags=['normativa', 'general']
Agent-tag pass 2: tags=['normativa', 'general'], candidates=27, returned=5, top1_score=0.688
```

The `candidates=` figure is the distinct-chunk pool before ranking. Comparing
it between an agent-scoped call and an unscoped one is the direct evidence
that the `agent_tags && ARRAY[...]` predicate is narrowing retrieval.

Requests without `agent_tags` emit no `Agent-tag` lines at all.

## Trying it out

Migration `009_agent_tags.sql` must be applied and the corpus re-tagged
(`python scripts/seed_documents.py --force`) — without the re-tag every chunk
has `agent_tags = '{}'` and *any* agent filter returns zero rows.

```json
POST /ai/query
{
  "query": "¿Cuáles son los requisitos de transparencia algorítmica según ISO 42001?",
  "country": "costa-rica",
  "industry": "financial",
  "agent_tags": ["normativa"]
}
```

To force the fallback for a demo, either ask about a topic the `normativa`
corpus does not cover, or set `AGENT_FALLBACK_THRESHOLD=0.95` and restart.
