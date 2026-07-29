# `industry_codes`: sector filtering, and the bug where it filtered nothing

Status: **fixed** by migration `010_industry_filter_fix.sql` + the SQL-side
filtering in `app/domain/retrieval/`. Before that migration the filter was a
silent no-op in every deployment.

---

## The diagnosis (state before migration 010)

Migration `008_industry_codes.sql` added `industry_codes TEXT[]` to
`documents` and `document_chunks` and backfilled the SUGEF/CONASSIF rows. It
**never touched the retrieval RPCs**. Three consequences stacked up:

| # | Where | What was wrong |
|---|---|---|
| a | `001`–`009` migrations | None of `match_chunks`, `match_global_chunks`, `match_chunks_hybrid`, `match_global_chunks_hybrid` had an industry parameter — no industry predicate ever reached SQL. |
| b | same | None of them listed `industry_codes` in their `RETURNS TABLE`, so the column never travelled to the client. |
| c | `hybrid.py` | `reciprocal_rank_fusion()` built `ChunkResult` without mapping `industry_codes` at all — so even if the RPC had returned it, the hybrid path would have dropped it. |
| d | `service.py` | `_search_tenant_chunks` / `_search_global_chunks` did read `row.get("industry_codes")`, but because of (b) the key was never present → always `None`. |

`RetrievalService._filter_by_industry` then applied this permissive rule:

```python
if not codes or "ALL" in codes or industry in codes:
    result.append(chunk)     # keep
```

Because (b)/(c)/(d) guaranteed `codes` was `None` for **every** chunk, the
first clause matched every time. The post-filter accepted the entire corpus.

**Net effect:** a cooperative-only institution could be served bank-only
regulatory content. No exception, no log line, no failing test — the unit
tests passed because they constructed chunks *with* codes, which production
never did. This is an accuracy bug of the same class as a hallucination, not
a tidiness issue.

There was also no GIN index on `document_chunks.industry_codes` (008 never
created one), unlike `agent_tags` which got `idx_chunks_agent_tags` in 009.

---

## The fix

Filter in SQL, exactly as `agent_tags` does. A metadata predicate must run
inside the query — a Python post-filter runs *after* fusion and ranking, by
which point an excluded chunk has already displaced a legitimate candidate
from the top-k, so even a *working* post-filter would silently shrink results.

### Database — `010_industry_filter_fix.sql`

- Adds `p_industry_codes TEXT[] DEFAULT NULL` to all four retrieval RPCs.
- Applies the predicate in **both branches** of the hybrid functions, so a
  sector-restricted chunk cannot re-enter the pool through a keyword match.
- Adds `industry_codes` to every `RETURNS TABLE`, so the client can verify the
  filter applied instead of trusting it blindly.
- Creates the GIN index `idx_chunks_industry_codes`.
- Drops **every** historical signature before recreating, so exactly one
  version of each function survives, and spells out argument types on
  `COMMENT ON FUNCTION` (the unqualified form in 004/005 errors out while an
  overload transiently exists).

### Filter semantics — deliberately permissive

| Condition | Result |
|---|---|
| `p_industry_codes` NULL / empty | no restriction (every pre-existing caller unaffected) |
| chunk `industry_codes` NULL / empty | always included (safety net for pre-008 rows) |
| chunk `industry_codes` contains `ALL` | always included (Layer 1-2: ISO, NIST, OWASP…) |
| otherwise | included only when the arrays overlap |

Only sector-restricted material is ever excluded; cross-sector material never
is.

### Python

- `hybrid_search()` / `hybrid_search_global()` and
  `_search_tenant_chunks()` / `_search_global_chunks()` take `industry_codes`
  and send `p_industry_codes` — **only when non-empty**, so payloads for
  callers that don't filter stay valid against a pre-010 database.
- `reciprocal_rank_fusion()` now maps `industry_codes` onto `ChunkResult`.
- `retrieve()` wraps the scalar `industry` into a one-element array.
- The hybrid→vector degradation carries the filter on **both** the tenant and
  the global branch.

### What replaced `_filter_by_industry`

It is gone as a filter. Two things took its place:

- `RetrievalService.chunk_matches_industry()` — the visibility rule as a pure
  predicate, the Python mirror of the SQL condition. It filters nothing; it
  exists so the rule is stated once and testable without a database.
- `RetrievalService._warn_on_industry_mismatch()` — a sanity check that
  **drops nothing** and logs a WARNING if the database returns a chunk that
  should have been filtered out.

Keeping a second Python filter would have re-created two sources of truth and
hidden the same class of failure again. The check reports instead:

```
Industry filter mismatch: 2 chunk(s) not applicable to industry='cooperative'
were returned by the database. The SQL filter (migration 010) may not be
applied. Offending chunks: [('abc…', ['bank']), …]
```

Chunks whose `industry_codes` come back as `None` are skipped by the check —
that means a pre-010 database not returning the column, which is a different
situation from the filter returning wrong rows.

### Error policy

When `industry` is set, a failing RPC raises `UpstreamError` naming migration
010 rather than degrading to an unfiltered search. Degrading silently is what
produced the original bug.

---

## Verification

`010` was applied against a real Postgres (pgvector pg16) on top of the full
`001`→`009` chain inside a transaction, then rolled back. Signature check
afterwards: exactly four functions, each ending in `p_industry_codes text[]
DEFAULT NULL`.

Functional results over a four-document corpus (`iso`=`{ALL}`,
`sugef_bank`=`{bank}`, `sugef_coop`=`{cooperative}`, `legacy`=`NULL`):

| Case | Requested | Returned |
|---|---|---|
| A | no filter | all 4 — additive, no regression |
| B | `{bank}` | iso, sugef_bank, legacy — **cooperative excluded** |
| C | `{cooperative}` | iso, sugef_coop, legacy — **bank excluded** |
| D | `{insurance}` | iso, legacy — both sector docs excluded |
| E | `{bank}`, hybrid RPC | same as B — both branches filter |
| F | `{normativa}` + `{bank}` | iso, sugef_bank — filters compose |

To re-check on Supabase after applying:

```sql
SELECT proname, pg_get_function_arguments(oid)
FROM pg_proc
WHERE proname IN ('match_chunks','match_global_chunks',
                  'match_chunks_hybrid','match_global_chunks_hybrid');

SELECT industry_codes, count(*) FROM document_chunks GROUP BY 1 ORDER BY 2 DESC;
```

Then run the same question twice via `POST /ai/query`, once with
`"industry": "cooperative"` and once without, and confirm bank-only content
appears only in the second.

Unit coverage: `app/tests/test_retrieval_industry.py` (the rule and the sanity
check) and `app/tests/test_agent_tags_retrieval.py` (`p_industry_codes`
plumbing, real exclusion, hybrid degradation symmetry).

---

## Related

- [`agent_tags.md`](agent_tags.md) — the same SQL-side filtering pattern, which
  this fix was modelled on.
