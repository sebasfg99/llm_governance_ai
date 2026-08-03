# ============================================================
# app/domain/retrieval/service.py — RAG retrieval service
#
# WHY THIS FILE EXISTS:
# This is the retrieval half of RAG (Retrieval-Augmented Generation).
# When a user asks a question, this service:
#
#   1. Converts the question into a vector (embedding)
#   2. Searches the database for the most similar text chunks
#   3. Optionally adds chunks from the global shared library
#   4. Re-ranks the results (stub in MVP)
#   5. Returns the top chunks for the LLM to use as context
#
# WHY RETRIEVAL IS THE MOST IMPORTANT PART OF RAG:
# The LLM can only answer well if it receives relevant context.
# Garbage in → garbage out. If the wrong chunks are retrieved,
# the LLM will either hallucinate or say "I don't know."
# Good retrieval is the single biggest lever for RAG quality.
#
# TENANT ISOLATION:
# The pgvector search is filtered by tenant_id at the DB level.
# A tenant's chunks can NEVER appear in another tenant's search results.
# ============================================================

import re

from supabase import Client

from app.core.config import settings
from app.core.logging import get_logger
from app.core.errors import UpstreamError

from app.domain.documents.embeddings import embed_single
from app.domain.retrieval.query_expansion import generate_bilingual_queries
from app.domain.retrieval.hybrid import (
    hybrid_search,
    hybrid_search_global,
    migration_hints,
)
from app.domain.retrieval.ranking import (
    ChunkResult,
    apply_per_document_cap,
    rerank_chunks,
)

# Tag used as the broadening target when an agent-scoped retrieval comes back
# weak. Documents whose framework is unknown or missing are tagged 'general'
# by scripts/seed_documents.py, so this set is the corpus-wide safety net that
# keeps an agent from answering "I have no information" purely because its own
# slice of the corpus is thin. See docs/rag/agent_tags.md.
GENERAL_AGENT_TAG = "general"

_INSTRUCTION_PREFIXES = re.compile(
    r"^(resumeme|resume|explícame|explicame|explica|qué es|que es|"
    r"cuéntame sobre|cuéntame acerca de|hablame de|háblame de|"
    r"describe|analiza|analízame|dime sobre|habla de|habla acerca de|"
    r"puedes explicar|puedes resumir|puedes describir)\s+(el|la|los|las|un|una|sobre|acerca de)?\s*",
    re.IGNORECASE,
)


def _extract_search_topic(query: str) -> str:
    cleaned = _INSTRUCTION_PREFIXES.sub("", query.strip())
    return cleaned if cleaned else query

# Logger for this module
logger = get_logger(__name__)


class RetrievalService:
    """
    Performs vector similarity search to find relevant document chunks.

    Uses Supabase RPC to call the match_chunks PostgreSQL function,
    which runs the pgvector cosine similarity search.

    Usage:
        db = get_supabase_client()
        service = RetrievalService(db)
        chunks = service.retrieve(
            tenant_id="acme",
            query="What are the termination clauses?",
            top_k=5,
        )
    """

    def __init__(self, db: Client) -> None:
        """
        Initialize with a Supabase client.

        Args:
            db: An initialized Supabase Client instance.
        """
        self.db = db

    def retrieve(
        self,
        tenant_id: str,
        query: str,
        top_k: int | None = None,
        include_global: bool = True,
        country: str | None = None,
        industry: str | None = None,
        framework_filter: list[str] | None = None,
        use_hybrid: bool = False,
        agent_tags: list[str] | None = None,
        enable_general_fallback: bool = True,
    ) -> list[ChunkResult]:
        """
        Find the most relevant document chunks for a user query.

        Steps:
          1. Embed the query text into a vector
          2. Call match_chunks RPC for tenant-specific chunks
          3. Optionally call match_global_chunks for shared library
          4. Merge, deduplicate, and re-rank results
          5. (industry / agent / framework filtering all happen in SQL,
             inside the RPCs of step 2-3 — never as a post-filter)
          6. Return top_k results

        When `agent_tags` is supplied, the whole of the above becomes "pass 1"
        and a second, broader pass may follow — see the AGENT-TAG FALLBACK
        section below.

        Args:
            tenant_id:        Only search this tenant's documents.
            query:            The user's question or search text.
            top_k:            Max number of chunks to return.
                              Defaults to settings.retrieval_top_k (5).
            include_global:   If True, also search the global shared library
                              and merge results. Default: True.
            country:          Country filter (e.g. "costa-rica").
            industry:         Institution industry code (e.g. "cooperative").
                              Sent to the RPCs as p_industry_codes and applied
                              in SQL (migration 010). Chunks tagged with
                              specific industry_codes are included only if
                              they match; chunks tagged ALL, or with no tag at
                              all, are always included. None → no restriction,
                              and the parameter is not sent.
            framework_filter: Optional list of framework slugs. If provided,
                              only chunks tagged with one of these frameworks
                              (or with NULL framework) are returned. None /
                              empty list means no framework restriction.
            agent_tags:       Optional business-agent slugs, e.g. ["normativa"].
                              None / [] → no agent restriction whatsoever; the
                              p_agent_tags RPC argument is not even sent, so
                              every pre-existing caller keeps its exact
                              behaviour. Non-empty → the SQL side keeps only
                              chunks whose agent_tags array overlaps this list.
            enable_general_fallback:
                              Whether a weak agent-scoped result may be retried
                              with the broader ["<agent>", "general"] set.
                              Ignored when agent_tags is None / [].

        AGENT-TAG FALLBACK (docs/rag/agent_tags.md):
            1. Pass 1 runs the full pipeline scoped to `agent_tags`.
            2. The top-1 score of pass 1 is compared against
               settings.agent_fallback_threshold.
            3. If it is below the threshold — meaning the agent's own slice of
               the corpus does not cover this question well — pass 2 re-runs
               the pipeline with the UNION of the requested tags and "general"
               (not "general" alone: the agent's own material must stay in
               play). Threshold and diversity cap are applied to pass 2 too,
               because both passes go through the same helper.
            4. Pass 2's result is returned. It can never be worse than pass 1's
               in terms of coverage, because its tag set is a strict superset
               and therefore matches a superset of chunks.

        Returns:
            List of ChunkResult objects, ordered by relevance (most relevant first).
            May return fewer than top_k results if not enough chunks exist, and
            may be empty when nothing clears settings.similarity_threshold.

        Raises:
            UpstreamError: If the embedding API or DB call fails. When agent
                           filtering is active, a failing RPC is reported
                           loudly (with the migration-009 hint) instead of
                           being degraded to an unfiltered search — returning
                           the whole corpus while the caller believes it is
                           reading one agent's slice would be a silent
                           correctness bug.
        """
        top_k = top_k or settings.retrieval_top_k
        # Fetch more candidates than top_k so the threshold filter has room to
        # discard low-quality chunks without starving the final result set.
        candidate_k = max(settings.retrieval_candidate_k, top_k)

        # Normalise framework_filter so downstream code can always treat
        # None and [] identically ("no restriction").
        normalized_frameworks = framework_filter or None

        # Same normalisation for the agent filter: None and [] both mean
        # "no agent scoping", which is the pre-existing behaviour.
        normalized_agent_tags = agent_tags or None

        logger.info(
            f"Retrieving top-{top_k} chunks (candidate_k={candidate_k}) for "
            f"tenant={tenant_id}, country={country}, industry={industry}, "
            f"frameworks={normalized_frameworks}, agent_tags={normalized_agent_tags}, "
            f"query='{query[:80]}...'"
        )

        # --- Step 1: Prepare query variants ---
        # Strip instruction verbs ("Resumeme el X" → "X") so the embedding
        # captures the topic, not the instruction pattern.
        search_topic = _extract_search_topic(query)
        if search_topic != query:
            logger.debug(f"Query topic extracted: '{query[:60]}' → '{search_topic}'")

        # Cross-lingual expansion: turn the topic into a bilingual pair so the
        # search is not biased toward the corpus' majority language.
        query_variants = generate_bilingual_queries(search_topic)
        if not query_variants:
            query_variants = [search_topic]

        logger.debug(
            f"Searching with {len(query_variants)} query variant(s): "
            f"{[v[:60] for v in query_variants]}"
        )

        # --- Step 2: Embed every variant ONCE ---
        # The embedding of a variant does not depend on the agent filter, so
        # computing it here (instead of inside the pass) means the fallback
        # pass costs zero extra OpenAI calls. Kept as (text, vector) pairs
        # because the hybrid RPC needs the raw text for its keyword branch.
        embedded_variants: list[tuple[str, list[float]]] = [
            (variant, embed_single(variant)) for variant in query_variants
        ]

        # --- Pass 1: agent-scoped (or unscoped when agent_tags is None) ---
        pass1_chunks, pass1_candidates = self._run_retrieval_pass(
            tenant_id=tenant_id,
            query=query,
            embedded_variants=embedded_variants,
            top_k=top_k,
            candidate_k=candidate_k,
            include_global=include_global,
            country=country,
            industry=industry,
            framework_filter=normalized_frameworks,
            use_hybrid=use_hybrid,
            agent_tags=normalized_agent_tags,
        )

        # Without an agent filter there is nothing to broaden — return the
        # legacy result immediately and emit no agent-specific log lines, so
        # existing deployments see byte-identical logs.
        if not normalized_agent_tags:
            return pass1_chunks

        # --- Step 3: Evaluate pass 1 ---
        # The list is ordered best-first, so element 0 is the top-1 result.
        # An empty list means nothing survived the similarity threshold, which
        # is the worst possible coverage: score it 0.0 so it always triggers
        # the fallback.
        top1_score = pass1_chunks[0].similarity if pass1_chunks else 0.0
        fallback_threshold = settings.agent_fallback_threshold

        logger.info(
            f"Agent-tag pass 1: tags={normalized_agent_tags}, "
            f"candidates={pass1_candidates}, returned={len(pass1_chunks)}, "
            f"top1_score={top1_score:.3f}"
        )

        # --- Step 4: Decide whether to broaden ---
        # Every condition is checked explicitly so the reason for skipping the
        # fallback is obvious when reading the code (and, via the log below,
        # when reading a production trace).
        already_includes_general = GENERAL_AGENT_TAG in normalized_agent_tags
        should_fallback = (
            enable_general_fallback
            # Retrying with a tag set we already used would repeat the exact
            # same query for the exact same result — pure waste.
            and not already_includes_general
            and top1_score < fallback_threshold
        )

        if not should_fallback:
            logger.info(
                f"Agent-tag fallback NOT triggered "
                f"(top1={top1_score:.3f}, threshold={fallback_threshold}, "
                f"enabled={enable_general_fallback}, "
                f"already_general={already_includes_general}, "
                f"tags={normalized_agent_tags})"
            )
            return pass1_chunks

        # --- Step 5: Pass 2 with the broadened tag set ---
        # Union, not replacement: the agent's own documents must remain
        # eligible, otherwise a weak-but-relevant agent chunk could be pushed
        # out by generic material.
        broadened_tags = list(normalized_agent_tags) + [GENERAL_AGENT_TAG]

        logger.info(
            f"Agent-tag fallback TRIGGERED: top1={top1_score:.3f} < "
            f"{fallback_threshold} → retrying with tags={broadened_tags}"
        )

        pass2_chunks, pass2_candidates = self._run_retrieval_pass(
            tenant_id=tenant_id,
            query=query,
            # Reusing the same embeddings: identical variants → identical
            # vectors, so a second embedding round-trip would be pure cost.
            embedded_variants=embedded_variants,
            top_k=top_k,
            candidate_k=candidate_k,
            include_global=include_global,
            country=country,
            industry=industry,
            framework_filter=normalized_frameworks,
            use_hybrid=use_hybrid,
            agent_tags=broadened_tags,
        )

        pass2_top1 = pass2_chunks[0].similarity if pass2_chunks else 0.0
        logger.info(
            f"Agent-tag pass 2: tags={broadened_tags}, "
            f"candidates={pass2_candidates}, returned={len(pass2_chunks)}, "
            f"top1_score={pass2_top1:.3f}"
        )

        # Pass 2's tag set is a strict superset of pass 1's, so its candidate
        # pool contains pass 1's entirely. There is no "keep the better of the
        # two" decision to make: pass 2 is the answer.
        return pass2_chunks

    def _run_retrieval_pass(
        self,
        *,
        tenant_id: str,
        query: str,
        embedded_variants: list[tuple[str, list[float]]],
        top_k: int,
        candidate_k: int,
        include_global: bool,
        country: str | None,
        industry: str | None,
        framework_filter: list[str] | None,
        use_hybrid: bool,
        agent_tags: list[str] | None,
    ) -> tuple[list[ChunkResult], int]:
        """
        Run one complete search → rank → filter cycle over pre-embedded queries.

        This is the body that `retrieve()` used to hold inline. It was extracted
        so the agent-tag fallback can execute it twice with different tag sets
        without duplicating a single line of ranking logic.

        Expected input:
            tenant_id:         Tenant whose chunks may be searched.
            query:             The ORIGINAL user query. Used only by the
                               reranker, which scores chunks against what the
                               user actually asked rather than against a
                               translated variant.
            embedded_variants: (variant_text, embedding) pairs produced once by
                               the caller. Never empty in practice — retrieve()
                               guarantees at least one entry.
            top_k:             Size of the returned list.
            candidate_k:       How many rows each individual RPC may return,
                               before merging/ranking. Larger than top_k on
                               purpose (see settings.retrieval_candidate_k).
            include_global:    Whether the shared 'global' library is searched
                               in addition to the tenant's own documents.
            country/framework_filter:
                               Pre-normalised filters, forwarded unchanged.
            industry:          Single industry code; wrapped into a one-element
                               array here and forwarded to SQL as
                               p_industry_codes.
            use_hybrid:        True → vector+keyword RPCs with a vector-only
                               fallback if they are unavailable.
            agent_tags:        Pre-normalised agent filter (None or non-empty).

        Logic:
            For each variant: run the tenant search and (optionally) the global
            search — every metadata filter (country, framework, agent, industry)
            is applied by the RPC itself, so excluded chunks never enter the
            pool and never compete for a top-k slot. Results merge into a pool
            keyed by chunk_id keeping the highest similarity seen, and the pool
            is then reranked, diversity-capped, cut at the similarity threshold
            and sliced to top_k.

        Expected output:
            (final_chunks, candidate_pool_size)

            final_chunks:        up to top_k ChunkResults, best first. Empty if
                                 nothing cleared settings.similarity_threshold.
            candidate_pool_size: number of DISTINCT chunks the searches produced
                                 before ranking. Returned so the caller can log
                                 how much the agent filter narrowed the corpus —
                                 comparing this number with and without
                                 agent_tags is the direct evidence that the SQL
                                 filter is doing something.
        """
        # --- Step 1: Translate the industry code into an SQL array filter ---
        # The RPCs take a TEXT[] because the underlying column is a TEXT[] and
        # the predicate is array overlap; the caller supplies a single code.
        #
        # HISTORICAL NOTE: this filter used to be applied in Python, AFTER
        # retrieval, over ChunkResult.industry_codes — a field the RPCs never
        # returned. It was therefore always None, the permissive rule
        # "no codes means applies to everyone" accepted every chunk, and the
        # filter silently did nothing. Migration 010 moved the predicate into
        # SQL, which is the only place it can run before candidates compete
        # for the top-k slots. See docs/rag/industry_codes.md.
        industry_codes = [industry] if industry else None

        # --- Step 2+3: Run both searches for every pre-embedded variant ---
        # Chunks are accumulated in a dict keyed by chunk_id so duplicates
        # across variants collapse, keeping the max similarity seen.
        candidate_pool: dict[str, ChunkResult] = {}

        for variant, variant_embedding in embedded_variants:
            if use_hybrid:
                # Hybrid path: vector + keyword in one RPC, fused via RRF.
                # Falls back gracefully if the new RPCs aren't deployed yet.
                try:
                    tenant_chunks = hybrid_search(
                        self.db,
                        tenant_id=tenant_id,
                        query_embedding=variant_embedding,
                        query_text=variant,
                        top_k=candidate_k,
                        framework_filter=framework_filter,
                        agent_tags=agent_tags,
                        industry_codes=industry_codes,
                    )
                except UpstreamError as e:
                    # The vector-only fallback below also carries agent_tags,
                    # so degrading from hybrid to vector never silently drops
                    # the agent filter. If the RPCs lack p_agent_tags entirely
                    # (migration 009 missing), that call raises in turn and the
                    # error surfaces to the caller — which is what we want.
                    logger.warning(
                        f"Hybrid search unavailable; falling back to vector-only: {e}"
                    )
                    tenant_chunks = self._search_tenant_chunks(
                        tenant_id=tenant_id,
                        query_embedding=variant_embedding,
                        top_k=candidate_k,
                        framework_filter=framework_filter,
                        agent_tags=agent_tags,
                        industry_codes=industry_codes,
                    )
                global_chunks: list[ChunkResult] = []
                if include_global and tenant_id != "global":
                    try:
                        global_chunks = hybrid_search_global(
                            self.db,
                            query_embedding=variant_embedding,
                            query_text=variant,
                            top_k=candidate_k,
                            country=country,
                            framework_filter=framework_filter,
                            agent_tags=agent_tags,
                            industry_codes=industry_codes,
                        )
                    except UpstreamError as e:
                        # hybrid_search_global only raises when an agent filter
                        # is active (otherwise it fails soft and returns []).
                        # Mirror the tenant branch: degrade to the vector-only
                        # global RPC, still carrying agent_tags. This keeps a
                        # missing migration 006 (no *_hybrid functions) from
                        # taking down agent queries, while a missing migration
                        # 009 still surfaces — the vector call raises in turn.
                        logger.warning(
                            f"Hybrid global search unavailable; falling back "
                            f"to vector-only: {e}"
                        )
                        global_chunks = self._search_global_chunks(
                            query_embedding=variant_embedding,
                            top_k=candidate_k,
                            country=country,
                            framework_filter=framework_filter,
                            agent_tags=agent_tags,
                            industry_codes=industry_codes,
                        )
            else:
                tenant_chunks = self._search_tenant_chunks(
                    tenant_id=tenant_id,
                    query_embedding=variant_embedding,
                    top_k=candidate_k,
                    framework_filter=framework_filter,
                    agent_tags=agent_tags,
                    industry_codes=industry_codes,
                )

                global_chunks = []
                if include_global and tenant_id != "global":
                    global_chunks = self._search_global_chunks(
                        query_embedding=variant_embedding,
                        top_k=candidate_k,
                        country=country,
                        framework_filter=framework_filter,
                        agent_tags=agent_tags,
                        industry_codes=industry_codes,
                    )

            logger.debug(
                f"Variant '{variant[:40]}…' → tenant={len(tenant_chunks)} "
                f"global={len(global_chunks)} (hybrid={use_hybrid}, "
                f"agent_tags={agent_tags})"
            )

            for chunk in tenant_chunks + global_chunks:
                existing = candidate_pool.get(chunk.chunk_id)
                if existing is None or chunk.similarity > existing.similarity:
                    candidate_pool[chunk.chunk_id] = chunk

        unique_chunks = list(candidate_pool.values())
        logger.debug(
            f"Merged candidate pool: {len(unique_chunks)} unique chunks across "
            f"{len(embedded_variants)} variant(s)"
        )

        # --- Step 4: Verify the SQL industry filter actually applied ---
        # This does NOT filter. The filtering happened in SQL; this only
        # detects the situation that caused the original bug — an industry
        # filter that was requested but silently had no effect — and says so
        # in the log instead of letting wrong-sector content through unnoticed.
        self._warn_on_industry_mismatch(unique_chunks, industry)

        # --- Step 5: Re-rank ---
        ranked_chunks = rerank_chunks(unique_chunks, query)

        # --- Step 5b: Per-document diversity cap ---
        # Keep all chunks but reorder so each document contributes at most
        # max_per_document chunks before any document contributes a second
        # batch. Prevents a single high-similarity document (e.g. one that
        # shares the query language) from monopolising every top-k slot.
        #
        # The cap depends on whether this pass is agent-scoped. Searching the
        # whole heterogeneous corpus needs the strict cap of 1; once agent_tags
        # has narrowed the pool to one domain, that same cap starves the answer
        # of depth from the single most relevant document (see
        # settings.agent_max_chunks_per_document for the full rationale).
        # agent_tags is already normalised to None-or-non-empty by retrieve(),
        # so a plain truth test is enough to tell the two regimes apart.
        if agent_tags:
            max_per_doc = settings.agent_max_chunks_per_document
            # INFO, not DEBUG: this is the non-default regime, and being able
            # to see which cap was in force is what makes an agent's grounding
            # score interpretable after the fact. Kept inside the branch so
            # callers that don't use agent_tags see byte-identical logs.
            logger.info(
                f"Diversity cap: max {max_per_doc} chunk(s) per document "
                f"(agent-scoped regime, tags={agent_tags})"
            )
        else:
            max_per_doc = settings.retrieval_max_chunks_per_document
            logger.debug(
                f"Diversity cap: max {max_per_doc} chunk(s) per document "
                f"(broad-corpus regime)"
            )
        capped_chunks = apply_per_document_cap(ranked_chunks, max_per_doc)

        # --- Step 6: Filter by similarity threshold ---
        threshold = settings.similarity_threshold
        above_threshold = [c for c in capped_chunks if c.similarity >= threshold]
        discarded = len(capped_chunks) - len(above_threshold)
        if discarded:
            logger.info(
                f"Threshold filter ({threshold}): discarded {discarded} low-similarity chunks, "
                f"{len(above_threshold)} remaining"
            )

        if not above_threshold:
            logger.info(
                f"No chunks above similarity threshold {threshold}. Returning empty list."
            )
            return [], len(unique_chunks)

        final_chunks = above_threshold[:top_k]

        logger.info(
            f"Retrieval complete: returning {len(final_chunks)} chunks "
            f"(variants: {len(embedded_variants)}, unique_candidates: {len(unique_chunks)}, "
            f"threshold_discarded: {discarded})"
        )

        return final_chunks, len(unique_chunks)

    def _search_tenant_chunks(
        self,
        tenant_id: str,
        query_embedding: list[float],
        top_k: int,
        framework_filter: list[str] | None = None,
        agent_tags: list[str] | None = None,
        industry_codes: list[str] | None = None,
    ) -> list[ChunkResult]:
        """
        Call the match_chunks Postgres RPC function for a specific tenant.

        Args:
            tenant_id:        Filter chunks to this tenant only.
            query_embedding:  The vectorized query.
            top_k:            Number of results to return.
            framework_filter: Optional list of framework slugs. Passed to the
                              RPC as match_frameworks TEXT[]. None / [] means
                              no restriction.
            agent_tags:       Optional business-agent slugs. Passed to the RPC
                              as p_agent_tags TEXT[] ONLY when non-empty; see
                              the comment on the payload below for why the key
                              is omitted rather than sent as NULL.
            industry_codes:   Optional institution industry codes, passed as
                              p_industry_codes TEXT[] under the same
                              omit-when-empty rule.

        Returns:
            List of ChunkResult objects.

        Raises:
            UpstreamError: on any RPC failure. The message names migration 009
                           and/or 010 when the corresponding filter was
                           requested, since a missing parameter is the
                           likeliest cause.
        """
        # The RPC is defined in migration 004 (match_chunks with
        # match_frameworks TEXT[] DEFAULT NULL), extended by migration 009
        # with p_agent_tags TEXT[] DEFAULT NULL and by migration 010 with
        # p_industry_codes TEXT[] DEFAULT NULL. We always pass the framework
        # key so the function signature binding is unambiguous — Supabase's
        # PostgREST selects the function by argument names.
        rpc_args: dict = {
            "query_embedding": query_embedding,
            "match_tenant_id": tenant_id,
            "match_count": top_k,
            "match_frameworks": framework_filter,
        }
        normalized_agent_tags = agent_tags or None
        normalized_industry_codes = industry_codes or None
        # Omit a key entirely when its filter is empty. Sending it as NULL
        # would be semantically equivalent in SQL, but the omission also keeps
        # the payload valid against a database where migration 009/010 has not
        # been applied yet — which is what protects the existing callers that
        # use neither filter (risk/service.py, policies/service.py).
        if normalized_agent_tags:
            rpc_args["p_agent_tags"] = normalized_agent_tags
        if normalized_industry_codes:
            rpc_args["p_industry_codes"] = normalized_industry_codes

        try:
            result = self.db.rpc("match_chunks", rpc_args).execute()

        except Exception as e:
            message = f"Database vector search failed: {e}"
            hints = migration_hints(
                normalized_agent_tags, normalized_industry_codes
            )
            if hints:
                message = f"{message}. {hints}"
            raise UpstreamError(
                message,
                detail={
                    "tenant_id": tenant_id,
                    "agent_tags": normalized_agent_tags,
                    "industry_codes": normalized_industry_codes,
                },
            )

        # Convert the raw DB rows into ChunkResult objects.
        # section_ref, industry_codes and is_reference_list are read with
        # .get(...) because rows produced by older RPCs (pre-migration
        # 005/008/011) won't carry the columns. The `or False` keeps the
        # field a bool when the RPC predates migration 011, so a stale
        # deployment simply applies no penalty instead of erroring.
        return [
            ChunkResult(
                chunk_id=str(row["id"]),
                document_id=str(row["document_id"]),
                tenant_id=row["tenant_id"],
                content=row["content"],
                chunk_index=row["chunk_index"],
                similarity=float(row["similarity"]),
                section_ref=row.get("section_ref"),
                industry_codes=row.get("industry_codes"),
                is_reference_list=bool(row.get("is_reference_list") or False),
            )
            for row in (result.data or [])
        ]

    def _search_global_chunks(
        self,
        query_embedding: list[float],
        top_k: int,
        country: str | None = None,
        framework_filter: list[str] | None = None,
        agent_tags: list[str] | None = None,
        industry_codes: list[str] | None = None,
    ) -> list[ChunkResult]:
        """
        Call the match_global_chunks Postgres RPC for the shared library.

        Args:
            query_embedding:  The vectorized query.
            top_k:            Number of global chunks to return.
            country:          Country filter (NULL → only global docs).
            framework_filter: Optional framework slug list (NULL → no restriction).
            agent_tags:       Optional business-agent slugs, sent as
                              p_agent_tags only when non-empty.
            industry_codes:   Optional institution industry codes, sent as
                              p_industry_codes only when non-empty.

        Returns:
            List of ChunkResult objects from the global library, or an empty
            list when the RPC fails and no optional metadata filter was
            requested.

        Error policy:
            A global-library failure is normally NON-FATAL: it is supplementary
            to the tenant's own corpus, so we log and continue.

            That is deliberately reversed when agent_tags or industry_codes is
            set, because both the regulatory corpus the agents rely on (ISO,
            NIST) and the Layer-3 sector documents (SUGEF, CONASSIF) live in
            the global library. Swallowing the error there would return zero
            filtered chunks and be indistinguishable from "there is no such
            content" — the precise silent failure these filters exist to
            eliminate. So we raise instead.
        """
        rpc_args: dict = {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "match_country": country,
            "match_frameworks": framework_filter,
        }
        # Same conditional-key rule as _search_tenant_chunks.
        normalized_agent_tags = agent_tags or None
        normalized_industry_codes = industry_codes or None
        if normalized_agent_tags:
            rpc_args["p_agent_tags"] = normalized_agent_tags
        if normalized_industry_codes:
            rpc_args["p_industry_codes"] = normalized_industry_codes

        try:
            result = self.db.rpc("match_global_chunks", rpc_args).execute()

        except Exception as e:
            hints = migration_hints(
                normalized_agent_tags, normalized_industry_codes
            )
            if hints:
                raise UpstreamError(
                    f"Global library search failed: {e}. {hints}",
                    detail={
                        "rpc": "match_global_chunks",
                        "agent_tags": normalized_agent_tags,
                        "industry_codes": normalized_industry_codes,
                    },
                )
            # Global library failure is non-fatal — log and return empty list
            logger.warning(f"Global library search failed (continuing without it): {e}")
            return []

        return [
            ChunkResult(
                chunk_id=str(row["id"]),
                document_id=str(row["document_id"]),
                tenant_id=row["tenant_id"],
                content=row["content"],
                chunk_index=row["chunk_index"],
                similarity=float(row["similarity"]),
                section_ref=row.get("section_ref"),
                industry_codes=row.get("industry_codes"),
                is_reference_list=bool(row.get("is_reference_list") or False),
            )
            for row in (result.data or [])
        ]

    @staticmethod
    def chunk_matches_industry(
        chunk_industry_codes: list[str] | None,
        industry: str | None,
    ) -> bool:
        """
        The industry visibility rule, as a single reusable predicate.

        This is the Python mirror of the SQL condition in migration
        010_industry_filter_fix.sql. It is NOT the filter — SQL is the single
        source of truth and does the actual filtering — but keeping the rule
        expressed once here lets `_warn_on_industry_mismatch` detect a filter
        that failed to apply, and lets tests pin the semantics down without a
        database.

        Expected input:
            chunk_industry_codes: the chunk's industry_codes column, e.g.
                                  ["ALL"], ["bank", "cooperative"], [] or None.
            industry:             the institution's industry code, e.g. "bank".
                                  None means "no filtering requested".

        Logic (permissive by design — cross-sector material must never be
        excluded, only sector-restricted material may be):
            industry is None                       → visible
            chunk codes are None or empty          → visible (safety net for
                                                     rows predating migration
                                                     008)
            chunk codes contain 'ALL'              → visible (Layer 1-2: ISO,
                                                     NIST, OWASP …)
            chunk codes contain the institution's
            industry                               → visible (Layer 3 match)
            otherwise                              → NOT visible

        Expected output:
            True when the chunk may be shown to this institution.
        """
        if not industry:
            return True
        codes = chunk_industry_codes
        if not codes:
            return True
        return "ALL" in codes or industry in codes

    def _warn_on_industry_mismatch(
        self,
        chunks: list[ChunkResult],
        industry: str | None,
    ) -> None:
        """
        Sanity check: assert (in the log) that the SQL industry filter worked.

        WHY THIS EXISTS RATHER THAN A SECOND FILTER:
        The original bug was a Python post-filter silently accepting
        everything because the data it needed never arrived. Replacing it with
        another Python filter would just re-create a second source of truth
        and hide the same class of failure again. So this method deliberately
        DROPS NOTHING — it only reports. If it ever fires, the SQL predicate
        is not doing its job and that is what needs fixing.

        Expected input:
            chunks:   the candidate pool returned by the RPCs.
            industry: the industry filter that was requested, or None.

        Logic:
            With no industry requested there is nothing to verify. Otherwise
            every returned chunk should already satisfy
            `chunk_matches_industry`. Chunks whose industry_codes came back
            as None are skipped rather than reported: that means the database
            predates migration 010 and simply is not returning the column, a
            different (and separately logged) situation from the filter
            returning wrong rows.

        Expected output:
            None. Emits a single WARNING listing the offending chunk ids when
            the invariant is violated, and nothing at all when it holds.
        """
        if not industry:
            return

        offenders = [
            chunk
            for chunk in chunks
            # `is not None` on purpose: an absent column (pre-010 database) is
            # not evidence of a filter failure, an actually-wrong value is.
            if chunk.industry_codes is not None
            and not self.chunk_matches_industry(chunk.industry_codes, industry)
        ]
        if offenders:
            logger.warning(
                f"Industry filter mismatch: {len(offenders)} chunk(s) not "
                f"applicable to industry='{industry}' were returned by the "
                f"database. The SQL filter (migration 010) may not be applied. "
                f"Offending chunks: "
                f"{[(c.chunk_id, c.industry_codes) for c in offenders[:5]]}"
            )
