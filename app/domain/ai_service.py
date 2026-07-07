# ============================================================
# app/domain/ai_service.py — Central AI query orchestrator
#
# WHY THIS FILE EXISTS:
# The AIService is the most important class in the system. It is
# the only place where prompts, retrieval, guardrails, the LLM
# router, tracing, evaluation, threads (L1) and projects (L2)
# are wired together.
#
# PIPELINE (in order):
#
#   0. Resolve or create the chat thread  (L1)
#   1. Load the project expediente if any (L2)
#   2. Load the system prompt for the use case
#   3. Retrieve relevant chunks (RAG) with optional framework filter
#   4. Detect "why…?" queries and load reasoning context from traces
#   5. Apply guardrails
#   6. Assemble the final prompt: system + memory block + context +
#      recent turns + user query
#   7. Call the LLM
#   8. Persist: trace + user message + assistant message
#   9. Run automatic evaluation
#  10. Return the response + thread_id + project_id to the caller
#
# DESIGN PRINCIPLE:
# This file is the only class that imports from multiple domains.
# Every other service is isolated and unaware of siblings.
# ============================================================

import json
from dataclasses import dataclass
from typing import Iterator, Optional

from supabase import Client

from app.core.config import settings
from app.core.logging import get_logger

# Domain services
from app.domain.prompts.service import PromptService
from app.domain.retrieval.service import RetrievalService
from app.domain.llm.guardrails import apply_guardrails, validate_response_guardrails
from app.domain.llm.router import LLMRouter
from app.domain.llm.models import LLMRequest
from app.domain.traces.service import TraceService
from app.domain.traces.models import CreateTraceRequest, Trace
from app.domain.evals.service import EvalService
from app.domain.evals.evaluator import evaluate_response
from app.domain.observability.cost import estimate_cost
from app.domain.projects.service import ProjectService
from app.domain.projects.models import Project
from app.domain.response_risk.scorer import compute_risk
from app.domain.response_risk.service import ResponseRiskService
from app.domain.threads.service import ThreadService
from app.domain.threads.models import AppendMessageRequest, Message, Thread

# Schemas
from app.schemas.ai import AIQueryRequest, AIQueryResponse, RetrievedChunkInfo

# Context management
from app.domain.context.token_counter import TokenCounter

_token_counter = TokenCounter()

logger = get_logger(__name__)


# ------------------------------------------------------------
# PreparedPipeline
#
# Captures the state built by every step that runs BEFORE the
# LLM call. Both query() (blocking) and query_stream() reuse
# this so prompt assembly, retrieval, memory hydration and
# guardrails are guaranteed to behave identically across the
# two execution modes.
# ------------------------------------------------------------
@dataclass
class _PreparedPipeline:
    final_prompt: str
    safe_query: str
    retrieved_chunks: list
    project: Project | None
    thread: Thread | None
    used_reasoning_memory: bool
    has_project_context: bool = False  # True when project data was injected into prompt


# ------------------------------------------------------------
# "Why did you X?" detection
#
# The plan (section 5) calls this lightweight L3: when the user is
# evidently asking for a justification of a previous decision, we
# load the last N traces from the same project and feed them to the
# LLM so it can reconstruct its reasoning narratively.
#
# Keeping it as a heuristic (regex-like keyword match) is enough for
# v1 — we do not want to pay an extra LLM call for intent detection.
# False positives only cost a bit of extra prompt context.
# ------------------------------------------------------------
_WHY_KEYWORDS = (
    "por qué",
    "porque ",
    "justifica",
    "justificación",
    "razón",
    "razones",
    "fundamenta",
    "fundamento",
    "cómo llegaste",
    "como llegaste",
    "explica por qué",
    "por que",
)


def _looks_like_why_query(query: str) -> bool:
    lowered = query.strip().lower()
    return any(keyword in lowered for keyword in _WHY_KEYWORDS)


# ------------------------------------------------------------
# Use cases that expect a conversational experience. When one of
# these is used without an explicit thread_id we auto-create a
# thread so future turns continue the same conversation.
# ------------------------------------------------------------
_CONVERSATIONAL_USE_CASES = {
    "chatbot",
    "risk_intake",
    "policy_generation",
    # Variantes UI dedicadas del chatbot — mismo flujo conversacional.
    "field_draft",
    "step_guide",
    "deliverable_review",
}

# Use cases that bypass the full RAG pipeline and use dedicated structured prompts.
# These always run with stream=False and return a JSON string as response_text.
_STRUCTURED_USE_CASES = {"control_ranking", "iso_checklist"}

# UI use cases whose task is GENERATIVE (drafting/review), not Q&A: the
# retrieved chunks are supporting material, so "la respuesta no está en
# los chunks" es esperado y no debe reescribir el borrador como
# abstención (should_have_said_idk).
_GENERATIVE_UI_USE_CASES = {"field_draft", "step_guide", "deliverable_review"}


class AIService:
    """Orchestrates the complete AI query pipeline."""

    def __init__(self, db: Client) -> None:
        self.db = db

        self.prompt_service = PromptService(db)
        self.retrieval_service = RetrievalService(db)
        self.trace_service = TraceService(db)
        self.eval_service = EvalService(db)
        self.project_service = ProjectService(db)
        self.thread_service = ThreadService(db)
        self.response_risk_service = ResponseRiskService(db)

        # The LLM router is stateless — no DB needed
        self.llm_router = LLMRouter()

    # ================================================================
    # Public entrypoints
    # ================================================================
    def query(
        self,
        request: AIQueryRequest,
        tenant_id: str,
        user_id: str,
    ) -> AIQueryResponse:
        logger.info(
            f"AI query: tenant={tenant_id}, user={user_id}, "
            f"use_case={request.use_case}, project_id={request.project_id}, "
            f"thread_id={request.thread_id}, query='{request.query[:80]}...'"
        )

        if request.use_case in _STRUCTURED_USE_CASES:
            return self._query_structured(request, tenant_id, user_id)

        prepared = self._prepare_pipeline(request, tenant_id, user_id)

        # --------------------------------------------------------
        # Step 8 — Call the LLM
        # --------------------------------------------------------
        llm_response = self.llm_router.route(
            LLMRequest(prompt=prepared.final_prompt, use_case=request.use_case)
        )
        logger.info(
            f"LLM response: {len(llm_response.text)} chars, "
            f"{llm_response.latency_ms}ms via {llm_response.model_provider}"
        )

        # --------------------------------------------------------
        # Step 9 — Governance pipeline (eval → guardrails → risk)
        # Runs BEFORE persistence so the trace carries risk_level.
        # Best-effort: failures degrade gracefully but never raise.
        # --------------------------------------------------------
        governance = self._run_governance_pipeline(
            request=request,
            tenant_id=tenant_id,
            user_id=user_id,
            prepared=prepared,
            response_text=llm_response.text,
        )

        # --------------------------------------------------------
        # Step 10 — Persist trace + thread messages + evaluation + risk
        # --------------------------------------------------------
        trace_id = self._persist_trace_and_eval(
            request=request,
            tenant_id=tenant_id,
            user_id=user_id,
            prepared=prepared,
            response_text=governance["safe_response"],
            model_provider=llm_response.model_provider,
            model_name=llm_response.model_name,
            latency_ms=llm_response.latency_ms,
            governance=governance,
            prompt_tokens=getattr(llm_response, "prompt_tokens", None),
            completion_tokens=getattr(llm_response, "completion_tokens", None),
        )

        # --------------------------------------------------------
        # Step 11 — Build response (returns the SAFE response so
        # PII redactions / disclaimers are visible to the caller)
        # --------------------------------------------------------
        chunk_infos = [
            RetrievedChunkInfo(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                similarity=round(chunk.similarity, 4),
                content_preview=chunk.content[:200],
            )
            for chunk in prepared.retrieved_chunks
        ]

        # Cost is computed once here from the same usage numbers we
        # persist on the trace — keeping the response and trace in
        # sync so external consumers (Edge Function quota tracking,
        # billing exports) can rely on either source.
        cost_usd = estimate_cost(
            provider=llm_response.model_provider,
            model=llm_response.model_name,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
        )
        total_tokens = (
            (llm_response.prompt_tokens or 0) + (llm_response.completion_tokens or 0)
            if (llm_response.prompt_tokens is not None
                or llm_response.completion_tokens is not None)
            else None
        )

        return AIQueryResponse(
            response_text=governance["safe_response"],
            trace_id=trace_id,
            thread_id=str(prepared.thread.id) if prepared.thread else None,
            project_id=str(prepared.project.id) if prepared.project else None,
            model_provider=llm_response.model_provider,
            model_name=llm_response.model_name,
            latency_ms=llm_response.latency_ms,
            use_case=request.use_case,
            retrieved_chunk_count=len(prepared.retrieved_chunks),
            retrieved_chunks=chunk_infos,
            used_reasoning_memory=prepared.used_reasoning_memory,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=total_tokens,
            cost_estimate_usd=cost_usd,
        )

    # ----------------------------------------------------------
    # query_stream — Streaming variant of query()
    #
    # Runs the same pipeline as query() up to (and including)
    # prompt assembly, then opens a streaming LLM call. Tokens
    # are yielded as Server-Sent Events. When the upstream stream
    # closes, the orchestrator buffers the full text, persists
    # the trace + thread messages + evaluation, and emits a
    # final "done" event with the trace_id so the client can
    # follow up via /traces/{id} or /feedback.
    # ----------------------------------------------------------
    def query_stream(
        self,
        request: AIQueryRequest,
        tenant_id: str,
        user_id: str,
    ) -> Iterator[bytes]:
        logger.info(
            f"AI query (stream): tenant={tenant_id}, user={user_id}, "
            f"use_case={request.use_case}, project_id={request.project_id}, "
            f"thread_id={request.thread_id}, query='{request.query[:80]}...'"
        )

        prepared = self._prepare_pipeline(request, tenant_id, user_id)

        accumulated_text = ""
        latency_ms = 0
        model_provider: str = ""
        model_name: str = ""
        prompt_tokens: int | None = None
        completion_tokens: int | None = None
        completed = False

        try:
            for event in self.llm_router.stream_route(
                LLMRequest(prompt=prepared.final_prompt, use_case=request.use_case)
            ):
                if event.type == "token":
                    accumulated_text += event.text
                    yield _sse_event("token", {"text": event.text})
                elif event.type == "done":
                    latency_ms = event.latency_ms or 0
                    model_provider = event.model_provider or ""
                    model_name = event.model_name or ""
                    # Provider may omit token counts on stream done
                    # (Gemini occasionally does). When absent we forward
                    # None and downstream consumers fall back to their
                    # own estimation.
                    prompt_tokens = event.prompt_tokens
                    completion_tokens = event.completion_tokens
                    completed = True
                    # The done event from the LLM is consumed here;
                    # the SSE done emitted to the client is built
                    # below after persistence.
                    break
        except Exception as e:
            # Surface the upstream failure as a final error event
            # so the client knows the stream did not complete.
            logger.error(f"Streaming LLM call failed: {e}", exc_info=True)
            yield _sse_event(
                "error",
                {"message": "Upstream LLM error", "detail": str(e)},
            )
            # Best-effort: still try to persist the partial text so
            # debugging is possible.

        logger.info(
            f"Stream finished: {len(accumulated_text)} chars, "
            f"{latency_ms}ms via {model_provider}, completed={completed}"
        )

        # --------------------------------------------------------
        # Persist trace + thread messages + evaluation + risk now
        # that the full text is buffered. Mirrors query()'s steps
        # 9-11 but uses the streamed response.
        # --------------------------------------------------------
        governance = self._run_governance_pipeline(
            request=request,
            tenant_id=tenant_id,
            user_id=user_id,
            prepared=prepared,
            response_text=accumulated_text,
        )

        trace_id = self._persist_trace_and_eval(
            request=request,
            tenant_id=tenant_id,
            user_id=user_id,
            prepared=prepared,
            response_text=governance["safe_response"],
            model_provider=model_provider,
            model_name=model_name,
            latency_ms=latency_ms,
            governance=governance,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        chunk_infos_payload = [
            {
                "chunk_id": c.chunk_id,
                "document_id": c.document_id,
                "similarity": round(c.similarity, 4),
                "section_ref": getattr(c, "section_ref", None),
            }
            for c in prepared.retrieved_chunks
        ]

        # Compute the same cost number that was written to the trace
        # so SSE consumers (GovAI agent-chat Edge Function) can update
        # agent_interactions.{prompt,completion,total}_tokens +
        # cost_estimate_usd without a second DB lookup.
        cost_usd = estimate_cost(
            provider=model_provider,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )
        total_tokens = (
            (prompt_tokens or 0) + (completion_tokens or 0)
            if (prompt_tokens is not None or completion_tokens is not None)
            else None
        )

        yield _sse_event(
            "done",
            {
                "trace_id": trace_id,
                "thread_id": str(prepared.thread.id) if prepared.thread else None,
                "project_id": str(prepared.project.id) if prepared.project else None,
                "model_provider": model_provider,
                "model_name": model_name,
                "latency_ms": latency_ms,
                "use_case": request.use_case,
                "retrieved_chunk_count": len(prepared.retrieved_chunks),
                "retrieved_chunks": chunk_infos_payload,
                "used_reasoning_memory": prepared.used_reasoning_memory,
                "completed": completed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost_estimate_usd": cost_usd,
            },
        )

    # ================================================================
    # Pipeline preparation (shared by query and query_stream)
    # ================================================================
    def _prepare_pipeline(
        self,
        request: AIQueryRequest,
        tenant_id: str,
        user_id: str,
    ) -> _PreparedPipeline:
        """
        Run every step that precedes the LLM call:
        prompts, guardrails, thread/project resolution, retrieval,
        why-query detection, conversation hydration, source
        metadata, and prompt assembly. Returns a _PreparedPipeline
        snapshot that both query() and query_stream() consume.
        """
        # --------------------------------------------------------
        # Step -1 — Load system prompt and apply guardrails before
        # spending any persistence effort. The injection check
        # raises ValidationError (HTTP 422) which must NOT leave a
        # new thread behind.
        #
        # When the caller supplies system_prompt_override (e.g. a
        # GovAI admin-edited prompt_registry entry forwarded by the
        # Edge Function proxy) we honor it directly and skip the
        # use_case prompt lookup. The use_case is still recorded on
        # the trace for observability so we can compare results.
        # --------------------------------------------------------
        override = (request.system_prompt_override or "").strip()
        if override:
            base_system_prompt = override
            logger.debug(
                f"System prompt: caller override ({len(override)} chars) "
                f"for use_case={request.use_case}"
            )
        else:
            prompt_record = self.prompt_service.get_active_prompt(request.use_case)
            base_system_prompt = prompt_record.content
            logger.debug(
                f"System prompt: '{prompt_record.name}' v{prompt_record.version}"
            )

        safe_query, hardened_system_prompt = apply_guardrails(
            user_query=request.query,
            system_prompt=base_system_prompt,
        )

        # --------------------------------------------------------
        # Step 0 — Resolve / create the thread (L1)
        # --------------------------------------------------------
        thread: Optional[Thread] = None
        if request.thread_id or request.use_case in _CONVERSATIONAL_USE_CASES:
            thread = self.thread_service.get_or_create(
                tenant_id=tenant_id,
                user_id=user_id,
                thread_id=str(request.thread_id) if request.thread_id else None,
                use_case=request.use_case,
                project_id=str(request.project_id) if request.project_id else None,
                first_user_message=request.query,
            )

        # --------------------------------------------------------
        # Step 1 — Load project expediente (L2) if present
        # --------------------------------------------------------
        project = self.project_service.get_optional(
            project_id=str(request.project_id) if request.project_id else None,
            tenant_id=tenant_id,
        )

        # --------------------------------------------------------
        # Step 3 — Retrieval (RAG)
        # --------------------------------------------------------
        retrieved_chunks = self.retrieval_service.retrieve(
            tenant_id=tenant_id,
            query=request.query,
            top_k=request.top_k,
            include_global=request.include_global,
            country=request.country,
            industry=request.industry,
            framework_filter=request.framework_filter,
        )
        logger.debug(f"Retrieved {len(retrieved_chunks)} context chunks")

        # --------------------------------------------------------
        # Step 4 — "Why?" detection + reasoning memory lookup
        # --------------------------------------------------------
        used_reasoning_memory = False
        reasoning_traces: list[Trace] = []
        if project is not None and _looks_like_why_query(request.query):
            try:
                reasoning_traces = self.trace_service.list_project_traces(
                    project_id=str(project.id),
                    tenant_id=tenant_id,
                    use_cases=[
                        "risk_analysis",
                        "risk_intake",
                        "controls_generation",
                        "policy_generation",
                    ],
                    limit=5,
                )
                used_reasoning_memory = len(reasoning_traces) > 0
                logger.info(
                    f"Why-query detected: loaded {len(reasoning_traces)} "
                    f"reasoning traces for project {project.id}"
                )
            except Exception as e:
                logger.warning(f"Failed to load reasoning traces: {e}")

        # --------------------------------------------------------
        # Step 5 — Load recent messages (L1)
        # --------------------------------------------------------
        thread_messages: list[Message] = []
        if thread is not None:
            try:
                thread_messages = self.thread_service.get_recent_messages(
                    thread_id=str(thread.id)
                )
            except Exception as e:
                logger.warning(f"Failed to load thread messages: {e}")

        # --------------------------------------------------------
        # Step 6.5 — Hydrate chunk source metadata
        # --------------------------------------------------------
        doc_metadata = self._load_doc_metadata(retrieved_chunks)

        # --------------------------------------------------------
        # Step 7 — Assemble the final prompt
        #
        # inline_project_context is used only when no DB-persisted
        # project was loaded (sandbox Phase 2 of the GovAI sidecar
        # integration where ai_projects rows aren't mirrored yet).
        # When project is set, the DB expediente wins.
        # --------------------------------------------------------
        inline_project_context = (
            request.project_context if project is None else None
        )
        # Use inline chat_history from the request when no DB thread is active.
        # This restores turn-by-turn memory for callers (Edge Function) that
        # don't persist threads in the Python DB.
        inline_chat_history = (
            request.chat_history if thread is None and request.chat_history else None
        )
        final_prompt = self._assemble_prompt(
            system_prompt=hardened_system_prompt,
            project=project,
            inline_project_context=inline_project_context,
            inline_chat_history=inline_chat_history,
            thread_messages=thread_messages,
            reasoning_traces=reasoning_traces,
            chunks=retrieved_chunks,
            doc_metadata=doc_metadata,
            user_query=safe_query,
            response_mode=request.response_mode,
        )
        logger.debug(f"Final prompt assembled: {len(final_prompt)} chars")

        return _PreparedPipeline(
            final_prompt=final_prompt,
            safe_query=safe_query,
            retrieved_chunks=retrieved_chunks,
            project=project,
            thread=thread,
            used_reasoning_memory=used_reasoning_memory,
            has_project_context=(project is not None or inline_project_context is not None),
        )

    # ================================================================
    # Governance pipeline (shared by query and query_stream)
    # ================================================================
    def _run_governance_pipeline(
        self,
        request: AIQueryRequest,
        tenant_id: str,
        user_id: str,
        prepared: _PreparedPipeline,
        response_text: str,
    ) -> dict:
        """
        Run evaluation → response guardrails → risk scoring on a
        completed LLM response. Returns a dict with the four
        artefacts the persistence step needs:

          - "evaluation":  output of evaluate_response()
          - "guardrails":  output of validate_response_guardrails()
          - "risk":        output of compute_risk()
          - "safe_response": the response after PII redaction /
                             abstention rewriting / disclaimer.

        ALL exceptions are caught — degraded scoring is preferred
        over breaking the user-facing reply.
        """
        evaluation: dict = {}
        guardrails_result: dict = {
            "allowed": True,
            "violations": [],
            "safe_response": response_text,
            "requires_disclaimer": False,
            "pii_findings": [],
        }
        risk: dict = {}

        # --- 1. Evaluation (8 metrics + LLM judge) ---
        # The LLM judge only makes sense when we have chunks to verify
        # against. With no chunks (small-talk, system queries) we skip
        # the judge so it does not flag "should_have_said_idk" on every
        # conversational reply.
        run_judge = settings.eval_use_llm_judge and bool(prepared.retrieved_chunks)
        try:
            evaluation = evaluate_response(
                query=request.query,
                chunks=prepared.retrieved_chunks,
                response=response_text,
                citations=None,
                instructions=None,
                use_llm_judge=run_judge,
                response_mode=request.response_mode,
            )
        except Exception as e:
            logger.warning(f"Inline evaluator failed (non-fatal): {e}")

        # --- 2. Response guardrails (PII, hallucination, disclaimer) ---
        # When project context was injected into the prompt the model may
        # legitimately answer using project-level data that is NOT in the
        # retrieved document chunks. Clear should_have_said_idk so the
        # guardrail does not overwrite a correct project-grounded answer.
        if (
            prepared.has_project_context
            or request.use_case in _GENERATIVE_UI_USE_CASES
        ) and evaluation.get("should_have_said_idk"):
            evaluation = {**evaluation, "should_have_said_idk": False}
        try:
            guardrails_result = validate_response_guardrails(
                query=request.query,
                response=response_text,
                chunks=prepared.retrieved_chunks,
                client_id=tenant_id,
                user_id=user_id,
                evaluation=evaluation or None,
                use_case=request.use_case,
            )
        except Exception as e:
            logger.warning(f"Response guardrails failed (non-fatal): {e}")

        # --- 3. Risk scoring ---
        try:
            risk = compute_risk(
                evaluation=evaluation,
                guardrails=guardrails_result,
                metadata={
                    "use_case": request.use_case,
                    "chunks": prepared.retrieved_chunks,
                    "client_id": tenant_id,
                },
            )
        except Exception as e:
            logger.warning(f"Risk scoring failed (non-fatal): {e}")
            risk = {
                "risk_level": "medium",
                "confidence_score": 0.5,
                "risk_reasons": [f"risk_compute_error: {e}"],
                "retrieval_confidence": 0.0,
            }

        return {
            "evaluation": evaluation,
            "guardrails": guardrails_result,
            "risk": risk,
            "safe_response": guardrails_result.get("safe_response") or response_text,
        }

    # ================================================================
    # Trace + thread + eval persistence (shared)
    # ================================================================
    def _persist_trace_and_eval(
        self,
        request: AIQueryRequest,
        tenant_id: str,
        user_id: str,
        prepared: _PreparedPipeline,
        response_text: str,
        model_provider: str,
        model_name: str,
        latency_ms: int,
        governance: dict | None = None,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> Optional[str]:
        """
        Persist the trace, append user + assistant turns to the
        thread, run the evaluator, and write a response_risk row.
        Returns the trace id (or None when the trace write failed).
        Every step is wrapped in try/except so persistence failures
        never propagate to the user-visible response.
        """
        retrieved_chunks = prepared.retrieved_chunks
        retrieved_chunk_ids = [c.chunk_id for c in retrieved_chunks]
        retrieved_content = "\n\n".join(c.content for c in retrieved_chunks)

        risk_payload = (governance or {}).get("risk") or {}
        guardrails_payload = (governance or {}).get("guardrails") or {}

        cost = estimate_cost(
            provider=model_provider,
            model=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        trace_id: Optional[str] = None
        try:
            trace = self.trace_service.create_trace(
                CreateTraceRequest(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    project_id=str(prepared.project.id) if prepared.project else None,
                    thread_id=str(prepared.thread.id) if prepared.thread else None,
                    user_query=request.query,
                    prompt_text=prepared.final_prompt,
                    use_case=request.use_case,
                    model_provider=model_provider,
                    model_name=model_name,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieved_content=retrieved_content,
                    response_text=response_text,
                    response_mode=request.response_mode,
                    latency_ms=latency_ms,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    cost_estimate_usd=cost,
                    risk_level=risk_payload.get("risk_level"),
                    requires_disclaimer=bool(guardrails_payload.get("requires_disclaimer")),
                    client_id=tenant_id,
                )
            )
            trace_id = str(trace.id)
        except Exception as e:
            logger.error(f"Failed to save trace: {e}", exc_info=True)

        if prepared.thread is not None:
            try:
                self.thread_service.append_message(
                    thread_id=str(prepared.thread.id),
                    request=AppendMessageRequest(
                        role="user", content=request.query
                    ),
                )
                self.thread_service.append_message(
                    thread_id=str(prepared.thread.id),
                    request=AppendMessageRequest(
                        role="assistant",
                        content=response_text,
                        trace_id=trace_id,
                    ),
                )
            except Exception as e:
                logger.warning(f"Failed to persist thread messages: {e}")

        if trace_id is not None:
            try:
                self.eval_service.evaluate_trace(
                    trace_id=trace_id,
                    response_text=response_text,
                    chunk_texts=[c.content for c in retrieved_chunks],
                    response_mode=request.response_mode,
                    query=request.query,
                    chunks=retrieved_chunks,
                    instructions=None,
                    # The judge already ran in _run_governance_pipeline; the
                    # service still re-runs the heuristics + persists the
                    # full row. We disable the judge here to save the
                    # double-cost when governance was already produced.
                    use_llm_judge=governance is None,
                )
            except Exception as e:
                logger.warning(f"Evaluation failed for trace {trace_id}: {e}")

        if trace_id is not None and risk_payload:
            try:
                self.response_risk_service.persist(
                    trace_id=trace_id,
                    tenant_id=tenant_id,
                    client_id=tenant_id,
                    risk=risk_payload,
                )
            except Exception as e:
                logger.warning(f"Risk persistence failed for trace {trace_id}: {e}")

        return trace_id

    # ================================================================
    # Structured use-case pipeline (control_ranking, iso_checklist)
    # ================================================================
    def _query_structured(
        self,
        request: AIQueryRequest,
        tenant_id: str,
        user_id: str,
    ) -> AIQueryResponse:
        """
        Lightweight pipeline for use_cases that need a fixed JSON output schema.
        Skips RAG, thread management, and the full governance pipeline.
        Returns the raw LLM output as response_text (a JSON string).
        """
        full_prompt = self._build_structured_prompt(request)

        llm_response = self.llm_router.route(
            LLMRequest(prompt=full_prompt, use_case=request.use_case)
        )
        logger.info(
            f"Structured LLM response: {len(llm_response.text)} chars, "
            f"{llm_response.latency_ms}ms via {llm_response.model_provider}"
        )

        trace_id: Optional[str] = None
        try:
            trace = self.trace_service.create_trace(
                CreateTraceRequest(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    project_id=None,
                    thread_id=None,
                    user_query=request.query,
                    prompt_text=full_prompt,
                    use_case=request.use_case,
                    model_provider=llm_response.model_provider,
                    model_name=llm_response.model_name,
                    retrieved_chunk_ids=[],
                    retrieved_content="",
                    response_text=llm_response.text,
                    response_mode="text",
                    latency_ms=llm_response.latency_ms,
                    prompt_tokens=getattr(llm_response, "prompt_tokens", None),
                    completion_tokens=getattr(llm_response, "completion_tokens", None),
                    cost_estimate_usd=None,
                    risk_level=None,
                    requires_disclaimer=False,
                    client_id=tenant_id,
                )
            )
            trace_id = str(trace.id)
        except Exception as e:
            logger.warning(f"Structured trace save failed (non-fatal): {e}")

        total_tokens = (
            (llm_response.prompt_tokens or 0) + (llm_response.completion_tokens or 0)
            if (
                llm_response.prompt_tokens is not None
                or llm_response.completion_tokens is not None
            )
            else None
        )

        # Strip markdown code fences the LLM may wrap around JSON output.
        raw_text = llm_response.text.strip()
        if raw_text.startswith("```"):
            raw_text = raw_text.split("\n", 1)[-1]  # drop opening ```json line
            if raw_text.endswith("```"):
                raw_text = raw_text[: raw_text.rfind("```")].rstrip()

        return AIQueryResponse(
            response_text=raw_text,
            trace_id=trace_id,
            thread_id=None,
            project_id=None,
            model_provider=llm_response.model_provider,
            model_name=llm_response.model_name,
            latency_ms=llm_response.latency_ms,
            use_case=request.use_case,
            retrieved_chunk_count=0,
            retrieved_chunks=[],
            used_reasoning_memory=False,
            prompt_tokens=llm_response.prompt_tokens,
            completion_tokens=llm_response.completion_tokens,
            total_tokens=total_tokens,
            cost_estimate_usd=None,
        )

    def _build_structured_prompt(self, request: AIQueryRequest) -> str:
        if request.use_case == "control_ranking":
            return self._build_control_ranking_prompt(request)
        if request.use_case == "iso_checklist":
            return self._build_iso_checklist_prompt(request)
        raise ValueError(f"No structured prompt builder for use_case: {request.use_case}")

    @staticmethod
    def _build_control_ranking_prompt(request: AIQueryRequest) -> str:
        ctx = request.control_ranking_context or {}
        threat = ctx.get("threat") or {}
        controls = ctx.get("controls") or []
        project = ctx.get("project") or {}

        project_lines = []
        if project.get("ai_component_type"):
            project_lines.append(f"- Tipo de componente IA: {project['ai_component_type']}")
        if project.get("direct_impact") is not None:
            project_lines.append(f"- Impacto directo sobre personas: {project['direct_impact']}")
        if project.get("affected_users_count"):
            project_lines.append(f"- Usuarios afectados: {project['affected_users_count']}")
        if project.get("uses_third_party") is not None:
            project_lines.append(f"- Usa modelos/datos de terceros: {project['uses_third_party']}")
        if project.get("human_oversight_level"):
            project_lines.append(f"- Nivel de supervisión humana: {project['human_oversight_level']}")
        if project.get("explainability_level"):
            project_lines.append(f"- Nivel de explicabilidad requerido: {project['explainability_level']}")
        if project.get("bias_evaluation_status"):
            project_lines.append(f"- Estado evaluación de sesgo: {project['bias_evaluation_status']}")

        controls_text = "\n".join(
            f"- {c.get('control_code')}: {c.get('title_es')}"
            + (f" — {c.get('description_es')}" if c.get("description_es") else "")
            for c in controls
        )

        system = (
            "Eres un experto en gestión de riesgos de sistemas de inteligencia artificial bajo ISO 42001.\n"
            "Tu tarea es rankear controles ISO 42001 según su relevancia para un proyecto específico.\n"
            'Debes devolver ÚNICAMENTE un JSON válido con la clave "rankings" que contenga un array, '
            "sin texto adicional, sin markdown, sin explicaciones fuera del JSON.\n\n"
            'Formato de respuesta requerido:\n{"rankings": [\n'
            '  { "control_code": "A.X.X", "rank": 1, "justification": "Máximo 2 oraciones." }\n]}\n\n'
            "Los ranks deben ser únicos, consecutivos desde 1, ordenados de mayor a menor relevancia."
        )

        user = (
            f"Rankea los siguientes controles ISO 42001:\n\n"
            f"CONTEXTO DEL PROYECTO:\n"
            + ("\n".join(project_lines) if project_lines else "- No especificado")
            + f"\n\nAMENAZA A TRATAR:\n"
            f"- Código: {threat.get('source_code', 'N/A')}\n"
            f"- Nombre: {threat.get('threat_name', 'N/A')}\n"
            f"- Dominio: {threat.get('threat_domain', 'general')}\n"
            f"- Severidad: {threat.get('severity_label', 'N/A')}\n"
            f"- Estrategia elegida: {threat.get('strategy', 'N/A')}\n\n"
            f"CONTROLES A RANKEAR ({len(controls)} controles):\n{controls_text}\n\n"
            f"Devuelve el JSON con los {len(controls)} controles rankeados."
        )

        return f"{system}\n\nPregunta del usuario: {user}"

    @staticmethod
    def _build_iso_checklist_prompt(request: AIQueryRequest) -> str:
        ctx = request.iso_checklist_context or {}
        project = ctx.get("project") or {}
        phases = ctx.get("phases") or []
        gates = ctx.get("gates") or []

        phases_text = json.dumps(
            [{"fase": p.get("phase_number"), "nombre": p.get("phase_name"), "estado": p.get("status")} for p in phases],
            ensure_ascii=False,
        )
        gates_text = json.dumps(
            [{"gate": g.get("gate_code"), "decision": g.get("decision")} for g in gates],
            ensure_ascii=False,
        )

        system = (
            "Eres un experto en ISO/IEC 42001:2023. Genera un checklist de conformidad en JSON.\n"
            "Responde SOLO con JSON válido con esta estructura:\n"
            "{\n"
            '  "overall_score": number (0-100),\n'
            '  "ready_for_go_live": boolean,\n'
            '  "clauses": [\n'
            '    {\n'
            '      "clause": "4.1",\n'
            '      "title": "Comprensión del contexto organizacional",\n'
            '      "status": "compliant" | "partial" | "gap" | "not_applicable",\n'
            '      "evidence": "descripción de evidencia existente",\n'
            '      "gap": "descripción de la brecha o null"\n'
            '    }\n'
            '  ],\n'
            '  "critical_gaps": ["brechas críticas que bloquean el lanzamiento"],\n'
            '  "recommendations": ["recomendaciones antes del go-live"]\n'
            "}\n"
            "Cláusulas a evaluar: 4.1, 4.2, 4.3, 5.1, 5.2, 5.3, 6.1, 6.2, 7.1, 7.2, 7.4, 7.5, 8.1, 8.4, 9.1, 10.1"
        )

        user = (
            f"Proyecto: {project.get('name', 'Sin nombre')}\n"
            f"Tipo de componente IA: {project.get('ai_component_type', 'No especificado')}\n"
            f"Nivel de oversight humano: {project.get('human_oversight_level', 'No especificado')}\n"
            f"Usa terceros: {'Sí' if project.get('uses_third_party') else 'No'}\n"
            f"Impacto directo: {'Sí' if project.get('direct_impact') else 'No'}\n"
            f"Usuarios afectados: {project.get('affected_users_count', 'No especificado')}\n"
            f"Explicabilidad: {project.get('explainability_level', 'No especificado')}\n"
            f"Evaluación de sesgo: {project.get('bias_evaluation_status', 'No realizada')}\n"
            f"Fases: {phases_text}\n"
            f"Gates previos: {gates_text}"
        )

        return f"{system}\n\nPregunta del usuario: {user}"

    # ================================================================
    # Internal helpers
    # ================================================================
    def _load_doc_metadata(self, chunks: list) -> dict[str, dict]:
        """
        Fetch (filename, framework) for every distinct document_id that
        appears in the chunk list. Returns {} on any DB error — the prompt
        layer falls back to "(documento desconocido)".

        Best-effort: a failure here must not break the main query flow.
        """
        if not chunks:
            return {}
        doc_ids = list({c.document_id for c in chunks})
        try:
            r = (
                self.db.table("documents")
                .select("id, filename, framework")
                .in_("id", doc_ids)
                .execute()
            )
            return {
                str(row["id"]): {
                    "filename": row.get("filename") or "documento sin nombre",
                    "framework": row.get("framework"),
                }
                for row in (r.data or [])
            }
        except Exception as e:
            logger.warning(f"Could not hydrate doc metadata for chunks: {e}")
            return {}

    def _assemble_prompt(
        self,
        system_prompt: str,
        project: Project | None,
        thread_messages: list[Message],
        reasoning_traces: list[Trace],
        chunks: list,
        doc_metadata: dict[str, dict],
        user_query: str,
        response_mode: str = "text",
        inline_project_context: dict | None = None,
        inline_chat_history: list[dict] | None = None,
    ) -> str:
        """
        Build the final prompt string. Sections are added only when the
        corresponding memory is non-empty so the LLM is not distracted
        by empty scaffolding.

        Layout:

          [System Prompt + Safety Rules]

          [Memoria del proyecto]          ← only if project != None
          [Razonamiento previo relevante] ← only if reasoning_traces
          [Context Documents]             ← RAG output
          [Conversación reciente]         ← thread tail
          [User Question]
        """
        parts: list[str] = [system_prompt.strip()]

        # --- Project block (L2) ---
        # DB project wins; inline context is the fallback for callers
        # (e.g. the GovAI Edge Function proxy) that haven't synced
        # ai_projects into the Python projects table yet.
        if project is not None:
            parts.append(self._format_project_block(project))
        elif inline_project_context:
            parts.append(self._format_inline_project_block(inline_project_context))

        # --- Reasoning memory (lightweight L3) ---
        if reasoning_traces:
            parts.append(self._format_reasoning_block(reasoning_traces))

        # --- Retrieved context ---
        if chunks:
            context_header = "\n\nDocumentos recuperados:\n" + "=" * 50

            def _format_one(i: int, c) -> str:
                meta = doc_metadata.get(c.document_id) or {}
                filename = meta.get("filename") or "documento desconocido"
                framework = meta.get("framework")
                section_ref = getattr(c, "section_ref", None)
                # Optional fragments. Each is omitted entirely when its
                # source value is missing so the model never sees an
                # empty "Section: " or "Framework: " label.
                section_part = f" | Section: {section_ref}" if section_ref else ""
                fw_part = f" | Framework: {framework}" if framework else ""
                return (
                    f"[Chunk {i+1} | Fuente: {filename}{section_part}{fw_part} "
                    f"| similarity: {c.similarity:.3f}]\n{c.content}"
                )

            chunk_texts = "\n\n".join(
                _format_one(i, c) for i, c in enumerate(chunks)
            )
            context_footer = "=" * 50
            parts.append(f"{context_header}\n{chunk_texts}\n{context_footer}")
        else:
            parts.append(
                "\n\nNota: No se encontraron documentos relevantes en la base de "
                "conocimiento. Aplica el GROUNDING CONTRACT: responde solo si "
                "puedes hacerlo sin usar conocimiento de entrenamiento."
            )

        # --- Conversation tail (L1) ---
        # Priority: DB thread > inline history from Edge Function caller.
        # The current user query is already in user_query below, so it
        # must not appear in the history section.
        if thread_messages:
            parts.append(self._format_thread_block(thread_messages))
        elif inline_chat_history:
            block = self._format_inline_chat_block(inline_chat_history)
            if block:
                parts.append(block)

        # --- Structured output format hint ---
        if response_mode == "structured":
            parts.append(_STRUCTURED_OUTPUT_INSTRUCTION)

        # --- User question ---
        parts.append(f"\n\nPregunta del usuario: {user_query}")

        assembled = "\n".join(parts)

        # Pre-call token estimation — logged always, enforced only when
        # context_budget_enabled=True (future phases).
        estimated_tokens = _token_counter.count(assembled, settings.llm_default_model)
        input_budget = settings.context_window_tokens - settings.reserved_output_tokens
        logger.info(
            "context_tokens_estimated",
            extra={
                "estimated_input_tokens": estimated_tokens,
                "reserved_output_tokens": settings.reserved_output_tokens,
                "input_budget": input_budget,
                "budget_used_pct": round(estimated_tokens / input_budget * 100, 1),
                "chunks_count": len(chunks),
                "history_messages": len(thread_messages),
                "traces_count": len(reasoning_traces),
                "response_mode": response_mode,
                "context_budget_enabled": settings.context_budget_enabled,
            },
        )

        return assembled

    @staticmethod
    def _format_project_block(project: Project) -> str:
        """
        Compact, token-frugal rendering of the project expediente.

        Keep this short: the LLM does not need the raw JSON of the risk
        register, only a summary that it can reference. If the user asks
        for details, the reasoning block + chunks will provide them.
        """
        profile = project.project_profile
        profile_lines = []
        if profile.project_name:
            profile_lines.append(f"- Nombre del proyecto: {profile.project_name}")
        if profile.project_description:
            profile_lines.append(f"- Descripción: {profile.project_description}")
        if profile.ai_type:
            profile_lines.append(f"- Tipo de IA: {profile.ai_type}")
        if profile.data_used:
            profile_lines.append(f"- Datos usados: {', '.join(profile.data_used)}")
        if profile.industry:
            profile_lines.append(f"- Industria: {profile.industry}")
        if profile.affected_audience:
            profile_lines.append(f"- Público afectado: {profile.affected_audience}")
        if profile.autonomy_level:
            profile_lines.append(f"- Autonomía: {profile.autonomy_level}")
        if profile.deployment:
            profile_lines.append(f"- Despliegue: {profile.deployment}")
        if profile.country:
            profile_lines.append(f"- País: {profile.country}")

        risk_summary = ""
        if isinstance(project.risk_register, dict):
            tier = project.risk_register.get("eu_ai_act_tier")
            risks = project.risk_register.get("risks") or []
            open_risks = [
                f"{r.get('id', '?')} ({r.get('category', '?')}, score={r.get('inherent_score', '?')})"
                for r in risks[:5]
            ]
            risk_summary = (
                "\nRiesgos identificados: "
                + (", ".join(open_risks) if open_risks else "(ninguno)")
            )
            if tier:
                risk_summary += f"\nTier EU AI Act: {tier}"

        policy_summary = (
            "\nPolítica: generada"
            if isinstance(project.policy, dict)
            else "\nPolítica: pendiente"
        )

        block = (
            "\n\n═══ CONTEXTO DEL PROYECTO ═══\n"
            + (f"Estado: {project.status}\n" if project.status else "")
            + "\n".join(profile_lines)
            + risk_summary
            + policy_summary
        )
        return block

    @staticmethod
    def _format_inline_project_block(ctx: dict) -> str:
        """
        Render an inline `project_context` payload as a compact prompt block.

        Handles two shapes:
          1. Active-project context: name, risk_level, current_phase, active_risks, etc.
          2. Dashboard/institution context: institution_state (stats) + user_projects list.

        Unknown fields are silently dropped so the prompt stays token-frugal.
        """
        authority_note = (
            "DATOS AUTORITATIVOS — Para cualquier pregunta sobre proyectos, riesgos, "
            "aprobaciones o estado de la institución, responde DIRECTAMENTE desde este "
            "bloque. No es necesario citar documentos normativos para estos datos.\n"
        )

        # ── Field-draft block (rendered first, takes priority) ──
        field_block = ""
        field_info = ctx.get("field_to_draft")
        if isinstance(field_info, dict) and field_info.get("field_label"):
            label = field_info.get("field_label", "")
            desc = field_info.get("field_description") or ""
            step_num = field_info.get("step_number") or ""
            step_name = field_info.get("step_name") or ""
            step_ref = f"Paso {step_num} — {step_name}" if step_num else step_name
            field_block = (
                "\n\n═══ CAMPO A REDACTAR ═══\n"
                f"Campo: {label}\n"
                + (f"Descripción: {desc}\n" if desc else "")
                + (f"Paso: {step_ref}\n" if step_ref else "")
                + "INSTRUCCIÓN: Genera exclusivamente el texto del borrador para este campo. "
                "Debe ser profesional, conciso y adecuado para un expediente de gobernanza de IA. "
                "Responde solo con el contenido del campo, sin explicaciones ni encabezados adicionales."
            )

        # ── Institution block (shared for both shapes) ──
        institution = ctx.get("institution") or {}
        inst_lines: list[str] = []
        if isinstance(institution, dict):
            if institution.get("name"):
                inst_lines.append(f"- Institución: {institution['name']}")
            if institution.get("country"):
                inst_lines.append(f"- País: {institution['country']}")
            if institution.get("industry"):
                inst_lines.append(f"- Industria: {institution['industry']}")
            regs = institution.get("regulations") or []
            if regs:
                inst_lines.append(f"- Regulaciones aplicables: {', '.join(regs)}")

        # ── Shape 1: Active project ──
        project_lines: list[str] = []
        name = ctx.get("name") or ctx.get("project_name")
        if name:
            project_lines.append(f"- Nombre del proyecto: {name}")
        if ctx.get("description") or ctx.get("project_description"):
            project_lines.append(
                f"- Descripción: {ctx.get('description') or ctx.get('project_description')}"
            )
        if ctx.get("risk_level"):
            project_lines.append(f"- Nivel de riesgo: {ctx['risk_level']}")
        if ctx.get("risk_score") is not None:
            project_lines.append(f"- Score de riesgo: {ctx['risk_score']}/100")
        if ctx.get("current_phase"):
            project_lines.append(f"- Fase actual: {ctx['current_phase']}")
        if ctx.get("current_step"):
            project_lines.append(f"- Paso actual: {ctx['current_step']}")
        if ctx.get("role"):
            inst_lines.append(f"- Rol del usuario: {ctx['role']}")

        if ctx.get("phase_status"):
            project_lines.append(f"- Estado de la fase: {ctx['phase_status']}")
        if ctx.get("owner_name"):
            project_lines.append(f"- Project Leader: {ctx['owner_name']}")
        if ctx.get("assigned_tl_name"):
            project_lines.append(f"- Technical Leader: {ctx['assigned_tl_name']}")

        pending_steps = ctx.get("pending_steps") or []
        if pending_steps:
            project_lines.append(f"- Pasos pendientes en fase actual: {', '.join(str(s) for s in pending_steps)}")

        # Phase steps: full step-level breakdown for current phase
        phase_steps = ctx.get("phase_steps") or []
        if isinstance(phase_steps, list) and phase_steps:
            done = [s for s in phase_steps if isinstance(s, dict) and s.get("status") == "completed"]
            in_prog = [s for s in phase_steps if isinstance(s, dict) and s.get("status") == "in_progress"]
            pending = [s for s in phase_steps if isinstance(s, dict) and s.get("status") not in ("completed", "in_progress")]
            project_lines.append(
                f"- Pasos de la fase actual ({len(phase_steps)} total): "
                f"{len(done)} completados, {len(in_prog)} en progreso, {len(pending)} pendientes"
            )
            for s in phase_steps:
                if isinstance(s, dict):
                    snum = s.get("stepNumber") or s.get("step_number", "?")
                    sname = s.get("stepName") or s.get("step_name", "")
                    sstatus = s.get("status", "?")
                    label = f"{snum}"
                    if sname and sname != snum:
                        label += f" — {sname}"
                    project_lines.append(f"  · {label}: {sstatus}")

        # Step field completion: which fields are filled/empty in the active step
        step_field_completion = ctx.get("step_field_completion") or []
        if isinstance(step_field_completion, list) and step_field_completion:
            empty_fields = [f.get("field", "?") for f in step_field_completion if isinstance(f, dict) and not f.get("filled")]
            filled_count = sum(1 for f in step_field_completion if isinstance(f, dict) and f.get("filled"))
            total_fields = len(step_field_completion)
            project_lines.append(
                f"- Campos del paso activo: {filled_count}/{total_fields} completados"
            )
            if empty_fields:
                project_lines.append(f"  · Campos vacíos: {', '.join(empty_fields[:10])}")

        # Evidence summary: required vs uploaded per step
        evidence_summary = ctx.get("evidence_summary") or []
        if isinstance(evidence_summary, list) and evidence_summary:
            gaps = [
                e for e in evidence_summary
                if isinstance(e, dict) and (e.get("required", 0) or 0) > (e.get("uploaded", 0) or 0)
            ]
            if gaps:
                gap_texts = [
                    f"Paso {e.get('stepNumber', '?')}: {e.get('uploaded', 0)}/{e.get('required', 0)} evidencias"
                    for e in gaps[:5]
                ]
                project_lines.append(f"- Evidencias faltantes: {', '.join(gap_texts)}")

        pending_conditions = ctx.get("pending_conditions") or []
        open_conditions = [c for c in pending_conditions if isinstance(c, dict) and not c.get("resolved")]
        if open_conditions:
            cond_texts = [c.get("text", "?") for c in open_conditions[:3]]
            project_lines.append(f"- Condiciones Gate abiertas: {'; '.join(cond_texts)}")

        gate_history = ctx.get("gate_history") or []
        if gate_history:
            last_gate = gate_history[-1]
            if isinstance(last_gate, dict):
                project_lines.append(
                    f"- Último Gate: Fase {last_gate.get('phase', '?')} → {last_gate.get('decision', '?')}"
                )

        # Risks: prefer allRisks (full detail) over activeRisks
        risks_summary = ""
        all_risks = ctx.get("all_risks") or []
        active_risks = ctx.get("active_risks") or ctx.get("risks") or []
        risks_to_render = all_risks if all_risks else active_risks
        if isinstance(risks_to_render, list) and risks_to_render:
            unmitigated = [r for r in risks_to_render if isinstance(r, dict) and not r.get("mitigated")]
            mitigated = [r for r in risks_to_render if isinstance(r, dict) and r.get("mitigated")]
            risks_summary = f"\n- Registro de riesgos: {len(risks_to_render)} total ({len(unmitigated)} sin mitigar, {len(mitigated)} mitigados)"
            top_unmitigated = sorted(
                [r for r in unmitigated if isinstance(r, dict)],
                key=lambda r: r.get("severity") or 0,
                reverse=True
            )[:5]
            for r in top_unmitigated:
                threat = r.get("threatName") or r.get("threat_name") or "?"
                sev = r.get("severityLabel") or r.get("severity_label") or "?"
                iso = r.get("isoControlCode") or r.get("iso_control_code") or ""
                iso_part = f" | control ISO: {iso}" if iso else ""
                risks_summary += f"\n  · {threat} (severidad: {sev}{iso_part})"

        # ── Shape 2: Dashboard / institution-level context ──
        institution_state = ctx.get("institution_state")
        user_projects = ctx.get("user_projects") or []

        inst_state_section = ""
        if isinstance(institution_state, dict):
            total = institution_state.get("total_governed_projects", 0)
            pending_approvals = institution_state.get("pending_approvals", 0)
            cond_vencing = institution_state.get("conditions_vencing_this_week", 0)

            inst_state_lines = [
                f"\n\n═══ ESTADO DE LA INSTITUCIÓN ═══",
                f"{authority_note}",
                f"- Proyectos gobernados (total): {total}",
                f"- Aprobaciones pendientes: {pending_approvals}",
                f"- Condiciones por vencer esta semana: {cond_vencing}",
            ]

            projects_by_status = institution_state.get("projects_by_status") or []
            if isinstance(projects_by_status, list) and projects_by_status:
                for entry in projects_by_status:
                    if isinstance(entry, dict):
                        status = entry.get("status", "?")
                        count = entry.get("count", 0)
                        names = entry.get("names") or []
                        names_str = f" ({', '.join(names[:5])})" if names else ""
                        inst_state_lines.append(f"  · {status}: {count} proyecto(s){names_str}")

            pending_names = institution_state.get("pending_project_names") or []
            if pending_names:
                inst_state_lines.append(f"- Proyectos pendientes de aprobación: {', '.join(pending_names)}")

            upcoming = institution_state.get("upcoming_conditions") or []
            if upcoming:
                inst_state_lines.append("- Condiciones próximas a vencer:")
                for uc in upcoming[:3]:
                    if isinstance(uc, dict):
                        proj = uc.get("projectName", "?")
                        text = uc.get("conditionText", "?")
                        deadline = uc.get("deadline") or "sin fecha"
                        inst_state_lines.append(f"  · [{proj}] {text} (vence: {deadline})")

            inst_state_section = "\n".join(inst_state_lines)

        user_projects_section = ""
        if isinstance(user_projects, list) and user_projects:
            up_lines = ["\n\n═══ PROYECTOS DEL USUARIO ═══"]
            for p in user_projects:
                if isinstance(p, dict):
                    pname = p.get("name", "?")
                    pstatus = p.get("status", "?")
                    phase = p.get("currentPhase", "?")
                    risk = p.get("riskLevel") or "N/A"
                    pending = p.get("pendingSteps", 0)
                    up_lines.append(
                        f"- {pname}: estado={pstatus}, fase={phase}, riesgo={risk}, pasos_pendientes={pending}"
                    )
            user_projects_section = "\n".join(up_lines)

        # ── Assemble final output ──

        # Active-project path: only when a project name is present
        if name and project_lines:
            project_section = (
                "\n\n═══ CONTEXTO DEL PROYECTO ═══\n"
                + authority_note
                + "\n".join(inst_lines + project_lines)
                + risks_summary
            )
            return field_block + project_section

        # Dashboard path: institution-level stats (no active project selected)
        if inst_state_section or user_projects_section or inst_lines:
            inst_header = (
                ("\n\n═══ CONTEXTO INSTITUCIONAL ═══\n" + authority_note + "\n".join(inst_lines))
                if inst_lines else ""
            )
            return field_block + inst_header + inst_state_section + user_projects_section

        return field_block

    @staticmethod
    def _format_reasoning_block(traces: list[Trace]) -> str:
        """
        Condense the most recent analysis traces into a short block so
        the LLM can answer "¿por qué?" questions by paraphrasing its own
        past output.
        """
        if not traces:
            return ""
        lines = [
            "\n\n═══ ANÁLISIS PREVIOS (usa esto para justificar decisiones pasadas) ═══"
        ]
        for t in traces:
            # Truncate the assistant text to keep the prompt bounded.
            snippet = (t.response_text or "").strip().replace("\n", " ")
            if len(snippet) > 600:
                snippet = snippet[:600] + "…"
            lines.append(
                f"- [{t.created_at.isoformat(timespec='minutes')}] "
                f"{t.use_case}: {snippet}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_thread_block(messages: list[Message]) -> str:
        """
        Render conversation history as a transcript. System-role messages
        are omitted (they're internal notices, not user-visible).
        """
        lines = ["\n\n═══ CONVERSACIÓN RECIENTE ═══"]
        for m in messages:
            if m.role == "system":
                continue
            prefix = "Usuario" if m.role == "user" else "Asistente"
            content = m.content.strip().replace("\n", " ")
            if len(content) > 500:
                content = content[:500] + "…"
            lines.append(f"{prefix}: {content}")
        return "\n".join(lines)

    @staticmethod
    def _format_inline_chat_block(messages: list[dict]) -> str:
        """
        Render an inline chat_history list (from the Edge Function) as a
        conversation transcript. Used when no thread_id is provided so the
        sidecar can maintain turn-by-turn context across messages.
        """
        lines = ["\n\n═══ CONVERSACIÓN RECIENTE ═══"]
        for m in messages:
            role = m.get("role", "")
            if role == "system":
                continue
            prefix = "Usuario" if role == "user" else "Asistente"
            content = str(m.get("content", "")).strip().replace("\n", " ")
            if len(content) > 500:
                content = content[:500] + "…"
            lines.append(f"{prefix}: {content}")
        # Return empty string if only the header ended up in the list
        return "\n".join(lines) if len(lines) > 1 else ""


# ============================================================
# SSE encoding helper
#
# Server-Sent Events use a line-based framing:
#
#   event: <event_name>
#   data: <utf-8 line>
#   data: <utf-8 line>
#   <empty line>            <- terminates the event
#
# We always serialise the payload as a single-line JSON string
# so multiline content cannot break the framing. The empty line
# is required by the SSE spec — without it, the browser keeps
# buffering the event.
# ============================================================
def _sse_event(event_name: str, payload: dict) -> bytes:
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_name}\ndata: {body}\n\n".encode("utf-8")


# ============================================================
# Extracted constant — kept at module level so tests can assert
# its presence and manual prompt reviewers can eyeball it.
# ============================================================
_STRUCTURED_OUTPUT_INSTRUCTION = """

OUTPUT FORMAT: Respond with valid JSON only. No text before or after.

{
  "answer": "<string in Spanish, prose, the actual response>",
  "citations": [
    {"chunk_id": "<string>", "document_name": "<string>", "section": "<string>", "quote": "<string>"}
  ],
  "confidence": "<string: high | medium | low>",
  "suggested_followups": ["<string>"]
}

Rules:
- "citations" may be empty but MUST be present as an array.
- "confidence": "high" if context directly and completely answers; "medium" if partially; "low" if inference was required.
- Do not include JSON comments. No trailing commas. No markdown code fences.
- If unable to produce valid JSON, output: {"answer": "<plain text>", "citations": [], "confidence": "low", "suggested_followups": []}
"""
