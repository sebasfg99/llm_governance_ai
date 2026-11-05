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
from app.domain.retrieval.query_builder import (
    build_retrieval_query,
    extract_user_question,
    infer_framework_filter,
    split_top_k_across_frameworks,
)
from app.domain.retrieval.ranking import ChunkResult
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


_EVIDENCE_QUERY_KEYWORDS = (
    "evidencias",
    "evidencia",
    "controles",
    "cumplimiento",
    "checklist",
)


def _looks_like_evidence_query(query: str) -> bool:
    lowered = query.strip().lower()
    return any(keyword in lowered for keyword in _EVIDENCE_QUERY_KEYWORDS)


def _build_evidence_checklist_hint(project: Project) -> str:
    """
    Build a concrete, project-specific instruction for evidence/checklist
    questions so the LLM must reference expedition facts, not just generic
    ISO/NIST guidance.
    """
    profile = project.project_profile
    must_mention: list[str] = []

    if profile.ai_type:
        must_mention.append(f"tipo de IA: {profile.ai_type}")
    if profile.data_used:
        must_mention.append(
            "datos: " + ", ".join(str(d) for d in profile.data_used)
        )
    if profile.industry:
        must_mention.append(f"sector: {profile.industry}")
    if profile.autonomy_level:
        must_mention.append(f"autonomía: {profile.autonomy_level}")
    if profile.existing_vendor:
        must_mention.append(f"proveedores: {profile.existing_vendor}")
    if profile.notes:
        # First 300 chars of notes — enough for archivos/equipo/comité.
        notes_excerpt = " ".join(profile.notes.split())[:300]
        must_mention.append(f"notas del expediente: {notes_excerpt}")

    facts_line = (
        "; ".join(must_mention)
        if must_mention
        else "los hechos del CONTEXTO DEL PROYECTO anterior"
    )

    return (
        "\n[Instrucción para esta pregunta: estructura la respuesta como checklist "
        "accionable numerado. Cada ítem debe (1) citar un control/cláusula de "
        "ISO 42001 Anexo A o una función/outcome de NIST "
        "(GOVERN/MAP/MEASURE/MANAGE) con [Fuente: ...], y (2) personalizar al "
        f"proyecto. OBLIGATORIO mencionar explícitamente: {facts_line}. "
        "Al menos un ítem debe referirse a proveedores externos y al menos uno "
        "a archivos o equipo del expediente cuando estén disponibles.]"
    )


# ------------------------------------------------------------
# Use cases that expect a conversational experience. When one of
# these is used without an explicit thread_id we auto-create a
# thread so future turns continue the same conversation.
# ------------------------------------------------------------
_CONVERSATIONAL_USE_CASES = {"chatbot", "risk_intake", "policy_generation"}


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
            },
        )

    def _retrieve_context_chunks(
        self,
        *,
        tenant_id: str,
        retrieval_query: str,
        top_k: int,
        include_global: bool,
        country: str | None,
        framework_filter: list[str] | None,
    ) -> list[ChunkResult]:
        """
        Run RAG retrieval, splitting top_k across frameworks when multiple
        are inferred so no single framework monopolises the context window.
        """
        if not framework_filter or len(framework_filter) < 2:
            return self.retrieval_service.retrieve(
                tenant_id=tenant_id,
                query=retrieval_query,
                top_k=top_k,
                include_global=include_global,
                country=country,
                framework_filter=framework_filter,
            )

        allocation = split_top_k_across_frameworks(top_k, framework_filter)
        pool: dict[str, ChunkResult] = {}
        for fw, per_fw_k in allocation:
            fw_chunks = self.retrieval_service.retrieve(
                tenant_id=tenant_id,
                query=retrieval_query,
                top_k=per_fw_k,
                include_global=include_global,
                country=country,
                framework_filter=[fw],
            )
            for chunk in fw_chunks:
                existing = pool.get(chunk.chunk_id)
                if existing is None or chunk.similarity > existing.similarity:
                    pool[chunk.chunk_id] = chunk

        merged = sorted(pool.values(), key=lambda c: c.similarity, reverse=True)
        logger.info(
            "balanced_framework_retrieval",
            extra={
                "frameworks": framework_filter,
                "allocation": allocation,
                "merged_count": len(merged[:top_k]),
            },
        )
        return merged[:top_k]

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
        # --------------------------------------------------------
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
        #
        # user_question: the core question extracted from safe_query
        #   (strips long form-fill preamble when present).
        # retrieval_query: enriched with project profile fields so the
        #   embedding captures both regulatory topic and project context.
        # resolved_frameworks: inferred from question text when the
        #   caller did not pass framework_filter explicitly.
        # effective_top_k: bumped to 8 when a project_id is present and
        #   the caller used the default top_k of 5.
        #
        # _assemble_prompt still receives safe_query (the original user
        # wording) — the enrichment only affects the vector search.
        # --------------------------------------------------------
        user_question = extract_user_question(safe_query)
        resolved_frameworks = infer_framework_filter(
            user_question, request.framework_filter
        )
        retrieval_query = build_retrieval_query(
            user_question=user_question, project=project
        )
        effective_top_k = (
            8 if (request.project_id and request.top_k == 5) else request.top_k
        )

        logger.info(
            "retrieval_query_built",
            extra={
                "retrieval_query": retrieval_query[:200],
                "resolved_frameworks": resolved_frameworks,
                "effective_top_k": effective_top_k,
            },
        )

        retrieved_chunks = self._retrieve_context_chunks(
            tenant_id=tenant_id,
            retrieval_query=retrieval_query,
            top_k=effective_top_k,
            include_global=request.include_global,
            country=request.country,
            framework_filter=resolved_frameworks,
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
        # --------------------------------------------------------
        final_prompt = self._assemble_prompt(
            system_prompt=hardened_system_prompt,
            project=project,
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
        if project is not None:
            parts.append(self._format_project_block(project))
            if _looks_like_evidence_query(user_query):
                parts.append(_build_evidence_checklist_hint(project))

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
        # We exclude the message that is about to be sent (it arrives as
        # user_query below) by trusting the caller to append it AFTER the
        # LLM call — the current query is not yet in the thread.
        if thread_messages:
            parts.append(self._format_thread_block(thread_messages))

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
        if profile.existing_vendor:
            profile_lines.append(f"- Proveedores externos: {profile.existing_vendor}")
        if profile.notes:
            notes = profile.notes.strip()
            if len(notes) > 800:
                notes = notes[:800] + "…"
            profile_lines.append(f"- Notas del expediente:\n{notes}")

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
