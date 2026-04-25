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

from typing import Optional

from supabase import Client

from app.core.logging import get_logger

# Domain services
from app.domain.prompts.service import PromptService
from app.domain.retrieval.service import RetrievalService
from app.domain.llm.guardrails import apply_guardrails
from app.domain.llm.router import LLMRouter
from app.domain.llm.models import LLMRequest
from app.domain.traces.service import TraceService
from app.domain.traces.models import CreateTraceRequest, Trace
from app.domain.evals.service import EvalService
from app.domain.projects.service import ProjectService
from app.domain.projects.models import Project
from app.domain.threads.service import ThreadService
from app.domain.threads.models import AppendMessageRequest, Message, Thread

# Schemas
from app.schemas.ai import AIQueryRequest, AIQueryResponse, RetrievedChunkInfo

logger = get_logger(__name__)


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

        # The LLM router is stateless — no DB needed
        self.llm_router = LLMRouter()

    # ================================================================
    # Public entrypoint
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

        # --------------------------------------------------------
        # Step -1 — Load system prompt first so we can apply
        # guardrails before spending any persistence effort. The
        # injection check raises ValidationError (HTTP 422) which
        # must NOT leave a new thread behind.
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
        #
        # Conversational use cases always get a thread. Non-conversational
        # ones (executive_summary, risk_analysis one-shots) only get a
        # thread if the caller explicitly passed one — we don't want to
        # litter the sidebar with summary requests.
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
            framework_filter=request.framework_filter,
        )
        logger.debug(f"Retrieved {len(retrieved_chunks)} context chunks")

        # --------------------------------------------------------
        # Step 4 — "Why?" detection + reasoning memory lookup
        #
        # Only triggered when there's a project in context AND the
        # query reads like a justification request. We fetch the
        # most recent analysis traces from the same project.
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
                # Reasoning memory is best-effort; never block the main flow.
                logger.warning(f"Failed to load reasoning traces: {e}")

        # --------------------------------------------------------
        # Step 5 — Load recent messages (conversation history, L1)
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
        #
        # The retrieval layer returns ChunkResult with document_id only.
        # The LLM needs the document name + framework so it can satisfy
        # the prompt's "[Fuente: <doc>]" citation rule and attribute
        # content to the framework the user is asking about. Without
        # this hydration the model sees opaque "Chunk 1, Chunk 2..."
        # and refuses framework-specific questions.
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

        # --------------------------------------------------------
        # Step 8 — Call the LLM
        # --------------------------------------------------------
        llm_response = self.llm_router.route(
            LLMRequest(prompt=final_prompt, use_case=request.use_case)
        )
        logger.info(
            f"LLM response: {len(llm_response.text)} chars, "
            f"{llm_response.latency_ms}ms via {llm_response.model_provider}"
        )

        # --------------------------------------------------------
        # Step 9 — Persist trace (bound to project & thread)
        # --------------------------------------------------------
        retrieved_chunk_ids = [c.chunk_id for c in retrieved_chunks]
        retrieved_content = "\n\n".join(c.content for c in retrieved_chunks)

        trace_id: Optional[str] = None
        try:
            trace = self.trace_service.create_trace(
                CreateTraceRequest(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    project_id=str(project.id) if project else None,
                    thread_id=str(thread.id) if thread else None,
                    user_query=request.query,
                    prompt_text=final_prompt,
                    use_case=request.use_case,
                    model_provider=llm_response.model_provider,
                    model_name=llm_response.model_name,
                    retrieved_chunk_ids=retrieved_chunk_ids,
                    retrieved_content=retrieved_content,
                    response_text=llm_response.text,
                    response_mode=request.response_mode,
                    latency_ms=llm_response.latency_ms,
                )
            )
            trace_id = str(trace.id)
        except Exception as e:
            logger.error(f"Failed to save trace: {e}", exc_info=True)

        # --------------------------------------------------------
        # Step 10 — Persist thread messages (user + assistant)
        #
        # We write both turns after the trace so the assistant message
        # can reference trace_id. If the trace write failed, we still
        # persist the turns so the conversation is not lost — just
        # without a trace reference.
        # --------------------------------------------------------
        if thread is not None:
            try:
                self.thread_service.append_message(
                    thread_id=str(thread.id),
                    request=AppendMessageRequest(
                        role="user", content=request.query
                    ),
                )
                self.thread_service.append_message(
                    thread_id=str(thread.id),
                    request=AppendMessageRequest(
                        role="assistant",
                        content=llm_response.text,
                        trace_id=trace_id,
                    ),
                )
            except Exception as e:
                logger.warning(f"Failed to persist thread messages: {e}")

        # --------------------------------------------------------
        # Step 11 — Heuristic evaluation (unchanged)
        # --------------------------------------------------------
        if trace_id is not None:
            try:
                self.eval_service.evaluate_trace(
                    trace_id=trace_id,
                    response_text=llm_response.text,
                    chunk_texts=[c.content for c in retrieved_chunks],
                    response_mode=request.response_mode,
                )
            except Exception as e:
                logger.warning(f"Evaluation failed for trace {trace_id}: {e}")

        # --------------------------------------------------------
        # Step 12 — Build response
        # --------------------------------------------------------
        chunk_infos = [
            RetrievedChunkInfo(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                similarity=round(chunk.similarity, 4),
                content_preview=chunk.content[:200],
            )
            for chunk in retrieved_chunks
        ]

        return AIQueryResponse(
            response_text=llm_response.text,
            trace_id=trace_id,
            thread_id=str(thread.id) if thread else None,
            project_id=str(project.id) if project else None,
            model_provider=llm_response.model_provider,
            model_name=llm_response.model_name,
            latency_ms=llm_response.latency_ms,
            use_case=request.use_case,
            retrieved_chunk_count=len(retrieved_chunks),
            retrieved_chunks=chunk_infos,
            used_reasoning_memory=used_reasoning_memory,
        )

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
                fw_part = f" | Framework: {framework}" if framework else ""
                return (
                    f"[Chunk {i+1} | Fuente: {filename}{fw_part} "
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

        return "\n".join(parts)

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
