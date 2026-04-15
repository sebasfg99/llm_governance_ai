# ============================================================
# app/domain/ai_service.py — Central AI query orchestrator
#
# WHY THIS FILE EXISTS:
# The AIService is the most important class in the system.
# It coordinates all domains to handle a single AI query:
#
#   1. Get the system prompt for the use case
#   2. Retrieve relevant document chunks (RAG)
#   3. Apply guardrails (injection detection + hardening)
#   4. Assemble the final prompt (system + context + question)
#   5. Call the LLM router (OpenAI or Gemini)
#   6. Save the full trace (audit log)
#   7. Run automatic evaluation (quality check)
#   8. Return the response to the API caller
#
# DESIGN PRINCIPLE:
# This is the ONLY class that imports from multiple domains.
# Each domain service (prompts, retrieval, llm, traces, evals)
# is isolated and only AIService knows how they connect.
#
# WHY ORCHESTRATION HERE AND NOT IN THE ROUTE:
# Keeping orchestration in a service (not in the FastAPI route)
# means we can test it without HTTP — just call .query() directly.
# Routes should be thin wrappers that call services.
# ============================================================

from supabase import Client

from app.core.logging import get_logger
from app.core.errors import UpstreamError

# Domain services
from app.domain.prompts.service import PromptService
from app.domain.retrieval.service import RetrievalService
from app.domain.llm.guardrails import apply_guardrails
from app.domain.llm.router import LLMRouter
from app.domain.llm.models import LLMRequest
from app.domain.traces.service import TraceService
from app.domain.traces.models import CreateTraceRequest
from app.domain.evals.service import EvalService

# Schemas (for type hints on the return value)
from app.schemas.ai import AIQueryRequest, AIQueryResponse, RetrievedChunkInfo

logger = get_logger(__name__)


class AIService:
    """
    Orchestrates the complete AI query pipeline.

    Coordinates: prompts → retrieval → guardrails → LLM → trace → eval

    Usage:
        db = get_supabase_client()
        service = AIService(db)
        response = service.query(
            request=AIQueryRequest(query="What are the risks?"),
            tenant_id="acme",
            user_id="u_123",
        )
        print(response.response_text)
        print(f"Trace ID: {response.trace_id}")
    """

    def __init__(self, db: Client) -> None:
        """
        Initialize all domain services.

        All services receive the same Supabase client instance.
        The LLM router is stateless and doesn't need a DB client.
        """
        self.db = db

        # Initialize all domain services
        self.prompt_service = PromptService(db)
        self.retrieval_service = RetrievalService(db)
        self.trace_service = TraceService(db)
        self.eval_service = EvalService(db)

        # The LLM router is stateless — no DB needed
        self.llm_router = LLMRouter()

    def query(
        self,
        request: AIQueryRequest,
        tenant_id: str,
        user_id: str,
    ) -> AIQueryResponse:
        """
        Process a complete AI query through the full pipeline.

        Args:
            request:   The validated API request from the caller.
            tenant_id: The tenant making the request (from auth).
            user_id:   The user making the request (from auth).

        Returns:
            An AIQueryResponse with the generated response and metadata.

        Raises:
            ValidationError: If prompt injection is detected (HTTP 422).
            UpstreamError:   If the LLM API fails (HTTP 502).
        """
        logger.info(
            f"Processing AI query: tenant={tenant_id}, "
            f"user={user_id}, use_case={request.use_case}, "
            f"query='{request.query[:80]}...'"
        )

        # --------------------------------------------------------
        # STEP 1: Get the system prompt for this use case
        #
        # The system prompt defines HOW the AI should behave.
        # It's retrieved from the prompts table by use_case key.
        # Falls back to the built-in default if none found in DB.
        # --------------------------------------------------------
        prompt_record = self.prompt_service.get_active_prompt(request.use_case)
        base_system_prompt = prompt_record.content

        logger.debug(
            f"System prompt loaded: '{prompt_record.name}' "
            f"(use_case={request.use_case}, version={prompt_record.version})"
        )

        # --------------------------------------------------------
        # STEP 2: Retrieve relevant document chunks (RAG)
        #
        # Embeds the user's query and finds the most similar
        # chunks from the tenant's documents (and global library).
        # Returns an empty list if no documents have been uploaded.
        # --------------------------------------------------------
        retrieved_chunks = self.retrieval_service.retrieve(
            tenant_id=tenant_id,
            query=request.query,
            top_k=request.top_k,
            include_global=request.include_global,
        )

        logger.debug(f"Retrieved {len(retrieved_chunks)} context chunks")

        # --------------------------------------------------------
        # STEP 3: Apply guardrails
        #
        # - Scan the user query for injection patterns
        # - Append non-negotiable rules to the system prompt
        # Raises ValidationError (HTTP 422) if injection detected.
        # --------------------------------------------------------
        safe_query, hardened_system_prompt = apply_guardrails(
            user_query=request.query,
            system_prompt=base_system_prompt,
        )

        # --------------------------------------------------------
        # STEP 4: Assemble the final prompt
        #
        # Combines everything into one string for the LLM:
        #   [System Instructions + Safety Rules]
        #   [Retrieved Context Chunks]
        #   [User Question]
        #
        # This structure teaches the LLM to answer from context.
        # --------------------------------------------------------
        final_prompt = self._assemble_prompt(
            system_prompt=hardened_system_prompt,
            chunks=retrieved_chunks,
            user_query=safe_query,
            response_mode=request.response_mode,
        )

        logger.debug(f"Final prompt assembled: {len(final_prompt)} chars")

        # --------------------------------------------------------
        # STEP 5: Call the LLM
        #
        # The router reads LLM_DEFAULT_PROVIDER from env and calls
        # the correct client (OpenAI or Gemini).
        # --------------------------------------------------------
        llm_response = self.llm_router.route(
            LLMRequest(
                prompt=final_prompt,
                use_case=request.use_case,
            )
        )

        logger.info(
            f"LLM response: {len(llm_response.text)} chars, "
            f"{llm_response.latency_ms}ms via {llm_response.model_provider}"
        )

        # --------------------------------------------------------
        # STEP 6: Save the trace (audit log)
        #
        # This must happen BEFORE returning to the caller.
        # The trace contains everything: query, context, response.
        # --------------------------------------------------------
        retrieved_chunk_ids = [c.chunk_id for c in retrieved_chunks]
        retrieved_content = "\n\n".join(c.content for c in retrieved_chunks)

        try:
            trace = self.trace_service.create_trace(
                CreateTraceRequest(
                    tenant_id=tenant_id,
                    user_id=user_id,
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
            # Tracing failure is logged but never blocks the AI response
            logger.error(f"Failed to save trace: {e}", exc_info=True)
            trace_id = "trace-error"

        # --------------------------------------------------------
        # STEP 7: Run automatic evaluation (heuristic graders)
        #
        # Checks response quality without making any API calls.
        # Runs synchronously but is very fast (~1ms).
        # --------------------------------------------------------
        try:
            self.eval_service.evaluate_trace(
                trace_id=trace_id,
                response_text=llm_response.text,
                chunk_texts=[c.content for c in retrieved_chunks],
                response_mode=request.response_mode,
            )
        except Exception as e:
            # Eval failure is logged but never blocks the response
            logger.warning(f"Evaluation failed for trace {trace_id}: {e}")

        # --------------------------------------------------------
        # STEP 8: Build and return the response
        # --------------------------------------------------------
        chunk_infos = [
            RetrievedChunkInfo(
                chunk_id=chunk.chunk_id,
                document_id=chunk.document_id,
                similarity=round(chunk.similarity, 4),
                content_preview=chunk.content[:200],  # First 200 chars for preview
            )
            for chunk in retrieved_chunks
        ]

        return AIQueryResponse(
            response_text=llm_response.text,
            trace_id=trace_id,
            model_provider=llm_response.model_provider,
            model_name=llm_response.model_name,
            latency_ms=llm_response.latency_ms,
            use_case=request.use_case,
            retrieved_chunk_count=len(retrieved_chunks),
            retrieved_chunks=chunk_infos,
        )

    def _assemble_prompt(
        self,
        system_prompt: str,
        chunks: list,  # list[ChunkResult]
        user_query: str,
        response_mode: str = "text",
    ) -> str:
        """
        Assemble the final prompt string for the LLM.

        Format:
            [System Instructions]

            Context Documents:
            ==================
            [Chunk 1 text]

            [Chunk 2 text]
            ...
            ==================

            User Question: [query]

        Args:
            system_prompt: The hardened system prompt.
            chunks:        Retrieved document chunks.
            user_query:    The user's question.
            response_mode: "text" or "structured" — adds output format instructions.

        Returns:
            The fully assembled prompt string.
        """
        # Start with the system prompt
        parts = [system_prompt.strip()]

        # Add retrieved context if any chunks were found
        if chunks:
            context_header = "\n\nContext Documents:\n" + "=" * 50
            chunk_texts = "\n\n".join(
                f"[Chunk {i+1} | similarity: {c.similarity:.3f}]\n{c.content}"
                for i, c in enumerate(chunks)
            )
            context_footer = "=" * 50
            parts.append(f"{context_header}\n{chunk_texts}\n{context_footer}")
        else:
            # No documents available — tell the LLM explicitly
            parts.append(
                "\n\nNota: No se encontraron documentos relevantes en la base de conocimiento. "
                "Aplica el GROUNDING CONTRACT: responde solo si puedes hacerlo sin usar "
                "conocimiento de entrenamiento."
            )

        # Add response format instruction for structured mode
        if response_mode == "structured":
            parts.append(
                "\n\nOUTPUT FORMAT: Respond with valid JSON only. No text before or after.\n\n"
                "{\n"
                '  "summary": "<string: 1-2 sentences summarizing the answer>",\n'
                '  "key_points": ["<string>"],\n'
                '  "risks": ["<string>"],\n'
                '  "has_risks": false,\n'
                '  "confidence": "<string: high | medium | low>"\n'
                "}\n\n"
                "Rules:\n"
                '- "risks": always include this key; use [] if no risks are identified.\n'
                '- "has_risks": true if risks array is non-empty, false otherwise.\n'
                '- "confidence": "high" = context directly and completely answers; '
                '"medium" = partially; "low" = requires inference.\n'
                "- No JSON comments. No trailing commas. No markdown code fences.\n"
                '- If unable to produce valid JSON, output: '
                '{"error": "Unable to structure response", "raw": "<plain text answer>"}'
            )

        # Add the user's question
        parts.append(f"\n\nUser Question: {user_query}")

        return "\n".join(parts)
