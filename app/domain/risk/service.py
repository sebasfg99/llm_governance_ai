# ============================================================
# app/domain/risk/service.py — Risk assessment orchestrator (B)
#
# WHY THIS FILE EXISTS:
# The risk assessment feature is the highest-value workflow of
# the platform. It can't be a single LLM call: quality improves
# dramatically when the prompt is fed with the completed profile
# AND normative chunks retrieved specifically for the profile's
# attributes (not just the original question).
#
# This service orchestrates:
#
#   a) run_intake_turn()     — conversational profile-filling via
#                              AIService (use_case='risk_intake').
#                              Returns the next assistant turn and
#                              whether the profile is complete.
#
#   b) run_risk_analysis()   — the structured B3 pipeline:
#                                1. Compose a synthetic query from
#                                   the profile.
#                                2. Retrieve normative chunks
#                                   filtered by the expected
#                                   frameworks for the profile.
#                                3. Call the LLM with the
#                                   risk_analysis prompt.
#                                4. Parse the JSON, validate it
#                                   against RiskRegister.
#                                5. Persist via ProjectService.
#                                6. Return the register + trace_id.
#
# This is the only service that "owns" a feature end-to-end: it
# writes to projects.risk_register and leaves traces tagged with
# project_id so the "why?" queries in AIService can find them.
# ============================================================

import json
import re
from datetime import datetime, timezone
from typing import Optional

from supabase import Client

from app.core.errors import UpstreamError, ValidationError
from app.core.logging import get_logger
from app.domain.ai_service import AIService
from app.domain.llm.guardrails import apply_guardrails
from app.domain.llm.models import LLMRequest
from app.domain.llm.router import LLMRouter
from app.domain.projects.models import Project, ProjectProfile
from app.domain.projects.service import ProjectService
from app.domain.prompts.service import PromptService
from app.domain.retrieval.service import RetrievalService
from app.domain.risk.models import (
    RiskIntakeRequest,
    RiskIntakeResponse,
    RiskRegister,
    RunRiskAnalysisRequest,
    RunRiskAnalysisResponse,
)
from app.domain.traces.models import CreateTraceRequest
from app.domain.traces.service import TraceService
from app.schemas.ai import AIQueryRequest

logger = get_logger(__name__)


# Frameworks that the risk analysis step always tries to include
# in retrieval, in addition to anything the profile implies. Listed
# here (not the prompt) because retrieval filters expect slugs.
_CORE_RISK_FRAMEWORKS = ["iso_42001", "nist_airmf", "eu_ai_act", "gdpr"]
_LLM_ADDITIONAL_FRAMEWORKS = ["owasp_llm_top10", "mitre_atlas"]

# Marker the risk_intake prompt emits when the interview is done.
# We keep the detection forgiving (case-insensitive, accent-optional)
# so a small LLM paraphrase does not break the UI's state machine.
_INTAKE_COMPLETE_MARKER = re.compile(
    r"perfil\s+completo", re.IGNORECASE
)


class RiskService:
    def __init__(self, db: Client) -> None:
        self.db = db
        self.project_service = ProjectService(db)
        self.prompt_service = PromptService(db)
        self.retrieval_service = RetrievalService(db)
        self.trace_service = TraceService(db)
        self.ai_service = AIService(db)
        self.llm_router = LLMRouter()

    # ================================================================
    # B2 — Conversational intake
    # ================================================================
    def run_intake_turn(
        self,
        project_id: str,
        tenant_id: str,
        user_id: str,
        request: RiskIntakeRequest,
    ) -> RiskIntakeResponse:
        """
        Run one turn of the profile-filling interview.

        The heavy lifting (memory hydration, trace, thread persistence)
        is delegated to AIService. We only interpret the response to
        decide whether the interview reached the "PERFIL COMPLETO"
        marker so the UI can switch to the analysis screen.
        """
        # Make sure the project exists and belongs to the tenant before
        # spending LLM tokens on an orphan conversation.
        project = self.project_service.get(project_id, tenant_id)

        ai_response = self.ai_service.query(
            request=AIQueryRequest(
                query=request.message,
                use_case="risk_intake",
                project_id=str(project.id),
                thread_id=request.thread_id,
                response_mode="text",
                # Intake is a conversation about the project, not a RAG-
                # grounded Q&A. We keep RAG off by asking for zero chunks
                # while still going through the unified pipeline so the
                # thread and trace machinery works the same way.
                top_k=1,
                include_global=False,
            ),
            tenant_id=tenant_id,
            user_id=user_id,
        )

        is_complete = bool(_INTAKE_COMPLETE_MARKER.search(ai_response.response_text))

        return RiskIntakeResponse(
            project_id=project_id,
            thread_id=ai_response.thread_id or "",
            assistant_message=ai_response.response_text,
            is_complete=is_complete,
            trace_id=ai_response.trace_id,
        )

    # ================================================================
    # B3 — Risk analysis orchestrator
    # ================================================================
    def run_risk_analysis(
        self,
        project_id: str,
        tenant_id: str,
        user_id: str,
        request: RunRiskAnalysisRequest,
    ) -> RunRiskAnalysisResponse:
        """
        Full orchestration of the risk register generation.

        Flow:
          1. Load project + profile (or use override).
          2. Derive retrieval frameworks from the profile.
          3. Compose a synthetic query describing the project.
          4. Run retrieval (tenant + global, framework-scoped).
          5. Call the LLM with the risk_analysis system prompt.
          6. Parse + validate the JSON into a RiskRegister.
          7. Persist the register on the project.
          8. Write a trace tagged with project_id.
        """
        project = self.project_service.get(project_id, tenant_id)
        profile = request.profile_override or project.project_profile
        country = request.country or profile.country

        self._require_profile_ready(profile)

        frameworks = self._frameworks_for_profile(profile)
        synthetic_query = self._profile_to_query(profile)

        logger.info(
            f"Risk analysis start: project={project_id}, frameworks={frameworks}, "
            f"country={country}"
        )

        # 4. Retrieval — scoped to the expected normative frameworks.
        chunks = self.retrieval_service.retrieve(
            tenant_id=tenant_id,
            query=synthetic_query,
            top_k=15,  # higher than chat so the LLM can cross-reference
            include_global=True,
            country=country,
            framework_filter=frameworks,
        )
        logger.info(f"Risk analysis retrieval: {len(chunks)} chunks")

        # 5. Compose and call the LLM.
        prompt_record = self.prompt_service.get_active_prompt("risk_analysis")
        _, hardened_system_prompt = apply_guardrails(
            user_query=synthetic_query,  # used for injection scanning only
            system_prompt=prompt_record.content,
        )

        final_prompt = self._assemble_risk_prompt(
            system_prompt=hardened_system_prompt,
            profile=profile,
            chunks=chunks,
            country=country,
        )

        llm_response = self.llm_router.route(
            LLMRequest(prompt=final_prompt, use_case="risk_analysis")
        )

        # 6. Parse JSON → RiskRegister.
        register = self._parse_register(llm_response.text)

        # 7. Persist on the project (updates status to risks_analyzed).
        self.project_service.set_risk_register(
            project_id=project_id,
            tenant_id=tenant_id,
            register=register.model_dump(),
        )

        # 8. Write the trace manually (we skipped AIService because it
        #    does not expose a structured risk pipeline entry point —
        #    doing it inline here avoids leaking risk-specific logic
        #    into AIService).
        trace_id: Optional[str] = None
        try:
            trace = self.trace_service.create_trace(
                CreateTraceRequest(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    project_id=project_id,
                    thread_id=None,
                    user_query=synthetic_query,
                    prompt_text=final_prompt,
                    use_case="risk_analysis",
                    model_provider=llm_response.model_provider,
                    model_name=llm_response.model_name,
                    retrieved_chunk_ids=[c.chunk_id for c in chunks],
                    retrieved_content="\n\n".join(c.content for c in chunks),
                    response_text=llm_response.text,
                    response_mode="structured",
                    latency_ms=llm_response.latency_ms,
                )
            )
            trace_id = str(trace.id)
        except Exception as e:
            logger.error(f"Failed to save risk analysis trace: {e}", exc_info=True)

        return RunRiskAnalysisResponse(
            project_id=project_id,
            risk_register=register,
            trace_id=trace_id,
        )

    # ================================================================
    # Helpers
    # ================================================================
    @staticmethod
    def _require_profile_ready(profile: ProjectProfile) -> None:
        """
        Guard against running the expensive analysis on an empty profile.

        We require the minimum signals needed for the LLM to tier and
        categorise: ai_type, deployment, data_used OR affected_audience.
        Anything less produces garbage that costs tokens.
        """
        missing: list[str] = []
        if not profile.ai_type:
            missing.append("ai_type")
        if not profile.deployment:
            missing.append("deployment")
        if not profile.data_used and not profile.affected_audience:
            missing.append("data_used or affected_audience")

        if missing:
            raise ValidationError(
                "Project profile is incomplete. Run the risk intake first.",
                detail={"missing_fields": missing},
            )

    @staticmethod
    def _frameworks_for_profile(profile: ProjectProfile) -> list[str]:
        """Core frameworks + LLM-specific ones when applicable."""
        frameworks = list(_CORE_RISK_FRAMEWORKS)
        if profile.ai_type == "llm":
            frameworks += _LLM_ADDITIONAL_FRAMEWORKS
        return frameworks

    @staticmethod
    def _profile_to_query(profile: ProjectProfile) -> str:
        """
        Translate the structured profile into a retrieval query.

        The query is not shown to the user — it is what the embedding
        model sees. We include enough salient words (ai_type, data,
        industry, deployment) so RAG pulls relevant normative content.
        """
        pieces = ["Análisis de riesgos para un sistema de IA"]
        if profile.ai_type:
            pieces.append(f"tipo {profile.ai_type}")
        if profile.industry:
            pieces.append(f"en industria {profile.industry}")
        if profile.deployment:
            pieces.append(f"despliegue {profile.deployment}")
        if profile.affected_audience:
            pieces.append(f"audiencia {profile.affected_audience}")
        if profile.data_used:
            pieces.append("datos: " + ", ".join(profile.data_used))
        if profile.autonomy_level:
            pieces.append(f"autonomía {profile.autonomy_level}")
        if profile.country:
            pieces.append(f"país {profile.country}")

        return " ".join(pieces)

    @staticmethod
    def _assemble_risk_prompt(
        system_prompt: str,
        profile: ProjectProfile,
        chunks: list,
        country: str | None,
    ) -> str:
        profile_block = (
            "\n\n═══ PERFIL DEL PROYECTO ═══\n"
            + json.dumps(profile.model_dump(), ensure_ascii=False, indent=2)
            + (f"\nPaís activo para normativa local: {country}" if country else "")
        )

        if chunks:
            chunk_texts = "\n\n".join(
                f"[Chunk {i+1} | similarity {c.similarity:.3f}]\n{c.content}"
                for i, c in enumerate(chunks)
            )
            context_block = (
                "\n\n═══ DOCUMENTOS NORMATIVOS RECUPERADOS ═══\n"
                + chunk_texts
            )
        else:
            context_block = (
                "\n\n═══ DOCUMENTOS NORMATIVOS RECUPERADOS ═══\n"
                "(ninguno encontrado — aplica GROUNDING CONTRACT con supuestos explícitos)"
            )

        instruction = (
            "\n\nProduce el risk register JSON según el esquema definido arriba. "
            "No agregues texto fuera del JSON."
        )
        return system_prompt.strip() + profile_block + context_block + instruction

    @staticmethod
    def _parse_register(raw_text: str) -> RiskRegister:
        """
        LLMs occasionally wrap JSON in ```json fences or add a short
        preamble despite instructions. We strip both before parsing.
        If the JSON is irreparable we raise UpstreamError so the caller
        sees a clear 502 instead of a silent Pydantic dump.
        """
        cleaned = raw_text.strip()

        # Strip ```json ... ``` fences if present
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
            cleaned = re.sub(r"\s*```$", "", cleaned)
            cleaned = cleaned.strip()

        # Extract the first top-level JSON object. A crude brace balance
        # is more robust than trusting the LLM to output exactly the
        # object with nothing else.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise UpstreamError(
                "Risk analysis LLM did not return JSON.",
                detail={"raw_text_preview": raw_text[:500]},
            )
        json_blob = cleaned[start : end + 1]

        try:
            data = json.loads(json_blob)
        except json.JSONDecodeError as e:
            raise UpstreamError(
                f"Risk analysis JSON parse failed: {e}",
                detail={"raw_text_preview": raw_text[:500]},
            )

        # Recompute inherent_score defensively — LLMs often disagree
        # with their own multiplication.
        for risk in data.get("risks", []) or []:
            try:
                likelihood = int(risk.get("likelihood") or 0)
                impact = int(risk.get("impact") or 0)
                risk["inherent_score"] = likelihood * impact
            except (TypeError, ValueError):
                risk["inherent_score"] = 0

        try:
            return RiskRegister(**data)
        except Exception as e:
            raise UpstreamError(
                f"Risk analysis JSON did not match RiskRegister schema: {e}",
                detail={"raw_text_preview": raw_text[:500]},
            )
