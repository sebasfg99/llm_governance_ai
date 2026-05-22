# ============================================================
# app/domain/policies/service.py — Policy + controls orchestrators
#
# WHY TWO ENTRY POINTS IN ONE SERVICE:
# Policies (C1) and controls (C2) share the same helpers:
#   - they both assemble a prompt from a project
#   - they both run RAG scoped to policy-relevant frameworks
#   - they both parse a structured JSON answer
#   - they both persist the result on the project row
# Keeping them in one class avoids duplicating those helpers.
#
# The heavy logic — RAG + LLM + JSON parsing — mirrors
# RiskService closely on purpose: once we see how it behaves in
# production we can extract the common skeleton into a reusable
# "StructuredOutputOrchestrator". For v1 duplication is fine.
# ============================================================

import json
import re
from typing import Optional

from supabase import Client

from app.core.errors import UpstreamError, ValidationError
from app.core.logging import get_logger
from app.domain.llm.guardrails import apply_guardrails
from app.domain.llm.models import LLMRequest
from app.domain.llm.router import LLMRouter
from app.domain.policies.models import (
    ControlsBundle,
    GenerateControlsRequest,
    GenerateControlsResponse,
    GeneratePolicyRequest,
    GeneratePolicyResponse,
    Policy,
)
from app.domain.projects.service import ProjectService
from app.domain.prompts.service import PromptService
from app.domain.retrieval.service import RetrievalService
from app.domain.traces.models import CreateTraceRequest
from app.domain.traces.service import TraceService

logger = get_logger(__name__)


# Policies always want ISO 42001 + NIST. EU AI Act is added when the
# project affects EU subjects (detected heuristically from country).
_POLICY_CORE_FRAMEWORKS = ["iso_42001", "nist_airmf"]


class PoliciesService:
    def __init__(self, db: Client) -> None:
        self.db = db
        self.project_service = ProjectService(db)
        self.prompt_service = PromptService(db)
        self.retrieval_service = RetrievalService(db)
        self.trace_service = TraceService(db)
        self.llm_router = LLMRouter()

    # ================================================================
    # C1 — Policy generation
    # ================================================================
    def generate_policy(
        self,
        tenant_id: str,
        user_id: str,
        project_id: str | None,
        request: GeneratePolicyRequest,
    ) -> GeneratePolicyResponse:
        """
        Produce an ISO 42001-aligned AIMS policy.

        project_id is optional: in "discover mode" a user can ask for
        a policy without a project (the company doesn't have one yet
        and wants a template). When provided, we personalize the
        policy with the project profile and risk_register.
        """
        project = None
        if project_id:
            project = self.project_service.get(project_id, tenant_id)

        country = request.country or (
            project.project_profile.country if project else None
        )

        frameworks = list(_POLICY_CORE_FRAMEWORKS)
        if country in {
            "espana", "españa", "alemania", "francia", "italia",
            "portugal", "holanda", "belgica", "bélgica",
        }:
            frameworks.append("eu_ai_act")

        # Synthetic query: capture the salient terms so retrieval
        # pulls the right chapters of ISO 42001, etc.
        query_bits = ["Política organizacional de gestión de IA ISO 42001 NIST AI RMF"]
        if project is not None:
            profile = project.project_profile
            if profile.industry:
                query_bits.append(f"industria {profile.industry}")
            if profile.ai_type:
                query_bits.append(f"tipo {profile.ai_type}")
            if profile.deployment:
                query_bits.append(f"despliegue {profile.deployment}")
        if country:
            query_bits.append(f"país {country}")
        synthetic_query = " ".join(query_bits)

        chunks = self.retrieval_service.retrieve(
            tenant_id=tenant_id,
            query=synthetic_query,
            top_k=15,
            include_global=True,
            country=country,
            framework_filter=frameworks,
        )

        prompt_record = self.prompt_service.get_active_prompt("policy_generation")
        _, hardened = apply_guardrails(
            user_query=synthetic_query, system_prompt=prompt_record.content
        )

        final_prompt = self._assemble_policy_prompt(
            system_prompt=hardened,
            project=project,
            chunks=chunks,
            country=country,
            include_project_context=request.include_project_context,
        )

        llm_response = self.llm_router.route(
            LLMRequest(prompt=final_prompt, use_case="policy_generation")
        )

        policy = _parse_json_into(llm_response.text, Policy, "policy")

        if project is not None:
            self.project_service.set_policy(
                project_id=str(project.id),
                tenant_id=tenant_id,
                policy=policy.model_dump(),
            )

        trace_id = self._persist_trace(
            tenant_id=tenant_id,
            user_id=user_id,
            project_id=str(project.id) if project else None,
            synthetic_query=synthetic_query,
            final_prompt=final_prompt,
            chunks=chunks,
            llm_response=llm_response,
            use_case="policy_generation",
        )

        return GeneratePolicyResponse(
            project_id=str(project.id) if project else None,
            policy=policy,
            trace_id=trace_id,
        )

    # ================================================================
    # C2 — Controls generation
    # ================================================================
    def generate_controls(
        self,
        tenant_id: str,
        user_id: str,
        project_id: str,
        request: GenerateControlsRequest,
    ) -> GenerateControlsResponse:
        """
        Produce a controls bundle for the project's risk register.

        Requires an existing risk_register unless the caller passes
        risk_register_override (e.g. a what-if from the UI).
        """
        project = self.project_service.get(project_id, tenant_id)
        register = request.risk_register_override or project.risk_register
        if not register:
            raise ValidationError(
                "Project has no risk_register. Run risk analysis first.",
                detail={"project_id": project_id},
            )

        # Retrieval for controls is lighter than for policy — we
        # mostly need ISO 42001 control clauses and NIST MAP/MANAGE.
        synthetic_query = "Controles ISO 42001 NIST AI RMF mitigación de riesgos de IA"
        chunks = self.retrieval_service.retrieve(
            tenant_id=tenant_id,
            query=synthetic_query,
            top_k=10,
            include_global=True,
            country=project.project_profile.country,
            framework_filter=_POLICY_CORE_FRAMEWORKS,
        )

        prompt_record = self.prompt_service.get_active_prompt("controls_generation")
        _, hardened = apply_guardrails(
            user_query=synthetic_query, system_prompt=prompt_record.content
        )

        final_prompt = self._assemble_controls_prompt(
            system_prompt=hardened,
            project=project,
            risk_register=register,
            chunks=chunks,
        )

        llm_response = self.llm_router.route(
            LLMRequest(prompt=final_prompt, use_case="controls_generation")
        )

        bundle = _parse_json_into(llm_response.text, ControlsBundle, "controls")

        self.project_service.set_controls(
            project_id=project_id,
            tenant_id=tenant_id,
            controls=bundle.model_dump(),
        )

        trace_id = self._persist_trace(
            tenant_id=tenant_id,
            user_id=user_id,
            project_id=project_id,
            synthetic_query=synthetic_query,
            final_prompt=final_prompt,
            chunks=chunks,
            llm_response=llm_response,
            use_case="controls_generation",
        )

        return GenerateControlsResponse(
            project_id=project_id, controls=bundle, trace_id=trace_id
        )

    # ================================================================
    # Prompt assembly helpers
    # ================================================================
    @staticmethod
    def _assemble_policy_prompt(
        system_prompt: str,
        project,
        chunks: list,
        country: str | None,
        include_project_context: bool,
    ) -> str:
        blocks: list[str] = [system_prompt.strip()]

        if project is not None and include_project_context:
            profile_json = json.dumps(
                project.project_profile.model_dump(),
                ensure_ascii=False,
                indent=2,
            )
            block = (
                "\n\n═══ CONTEXTO DEL PROYECTO ═══\n"
                f"Estado: {project.status}\n"
                f"Perfil:\n{profile_json}"
            )
            if isinstance(project.risk_register, dict):
                risks = project.risk_register.get("risks") or []
                top = sorted(
                    risks,
                    key=lambda r: (r.get("inherent_score") or 0),
                    reverse=True,
                )[:5]
                block += "\nTop riesgos:\n" + "\n".join(
                    f"- {r.get('id')}: {r.get('description', '')[:200]}"
                    for r in top
                )
            blocks.append(block)

        if country:
            blocks.append(f"\n\nPaís de referencia: {country}")

        if chunks:
            chunk_texts = "\n\n".join(
                f"[Chunk {i+1} | similarity {c.similarity:.3f}]\n{c.content}"
                for i, c in enumerate(chunks)
            )
            blocks.append(
                "\n\n═══ DOCUMENTOS NORMATIVOS RECUPERADOS ═══\n" + chunk_texts
            )

        blocks.append(
            "\n\nProduce el JSON de política según el esquema. No agregues texto fuera del JSON."
        )
        return "\n".join(blocks)

    @staticmethod
    def _assemble_controls_prompt(
        system_prompt: str, project, risk_register: dict, chunks: list
    ) -> str:
        register_json = json.dumps(risk_register, ensure_ascii=False, indent=2)
        blocks = [
            system_prompt.strip(),
            "\n\n═══ CONTEXTO DEL PROYECTO ═══\n"
            + json.dumps(project.project_profile.model_dump(), ensure_ascii=False, indent=2),
            "\n\n═══ RISK REGISTER ═══\n" + register_json,
        ]

        if chunks:
            chunk_texts = "\n\n".join(
                f"[Chunk {i+1} | similarity {c.similarity:.3f}]\n{c.content}"
                for i, c in enumerate(chunks)
            )
            blocks.append(
                "\n\n═══ DOCUMENTOS NORMATIVOS RECUPERADOS ═══\n" + chunk_texts
            )

        blocks.append(
            "\n\nProduce el JSON de controles según el esquema. No agregues texto fuera del JSON."
        )
        return "\n".join(blocks)

    def _persist_trace(
        self,
        tenant_id: str,
        user_id: str,
        project_id: str | None,
        synthetic_query: str,
        final_prompt: str,
        chunks: list,
        llm_response,
        use_case: str,
    ) -> Optional[str]:
        try:
            trace = self.trace_service.create_trace(
                CreateTraceRequest(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    project_id=project_id,
                    thread_id=None,
                    user_query=synthetic_query,
                    prompt_text=final_prompt,
                    use_case=use_case,
                    model_provider=llm_response.model_provider,
                    model_name=llm_response.model_name,
                    retrieved_chunk_ids=[c.chunk_id for c in chunks],
                    retrieved_content="\n\n".join(c.content for c in chunks),
                    response_text=llm_response.text,
                    response_mode="structured",
                    latency_ms=llm_response.latency_ms,
                    risk_level="medium",
                )
            )
            return str(trace.id)
        except Exception as e:
            logger.error(
                f"Failed to save {use_case} trace: {e}", exc_info=True
            )
            return None


# ============================================================
# JSON parsing helper shared with future structured orchestrators
# ============================================================
def _parse_json_into(raw_text: str, schema, label: str):
    cleaned = raw_text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
        cleaned = cleaned.strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise UpstreamError(
            f"{label} LLM did not return JSON.",
            detail={"raw_text_preview": raw_text[:500]},
        )

    try:
        data = json.loads(cleaned[start : end + 1])
    except json.JSONDecodeError as e:
        raise UpstreamError(
            f"{label} JSON parse failed: {e}",
            detail={"raw_text_preview": raw_text[:500]},
        )

    try:
        return schema(**data)
    except Exception as e:
        raise UpstreamError(
            f"{label} JSON did not match schema: {e}",
            detail={"raw_text_preview": raw_text[:500]},
        )
