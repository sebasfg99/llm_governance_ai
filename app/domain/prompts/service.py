# ============================================================
# app/domain/prompts/service.py — Prompt library service
#
# WHY THIS FILE EXISTS:
# The prompt library is what makes the AI engine configurable
# without code changes. Instead of hardcoding system prompts
# in the code, they live in the database and can be updated,
# versioned, and rolled back independently of deployments.
#
# USE CASES FOR THE PROMPT LIBRARY:
#   - "default_rag": General document Q&A
#   - "document_summary": Summarize a document
#   - "risk_analysis": Identify risks in a contract/policy
#   - "response_draft": Draft a response based on context
#
# VERSIONING:
# When you improve a prompt, create a new row with version+1
# and is_active=True. The old version stays in the DB for
# audit/rollback purposes. The service always returns the
# active version.
# ============================================================

from supabase import Client

from app.core.logging import get_logger
from app.core.errors import NotFoundError
from app.domain.prompts.models import Prompt, CreatePromptRequest

logger = get_logger(__name__)

# ============================================================
# Default fallback prompt
#
# Used when no prompt exists in the DB for a given use_case.
# This prevents the system from failing on a missing prompt
# during development or first setup.
# ============================================================
_CHATBOT_PROMPT = """\
Eres el Agente GovAI — copiloto de gobernanza de IA integrado en la plataforma.

━━━ ROL ━━━
Eres simultáneamente:
• Un asistente de datos: respondés preguntas sobre el estado real de la institución y sus proyectos.
• Un guía de procesos: acompañás al usuario en cada paso del ciclo de gobernanza.
• Un experto normativo: explicás ISO 42001, NIST AI RMF, EU AI Act, SUGEF, CONASSIF y la normativa aplicable.

━━━ FUENTES DE INFORMACIÓN (en orden de prioridad) ━━━

1. BLOQUES DE CONTEXTO (═══): Datos reales y autoritativos de la base de datos.
   Los bloques marcados con ═══ contienen información exacta y actualizada.
   Para preguntas sobre proyectos, riesgos, fases, pasos, aprobaciones, condiciones,
   proyectos gobernados o cualquier estado de la plataforma: usá EXCLUSIVAMENTE estos bloques.
   NUNCA digas "no tengo acceso a esa información" si el dato aparece en un bloque ═══.
   Respondé con los números y estados exactos del contexto, no con aproximaciones.

2. DOCUMENTOS RECUPERADOS: Para preguntas normativas y regulatorias.
   Citá la fuente cuando usés un documento: [Fuente: nombre_documento]
   Si el documento no cubre la pregunta completamente, indicalo y usá tu conocimiento
   de entrenamiento como complemento, distinguiendo claramente entre ambas fuentes.

3. CONOCIMIENTO DE ENTRENAMIENTO: Solo como último recurso para preguntas normativas generales.
   Cuando lo usés, aclaralo explícitamente: "Según mi conocimiento general..."

━━━ COMPORTAMIENTO DE COPILOTO ━━━

PROACTIVO: Cuando respondés sobre el estado actual, siempre añadí el próximo paso más urgente.
CONCISO: Máximo 3-4 párrafos o una lista de 6 ítems. Sin introducciones largas ni repetir la pregunta.
ACCIONABLE: Terminá con una acción concreta que el usuario puede hacer ahora mismo.
ROL-CONSCIENTE: Adaptá el enfoque según el rol indicado en el bloque de contexto:
  • compliance_officer → visión ejecutiva, aprobaciones pendientes, condiciones por vencer, estado de cumplimiento.
  • project_leader → pasos pendientes del proyecto, deliverables incompletos, evidencias faltantes.
  • technical_leader → aspectos técnicos, controles ISO 42001, mitigación de riesgos.
  • executive → métricas de alto nivel, proyectos en riesgo, estado global de la institución.

Respondé siempre en español, a menos que el usuario escriba en otro idioma.\
"""

_EXECUTIVE_SUMMARY_PROMPT = """\
Eres el Agente GovAI especializado en Gate Reviews y resúmenes ejecutivos de gobernanza.

Usá el bloque ═══ CONTEXTO DEL PROYECTO ═══ para preparar al usuario para el Gate Review:
• Estado de la fase actual: pasos completados vs. pendientes (con nombres específicos).
• Condiciones del gate anterior sin resolver.
• Top 3 riesgos más críticos y si tienen plan de mitigación definido.
• Evidencias: qué pasos tienen evidencias faltantes.
• Diagnóstico final: ¿está el proyecto listo para el Gate Review?

Formato de respuesta OBLIGATORIO:
✅ Completado: [lista específica]
⏳ Pendiente: [lista específica con paso número y nombre]
⚠️ Alertas: [condiciones abiertas, riesgos críticos, evidencias faltantes]
📋 Diagnóstico: listo para gate / necesita [X] antes del gate

Sé directo. Máximo 6 ítems por sección. Respondé en español.\
"""

_RISK_ANALYSIS_PROMPT = """\
Eres el Agente GovAI especializado en análisis de riesgos de IA bajo MITRE ATLAS, OWASP LLM Top 10 e ISO 42001.

Cuando el usuario pregunta sobre riesgos:
1. Usá el bloque ═══ CONTEXTO DEL PROYECTO ═══ para mostrar el registro real de riesgos.
   Los campos "Registro de riesgos" contienen los datos exactos — nunca inventes riesgos no listados.
2. Para cada riesgo sin mitigar: proponé una estrategia concreta y el control ISO 42001 más relevante.
3. Usá los "Documentos recuperados" para fundamentar las recomendaciones normativas.

Presentá la información así (ordenado por severidad descendente):
• [Amenaza] — Severidad: X | Control ISO: Y | Estado: mitigado/pendiente
• Recomendación de siguiente paso para los 3 riesgos más críticos

Sé preciso y accionable. Respondé en español.\
"""

# Generic fallback — used for use_cases without a specific prompt
# and when the DB has no active prompt for that use_case.
DEFAULT_SYSTEM_PROMPT = _CHATBOT_PROMPT

# Per-use-case defaults (checked before the generic fallback)
_USE_CASE_PROMPTS: dict[str, str] = {
    "chatbot": _CHATBOT_PROMPT,
    "executive_summary": _EXECUTIVE_SUMMARY_PROMPT,
    "risk_analysis": _RISK_ANALYSIS_PROMPT,
}


class PromptService:
    """
    Manages the versioned system prompt library.

    Prompts are stored in the `prompts` table and retrieved
    by use_case. The active (latest) version is always returned.

    Usage:
        service = PromptService(db)
        prompt = service.get_active_prompt("default_rag")
        print(prompt.content)
    """

    def __init__(self, db: Client) -> None:
        self.db = db

    def get_active_prompt(self, use_case: str) -> Prompt:
        """
        Retrieve the active prompt for a given use case.

        Queries the prompts table for the row where
        use_case matches AND is_active = true,
        ordered by version DESC (newest wins if multiple active).

        Args:
            use_case: The use case key (e.g., "default_rag").

        Returns:
            The active Prompt record.
            Falls back to DEFAULT_SYSTEM_PROMPT if none found.
        """
        logger.debug(f"Fetching active prompt for use_case='{use_case}'")

        result = (
            self.db.table("prompts")
            .select("*")
            .eq("use_case", use_case)
            .eq("is_active", True)
            .order("version", desc=True)
            .limit(1)
            .execute()
        )

        if not result.data:
            # Use a per-use-case copilot prompt if available, otherwise the generic default.
            fallback_content = _USE_CASE_PROMPTS.get(use_case, DEFAULT_SYSTEM_PROMPT)
            logger.warning(
                f"No active prompt found for use_case='{use_case}'. "
                f"Using built-in {'use-case-specific' if use_case in _USE_CASE_PROMPTS else 'generic'} fallback."
            )
            from uuid import uuid4
            from datetime import datetime, timezone
            return Prompt(
                id=uuid4(),
                name=f"Built-in copilot ({use_case})",
                version=0,
                use_case=use_case,
                content=fallback_content,
                is_active=True,
                created_at=datetime.now(timezone.utc),
            )

        return Prompt(**result.data[0])

    def list_prompts(self) -> list[Prompt]:
        """
        Return all prompts in the library (all use cases, all versions).

        Returns:
            List of Prompt records ordered by use_case and version.
        """
        result = (
            self.db.table("prompts")
            .select("*")
            .order("use_case")
            .order("version", desc=True)
            .execute()
        )

        return [Prompt(**row) for row in result.data]

    def create_prompt(self, request: CreatePromptRequest) -> Prompt:
        """
        Create a new prompt in the library.

        If is_active=True, deactivate any existing active prompt
        for the same use_case first (only one active at a time).

        Args:
            request: The prompt data to insert.

        Returns:
            The newly created Prompt record.
        """
        # If this new prompt will be active, deactivate the current one
        if request.is_active:
            self.db.table("prompts").update(
                {"is_active": False}
            ).eq("use_case", request.use_case).eq("is_active", True).execute()

        # Insert the new prompt
        from datetime import datetime, timezone
        result = self.db.table("prompts").insert({
            "name": request.name,
            "version": request.version,
            "use_case": request.use_case,
            "content": request.content,
            "is_active": request.is_active,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()

        if not result.data:
            raise RuntimeError("Failed to create prompt")

        logger.info(
            f"Prompt created: name='{request.name}', "
            f"use_case='{request.use_case}', version={request.version}"
        )

        return Prompt(**result.data[0])
