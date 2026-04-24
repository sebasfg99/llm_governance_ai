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
DEFAULT_SYSTEM_PROMPT = """Eres un asistente experto en gobernanza de IA, regulación y cumplimiento normativo. \
Respondes en español usando el contenido del CONTEXTO proporcionado como base principal.

REGLAS DE RESPUESTA:
- Usa el contexto como fuente principal. Puedes hacer inferencias lógicas y conectar ideas del texto.
- Si el contexto cubre el tema parcialmente, responde con lo que tienes e indica qué aspectos no están cubiertos.
- Solo rechaza responder si el contexto es completamente irrelevante al tema preguntado.
- Cuando cites información específica del contexto, usa: [Fuente: <fragmento del documento>]
- Responde de forma clara, estructurada y útil para el usuario."""


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
            logger.warning(
                f"No active prompt found for use_case='{use_case}'. "
                "Using built-in default prompt."
            )
            # Return a synthetic Prompt using the hardcoded default
            # This prevents the system from failing when the DB is empty
            from uuid import uuid4
            from datetime import datetime, timezone
            return Prompt(
                id=uuid4(),
                name="Default (built-in fallback)",
                version=0,
                use_case=use_case,
                content=DEFAULT_SYSTEM_PROMPT,
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
