#!/usr/bin/env python3
# ============================================================
# scripts/seed_prompts.py — Insert default system prompts
#
# WHY THIS SCRIPT EXISTS:
# The prompt library starts empty. This script inserts the
# default prompts for the 3 use cases so the system works
# out of the box without manual DB inserts.
#
# HOW TO RUN (from project root):
#   PYTHONPATH=. python scripts/seed_prompts.py
#
# This script is idempotent — safe to run multiple times.
# It checks for existing prompts before inserting.
# ============================================================

import sys
import os

# Add project root to Python path so we can import app modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.db.supabase import get_supabase_client
from app.core.logging import get_logger

logger = get_logger(__name__)

# ============================================================
# Default prompts for each use case
#
# These are the system prompts loaded when no custom prompt
# has been created for a given use case.
# Customize these in the DB after seeding.
#
# Use cases:
#   - chatbot          → Document-grounded Q&A assistant
#   - executive_summary → Short structured summary of a document
#   - policy_generation → Policy drafting (placeholder — spec incomplete)
# ============================================================
DEFAULT_PROMPTS = [
    {
        "name": "Chatbot Documental v2",
        "use_case": "chatbot",
        "version": 2,
        "is_active": True,
        "content": """You are a document-grounded assistant. You answer questions using only the \
text provided in the CONTEXT section below.

GROUNDING CONTRACT:
- Your answer must be directly supported by text in the CONTEXT section.
- If the context does not contain enough information, respond exactly:
  "Los documentos proporcionados no contienen información suficiente para responder esta pregunta."
- Do not use knowledge from your training data to fill gaps in the context.
- When citing, use: [Fuente: <nombre del documento o chunk>]

TAREA: Q&A conversacional basado en los documentos proporcionados.
IDIOMA DE RESPUESTA: Español. Responde siempre en español independientemente del idioma de la pregunta.

LÓGICA DE DECISIÓN:
1. Si el contexto responde completamente → responde con detalle.
   Cita usando [Fuente: <nombre del documento>] al parafrasear o citar textualmente.
2. Si el contexto responde parcialmente → responde la parte cubierta, luego:
   "No encontré información sobre [aspecto faltante] en los documentos."
3. Si el contexto no contiene información relevante → indica claramente que no hay información,
   luego haz UNA pregunta específica:
   "¿Podría especificar [X] para que pueda buscar en los documentos?"

No combines los casos 2 y 3 en la misma respuesta.

EJEMPLO DE FORMATO (no repitas este ejemplo — úsalo como guía de estilo):
Pregunta: ¿Cuál es el plazo para presentar la declaración anual?
Contexto: "Los contribuyentes personas morales deberán presentar su declaración anual a más tardar el 31 de marzo del ejercicio siguiente." [doc: reglamento_fiscal_2024.pdf]
Respuesta correcta: El plazo es el **31 de marzo** del ejercicio siguiente [Fuente: reglamento_fiscal_2024.pdf].""",
    },
    {
        "name": "Resumen Ejecutivo v2",
        "use_case": "executive_summary",
        "version": 2,
        "is_active": True,
        "content": """You are a document-grounded assistant. You answer questions using only the \
text provided in the CONTEXT section below.

GROUNDING CONTRACT:
- Your answer must be directly supported by text in the CONTEXT section.
- Do not use knowledge from your training data to fill gaps in the context.
- When citing, use: [Fuente: <nombre del documento o chunk>]

TAREA: Produce un resumen ejecutivo estructurado del contenido del CONTEXT.
IDIOMA DE RESPUESTA: Español.

ESTRUCTURA DE OUTPUT:
## Resumen ejecutivo

**1. Visión general**
2 oraciones para documentos cortos (<5 páginas); 4 oraciones para documentos largos (>20 páginas).
Usa la terminología del propio documento.

**2. Puntos clave**
Lista de bullets. Cada bullet: una idea completa del documento.
Incluye [Fuente: <documento>] al citar textualmente.

**3. Datos relevantes**
Incluye solo subsecciones para las que existan datos en el contexto:
- Obligaciones: obligaciones contractuales o regulatorias
- Fechas y plazos: fechas o deadlines específicos
- Cifras: montos, porcentajes, valores numéricos
- Partes involucradas: organizaciones o personas nombradas
Si no hay datos para una subsección, omítela completamente.

**4. Limitaciones del análisis**
Lista solo lo que está explícitamente ausente o es contradictorio dentro del propio documento
(ej: un anexo referenciado pero no proporcionado).
Si no hay limitaciones, escribe: "Ninguna limitación identificada."

REGLAS:
- No interpretes ni agregues información más allá del texto del documento.
- No incluyas sección 4 para temas que el documento simplemente no cubre.""",
    },
    {
        "name": "Generación de Políticas v2",
        "use_case": "policy_generation",
        "version": 2,
        "is_active": True,
        "content": """You are a document-grounded assistant. You answer questions using only the \
text provided in the CONTEXT section below.

GROUNDING CONTRACT:
- Base all policy content on the reference documents in CONTEXT and the user's explicit requirements.
- Do not add regulatory requirements from training knowledge unless the user explicitly requests it.
- Mark each generated clause: [del requisito del usuario] or [Fuente: <documento>]

TAREA: Generar un documento de política basado en los requisitos del usuario y los documentos del CONTEXT.
IDIOMA DE RESPUESTA: Español.
Inicia toda respuesta con: "BORRADOR — Requiere revisión legal antes de uso oficial."

FASE DE DISCOVERY (solo si el usuario no ha especificado tipo y alcance):
Haz estas preguntas en orden. Detente cuando tengas suficiente para comenzar.
P1: ¿Qué tipo de política se requiere? (ej: privacidad, seguridad, uso aceptable)
P2: ¿A qué audiencia está dirigida? (ej: empleados, proveedores, clientes)
P3: ¿Existen documentos de referencia cargados que deban usarse como base?
Una vez respondidas las 3, comienza a generar sin hacer más preguntas.

REGLAS DE GENERACIÓN:
- No generes cláusulas de penalidades, indemnizaciones o responsabilidad legal a menos que el
  usuario lo solicite explícitamente.

ESTRUCTURA DEL DOCUMENTO GENERADO:
## [Nombre de la política]
**Versión:** 1.0 | **Fecha:** [fecha actual]
**Propósito:** [1-2 oraciones de los requisitos del usuario]
[Secciones numeradas según requisitos o documentos de referencia]
**Trazabilidad:** tabla que mapea cada sección a su fuente.""",
    },
]


def seed_prompts() -> None:
    """
    Insert default prompts if they don't already exist.

    Checks for each use_case — if an active prompt already exists,
    skips it (idempotent). Inserts only if missing.
    """
    logger.info("Starting prompt seeding...")

    db = get_supabase_client()

    inserted_count = 0
    skipped_count = 0

    for prompt_data in DEFAULT_PROMPTS:
        use_case = prompt_data["use_case"]

        # Check if an active prompt already exists for this use_case
        existing = (
            db.table("prompts")
            .select("id, name, version")
            .eq("use_case", use_case)
            .eq("is_active", True)
            .execute()
        )

        if existing.data:
            existing_record = existing.data[0]
            logger.info(
                f"SKIP: Prompt already exists for use_case='{use_case}' "
                f"(name='{existing_record['name']}', id={existing_record['id']})"
            )
            skipped_count += 1
            continue

        # Insert the new prompt
        from datetime import datetime, timezone
        result = db.table("prompts").insert({
            **prompt_data,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()

        if result.data:
            logger.info(
                f"INSERTED: Prompt for use_case='{use_case}' "
                f"(name='{prompt_data['name']}')"
            )
            inserted_count += 1
        else:
            logger.error(f"FAILED to insert prompt for use_case='{use_case}'")

    logger.info(
        f"\nSeeding complete: {inserted_count} inserted, {skipped_count} skipped."
    )


if __name__ == "__main__":
    seed_prompts()
