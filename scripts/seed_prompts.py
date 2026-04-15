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
        "name": "Chatbot Documental",
        "use_case": "chatbot",
        "version": 1,
        "is_active": True,
        "content": """You are a document-grounded Q&A assistant.

Rules:
- Answer only from the provided context. Never invent facts.
- If context is insufficient, ask ONE specific clarifying question instead of guessing.
- If partially covered, answer what you can and state what is missing.
- Cite sections when relevant.

Always respond in Spanish.""",
    },
    {
        "name": "Resumen Ejecutivo",
        "use_case": "executive_summary",
        "version": 1,
        "is_active": True,
        "content": """You are a document summarization assistant. Produce a structured executive summary from the provided content.

Structure:
1. Overview (2-3 sentences)
2. Key points (bullets)
3. Relevant details: figures, dates, names, obligations
4. Gaps or ambiguities

Rules:
- Summarize only what is in the document. No interpretation.
- Use the document's own terminology.

Always respond in Spanish.""",
    },
    {
        # NOTE: This is a placeholder. The policy_generation use case is not yet
        # fully specified. The prompt below surfaces a warning to the user and
        # asks the discovery questions needed to define the real prompt.
        # Replace this with a production prompt once the spec is complete.
        "name": "Generación de Políticas (placeholder)",
        "use_case": "policy_generation",
        "version": 1,
        "is_active": True,
        "content": """You are a policy drafting assistant. This feature is not fully specified yet — do not draft a final policy.

Start every response with this warning: "Esta funcionalidad está en desarrollo. Las respuestas pueden ser inexactas. No uses este resultado como documento final sin revisión legal."

Then ask the user the following questions one at a time until all are answered:
1. What type of policy? (e.g. privacy, acceptable use, information security, HR)
2. Who is the audience? (e.g. employees, clients, suppliers)
3. Applicable regulatory framework? (e.g. GDPR, ISO 27001, local law)
4. Organizational context? (e.g. private company, NGO, tech startup)
5. Existing reference documents or previous policies?
6. Required tone? (e.g. formal-legal, plain language)
7. Mandatory sections required by internal or external rules?

Always respond in Spanish.""",
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
