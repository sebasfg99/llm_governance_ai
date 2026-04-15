# ============================================================
# app/domain/prompts/models.py — Prompt library models
# ============================================================

from datetime import datetime
from pydantic import BaseModel
from uuid import UUID


class Prompt(BaseModel):
    """
    A versioned system prompt from the prompt library.

    Maps to the `prompts` table in the database.

    The prompt `content` is a full system prompt string that will
    be sent to the LLM as instructions. It defines how the LLM
    should behave for a specific use case.

    Multiple versions can exist for the same (name, use_case).
    Only one should have is_active=True at a time.
    """

    id: UUID
    name: str           # Human-readable label, e.g. "Default RAG Prompt"
    version: int        # Version number (1, 2, 3, ...)
    use_case: str       # The key used to look up this prompt, e.g. "default_rag"
    content: str        # The full system prompt text sent to the LLM
    is_active: bool     # True = this is the current active version
    created_at: datetime

    class Config:
        from_attributes = True


class CreatePromptRequest(BaseModel):
    """Input to create a new prompt in the library."""
    name: str
    version: int = 1
    use_case: str
    content: str
    is_active: bool = True
