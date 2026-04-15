# ============================================================
# app/api/routes/prompts.py — Prompt library endpoints
# ============================================================

from fastapi import APIRouter, Depends
from supabase import Client
from pydantic import BaseModel

from app.api.deps import get_current_user, get_db
from app.domain.prompts.service import PromptService
from app.domain.prompts.models import CreatePromptRequest

router = APIRouter()


class PromptResponseItem(BaseModel):
    id: str
    name: str
    version: int
    use_case: str
    content: str
    is_active: bool
    created_at: str


@router.get(
    "",
    summary="List all prompts",
    description="Return all prompts in the library (all use cases and versions).",
    tags=["Prompts"],
)
def list_prompts(
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> dict:
    """List all versioned prompts in the library."""
    service = PromptService(db)
    prompts = service.list_prompts()

    return {
        "prompts": [
            {
                "id": str(p.id),
                "name": p.name,
                "version": p.version,
                "use_case": p.use_case,
                "content": p.content,
                "is_active": p.is_active,
                "created_at": p.created_at.isoformat(),
            }
            for p in prompts
        ],
        "total": len(prompts),
    }


@router.post(
    "",
    summary="Create a prompt",
    description="Add a new prompt to the library. If is_active=true, deactivates the current active prompt for the same use_case.",
    tags=["Prompts"],
)
def create_prompt(
    request: CreatePromptRequest,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> dict:
    """Create a new prompt in the library."""
    service = PromptService(db)
    prompt = service.create_prompt(request)

    return {
        "id": str(prompt.id),
        "name": prompt.name,
        "version": prompt.version,
        "use_case": prompt.use_case,
        "is_active": prompt.is_active,
        "message": "Prompt created successfully.",
    }
