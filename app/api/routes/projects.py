# ============================================================
# app/api/routes/projects.py — Project CRUD endpoints
#
# The projects endpoint is the backbone of the hybrid product:
# every risk assessment, policy, chat thread and trace hangs off
# a project. This router exposes plain CRUD + convenience getters
# for the UI's sidebar (list) and detail screen (get one).
# ============================================================

from fastapi import APIRouter, Depends
from supabase import Client

from app.api.deps import get_current_user, get_db
from app.domain.projects.models import (
    CreateProjectRequest,
    Project,
    UpdateProjectRequest,
)
from app.domain.projects.service import ProjectService

router = APIRouter()


def _serialize(project: Project) -> dict:
    """
    The Project pydantic model uses UUID / datetime types which the
    default FastAPI serializer handles fine, but we wrap in dict so
    the response stays consistent with the other endpoints in this
    codebase (they all return dicts explicitly).
    """
    data = project.model_dump()
    data["id"] = str(project.id)
    data["created_at"] = project.created_at.isoformat()
    data["updated_at"] = project.updated_at.isoformat()
    return data


@router.get("", summary="List projects", tags=["Projects"])
def list_projects(
    limit: int = 50,
    offset: int = 0,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> dict:
    service = ProjectService(db)
    projects = service.list(
        tenant_id=current_user["tenant_id"], limit=limit, offset=offset
    )
    return {
        "projects": [_serialize(p) for p in projects],
        "total": len(projects),
    }


@router.get("/{project_id}", summary="Get a project", tags=["Projects"])
def get_project(
    project_id: str,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> dict:
    service = ProjectService(db)
    project = service.get(project_id, current_user["tenant_id"])
    return _serialize(project)


@router.post("", summary="Create a project", tags=["Projects"])
def create_project(
    request: CreateProjectRequest,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> dict:
    service = ProjectService(db)
    project = service.create(current_user["tenant_id"], request)
    return _serialize(project)


@router.patch("/{project_id}", summary="Update a project", tags=["Projects"])
def update_project(
    project_id: str,
    request: UpdateProjectRequest,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> dict:
    service = ProjectService(db)
    project = service.update(project_id, current_user["tenant_id"], request)
    return _serialize(project)
