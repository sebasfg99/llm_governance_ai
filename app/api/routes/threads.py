# ============================================================
# app/api/routes/threads.py — Conversation threads endpoints
#
# The UI calls these to:
#   - Render the chat sidebar (list)
#   - Load a thread's history when the user clicks it (get + messages)
#   - Optionally create a thread without sending a message (rare;
#     normally threads are created implicitly by POST /ai/query).
#
# We do not expose a "send message" endpoint here on purpose. Every
# message goes through POST /ai/query which owns the LLM call. The
# only write endpoint is thread creation.
# ============================================================

from fastapi import APIRouter, Depends, Query
from supabase import Client

from app.api.deps import get_current_user, get_db
from app.domain.threads.models import CreateThreadRequest, Message, Thread
from app.domain.threads.service import ThreadService

router = APIRouter()


def _serialize_thread(t: Thread) -> dict:
    return {
        "id": str(t.id),
        "tenant_id": t.tenant_id,
        "user_id": t.user_id,
        "project_id": str(t.project_id) if t.project_id else None,
        "use_case": t.use_case,
        "title": t.title,
        "status": t.status,
        "created_at": t.created_at.isoformat(),
        "updated_at": t.updated_at.isoformat(),
    }


def _serialize_message(m: Message) -> dict:
    return {
        "id": str(m.id),
        "thread_id": str(m.thread_id),
        "role": m.role,
        "content": m.content,
        "trace_id": str(m.trace_id) if m.trace_id else None,
        "created_at": m.created_at.isoformat(),
    }


@router.get("", summary="List threads", tags=["Threads"])
def list_threads(
    project_id: str | None = Query(default=None),
    only_mine: bool = Query(default=True),
    limit: int = 50,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> dict:
    """
    Sidebar feed. Defaults to only_mine=True so the UI shows the
    user's own threads. Set only_mine=false for an admin view
    (once we have one).
    """
    service = ThreadService(db)
    threads = service.list(
        tenant_id=current_user["tenant_id"],
        user_id=current_user["user_id"] if only_mine else None,
        project_id=project_id,
        limit=limit,
    )
    return {
        "threads": [_serialize_thread(t) for t in threads],
        "total": len(threads),
    }


@router.post("", summary="Create an empty thread", tags=["Threads"])
def create_thread(
    request: CreateThreadRequest,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> dict:
    service = ThreadService(db)
    thread = service.create(
        tenant_id=current_user["tenant_id"],
        user_id=current_user["user_id"],
        request=request,
    )
    return _serialize_thread(thread)


@router.get("/{thread_id}", summary="Get a thread", tags=["Threads"])
def get_thread(
    thread_id: str,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> dict:
    service = ThreadService(db)
    thread = service.get(thread_id, current_user["tenant_id"])
    return _serialize_thread(thread)


@router.get(
    "/{thread_id}/messages",
    summary="List messages for a thread",
    tags=["Threads"],
)
def list_thread_messages(
    thread_id: str,
    limit: int = 200,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> dict:
    """
    Returns all messages in a thread in chronological order.
    limit is applied in the service (most recent N, then reversed).
    """
    service = ThreadService(db)
    # Enforce tenant check: load the thread first so a caller from
    # another tenant cannot peek at messages by guessing the id.
    service.get(thread_id, current_user["tenant_id"])
    messages = service.get_recent_messages(thread_id=thread_id, limit=limit)
    return {
        "messages": [_serialize_message(m) for m in messages],
        "total": len(messages),
    }
