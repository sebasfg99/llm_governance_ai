from __future__ import annotations

# ============================================================
# app/domain/threads/service.py — Thread + message persistence
#
# This service is the L1 memory primitive. AIService uses it on
# every query to:
#
#   1. Resolve or create the thread.
#   2. Fetch the tail of messages as conversation history.
#   3. Append the user turn and, after the LLM call, the assistant
#      turn (with its trace_id) so future calls can read them.
#
# Keeping this as its own service (rather than folding it into
# AIService) means the UI can also call it directly — e.g. to
# show the thread list or to rename a thread — without dragging
# the whole AI pipeline in.
# ============================================================

from datetime import datetime, timezone

from supabase import Client

from app.core.errors import NotFoundError
from app.core.logging import get_logger
from app.domain.threads.models import (
    AppendMessageRequest,
    CreateThreadRequest,
    Message,
    Thread,
)

logger = get_logger(__name__)


# How many of the most recent messages to include when hydrating
# the conversation history for the LLM. Chosen conservatively so
# prompts stay small; when the user's real usage pushes against
# it we can raise it or switch to a token-budget-based window.
DEFAULT_HISTORY_WINDOW = 10


class ThreadService:
    def __init__(self, db: Client) -> None:
        self.db = db

    # ------------------------------------------------------------
    # Read
    # ------------------------------------------------------------
    def get(self, thread_id: str, tenant_id: str) -> Thread:
        result = (
            self.db.table("threads")
            .select("*")
            .eq("id", thread_id)
            .eq("tenant_id", tenant_id)
            .limit(1)
            .execute()
        )
        if not result.data:
            raise NotFoundError(f"Thread not found: {thread_id}", resource_id=thread_id)
        return Thread(**result.data[0])

    def list(
        self,
        tenant_id: str,
        user_id: str | None = None,
        project_id: str | None = None,
        limit: int = 50,
    ) -> list[Thread]:
        query = (
            self.db.table("threads")
            .select("*")
            .eq("tenant_id", tenant_id)
            .order("updated_at", desc=True)
            .limit(limit)
        )
        if user_id is not None:
            query = query.eq("user_id", user_id)
        if project_id is not None:
            query = query.eq("project_id", project_id)
        result = query.execute()
        return [Thread(**row) for row in (result.data or [])]

    def get_recent_messages(
        self, thread_id: str, limit: int = DEFAULT_HISTORY_WINDOW
    ) -> list[Message]:
        """
        Return the most recent `limit` messages in chronological order.

        We query DESC + LIMIT then reverse in Python. Cheaper than sorting
        the whole thread and more robust to threads that accumulate
        thousands of messages over time.
        """
        result = (
            self.db.table("messages")
            .select("*")
            .eq("thread_id", thread_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        messages = [Message(**row) for row in (result.data or [])]
        messages.reverse()
        return messages

    # ------------------------------------------------------------
    # Write
    # ------------------------------------------------------------
    def create(
        self, tenant_id: str, user_id: str, request: CreateThreadRequest
    ) -> Thread:
        now = datetime.now(timezone.utc).isoformat()
        payload = {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "project_id": request.project_id,
            "use_case": request.use_case,
            "title": request.title or "Nuevo hilo",
            "status": "open",
            "created_at": now,
            "updated_at": now,
        }
        try:
            result = self.db.table("threads").insert(payload).execute()
        except Exception as exc:
            # GovAI project IDs are not mirrored in the sidecar's projects table.
            # When the FK violation fires (23503), retry as an unlinked thread.
            pg_code = getattr(exc, "code", None)
            if not pg_code:
                arg0 = exc.args[0] if exc.args else {}
                pg_code = arg0.get("code") if isinstance(arg0, dict) else None
            if not pg_code:
                pg_code = "23503" if "23503" in str(exc) else None
            if pg_code == "23503" and request.project_id:
                logger.warning(
                    "project_id=%s not found in sidecar projects table — creating unlinked thread",
                    request.project_id,
                )
                payload["project_id"] = None
                result = self.db.table("threads").insert(payload).execute()
            else:
                raise
        if not result.data:
            raise RuntimeError("Failed to create thread")
        return Thread(**result.data[0])

    def get_or_create(
        self,
        tenant_id: str,
        user_id: str,
        thread_id: str | None,
        use_case: str,
        project_id: str | None,
        first_user_message: str | None = None,
    ) -> Thread:
        """
        Resolve an existing thread or create a new one.

        Called at the top of every AIService.query() so the rest of
        the pipeline can always assume a thread exists. When creating
        a new thread, we seed the title from the first user message
        (truncated) — the UI can rename it later.
        """
        if thread_id:
            return self.get(thread_id, tenant_id)

        # Seed the title from the user query so the sidebar is not
        # full of "Nuevo hilo" entries. 60 chars is enough to recognise.
        title = "Nuevo hilo"
        if first_user_message:
            clean = first_user_message.strip().replace("\n", " ")
            title = (clean[:60] + "…") if len(clean) > 60 else clean or title

        return self.create(
            tenant_id,
            user_id,
            CreateThreadRequest(
                use_case=use_case, project_id=project_id, title=title
            ),
        )

    def append_message(
        self, thread_id: str, request: AppendMessageRequest
    ) -> Message:
        payload = {
            "thread_id": thread_id,
            "role": request.role,
            "content": request.content,
            "trace_id": request.trace_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        result = self.db.table("messages").insert(payload).execute()
        if not result.data:
            raise RuntimeError("Failed to append message")

        # Bump thread.updated_at so the sidebar sort stays correct.
        self.db.table("threads").update(
            {"updated_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", thread_id).execute()

        return Message(**result.data[0])
