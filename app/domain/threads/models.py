# ============================================================
# app/domain/threads/models.py — Conversation thread models
#
# A "thread" is a persistent chat session. Every message in a
# thread ties back via thread_id, and every assistant message
# further links to the trace that generated it via trace_id.
#
# Threads are cheap: we create one on the first call when the
# client doesn't pass thread_id, return the id in the response,
# and expect subsequent calls to include it.
# ============================================================

from datetime import datetime
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field


ThreadStatus = Literal["open", "archived"]
MessageRole = Literal["user", "assistant", "system"]


class Thread(BaseModel):
    id: UUID
    tenant_id: str
    user_id: str
    project_id: UUID | None = None
    use_case: str
    title: str
    status: ThreadStatus = "open"
    created_at: datetime
    updated_at: datetime

    class Config:
        from_attributes = True


class Message(BaseModel):
    id: UUID
    thread_id: UUID
    role: MessageRole
    content: str
    trace_id: UUID | None = None
    created_at: datetime

    class Config:
        from_attributes = True


class CreateThreadRequest(BaseModel):
    use_case: str = Field(default="chatbot")
    project_id: str | None = None
    title: str | None = None


class AppendMessageRequest(BaseModel):
    role: MessageRole
    content: str
    trace_id: str | None = None
