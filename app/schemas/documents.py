# ============================================================
# app/schemas/documents.py — Document API request/response schemas
# ============================================================

from datetime import datetime
from pydantic import BaseModel
from uuid import UUID


class DocumentUploadResponse(BaseModel):
    """Response after a successful document upload."""
    id: UUID
    tenant_id: str
    filename: str
    file_type: str
    status: str
    created_at: datetime
    message: str = "Document uploaded. Call /documents/{id}/process to start processing."


class DocumentResponse(BaseModel):
    """Response for a single document (list or detail view)."""
    id: UUID
    tenant_id: str
    filename: str
    file_type: str
    status: str
    error_message: str | None = None
    page_count: int | None = None
    chunk_count: int | None = None
    created_at: datetime
    processed_at: datetime | None = None


class DocumentListResponse(BaseModel):
    """Response for document list endpoint."""
    documents: list[DocumentResponse]
    total: int


class DocumentProcessResponse(BaseModel):
    """Response after triggering document processing."""
    id: UUID
    status: str
    chunk_count: int | None = None
    page_count: int | None = None
    message: str
