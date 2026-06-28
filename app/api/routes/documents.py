# ============================================================
# app/api/routes/documents.py — Document management endpoints
# ============================================================

from fastapi import APIRouter, Depends, UploadFile, File, Form
from supabase import Client

from app.api.deps import get_current_user, get_db
from app.domain.documents.service import DocumentService
from app.schemas.documents import (
    DocumentUploadResponse,
    DocumentResponse,
    DocumentListResponse,
    DocumentProcessResponse,
)

router = APIRouter()

# Maximum upload size: 50MB
MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    summary="Upload a document",
    description="Upload a PDF or DOCX file. Call /documents/{id}/process after upload to trigger processing.",
    tags=["Documents"],
)
async def upload_document(
    file: UploadFile = File(..., description="PDF or DOCX file to upload"),
    country: str | None = Form(default=None, description="Country this document applies to (e.g. 'costa-rica'). Omit for global documents."),
    framework: str | None = Form(default=None, description="Normative framework slug (e.g. 'iso_42001'). Omit for framework-agnostic documents."),
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> DocumentUploadResponse:
    """
    Upload a document file for a tenant.

    The bytes are written to the Supabase Storage bucket
    `rag-documents` and a `documents` row is created with
    status="pending". Processing (parsing + embedding) is
    triggered separately via POST /documents/{id}/process so
    large files don't block the upload response.
    """
    # Read the file bytes
    file_bytes = await file.read()

    # Create the service and upload
    service = DocumentService(db)
    doc = service.upload_document(
        tenant_id=current_user["tenant_id"],
        filename=file.filename or "untitled",
        file_bytes=file_bytes,
        country=country,
        framework=framework,
    )

    return DocumentUploadResponse(
        id=doc.id,
        tenant_id=doc.tenant_id,
        filename=doc.filename,
        file_type=doc.file_type.value,
        status=doc.status.value,
        country=doc.country,
        created_at=doc.created_at,
    )


@router.post(
    "/{document_id}/process",
    response_model=DocumentProcessResponse,
    summary="Process a document",
    description="Parse, chunk, and embed an uploaded document. Makes it available for RAG queries.",
    tags=["Documents"],
)
def process_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> DocumentProcessResponse:
    """
    Trigger the processing pipeline for an uploaded document.

    This runs synchronously: parses the file, creates text chunks,
    generates embeddings, and saves everything to the database.
    The document's status changes from "pending" → "ready" (or "failed").
    """
    service = DocumentService(db)
    doc = service.process_document(document_id)

    return DocumentProcessResponse(
        id=doc.id,
        status=doc.status.value,
        chunk_count=doc.chunk_count,
        page_count=doc.page_count,
        message=f"Document processed successfully: {doc.chunk_count} chunks created.",
    )


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List documents",
    description="List all documents for the current tenant.",
    tags=["Documents"],
)
def list_documents(
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> DocumentListResponse:
    """Return all documents belonging to the current tenant."""
    service = DocumentService(db)
    docs = service.list_documents(tenant_id=current_user["tenant_id"])

    return DocumentListResponse(
        documents=[
            DocumentResponse(
                id=d.id,
                tenant_id=d.tenant_id,
                filename=d.filename,
                file_type=d.file_type.value,
                status=d.status.value,
                country=d.country,
                error_message=d.error_message,
                page_count=d.page_count,
                chunk_count=d.chunk_count,
                created_at=d.created_at,
                processed_at=d.processed_at,
            )
            for d in docs
        ],
        total=len(docs),
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get document details",
    tags=["Documents"],
)
def get_document(
    document_id: str,
    current_user: dict = Depends(get_current_user),
    db: Client = Depends(get_db),
) -> DocumentResponse:
    """Get details for a specific document."""
    service = DocumentService(db)
    doc = service.get_document(
        document_id=document_id,
        tenant_id=current_user["tenant_id"],
    )

    return DocumentResponse(
        id=doc.id,
        tenant_id=doc.tenant_id,
        filename=doc.filename,
        file_type=doc.file_type.value,
        status=doc.status.value,
        country=doc.country,
        error_message=doc.error_message,
        page_count=doc.page_count,
        chunk_count=doc.chunk_count,
        created_at=doc.created_at,
        processed_at=doc.processed_at,
    )
