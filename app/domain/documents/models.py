# ============================================================
# app/domain/documents/models.py — Document domain models
#
# WHY THIS FILE EXISTS:
# Defines the data shapes (Pydantic models) that represent
# documents and chunks within our application logic.
#
# DIFFERENCE FROM SCHEMAS:
#   - models.py (this file): internal domain representations,
#     used by services and passed between modules
#   - schemas/documents.py: API-facing request/response shapes,
#     used by FastAPI route handlers
#
# These two layers can diverge as the app grows — for example,
# internal models might have fields you don't want to expose
# in the API (like raw file paths or internal IDs).
# ============================================================

from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field
from uuid import UUID


# ============================================================
# DocumentStatus enum
#
# Represents the lifecycle state of a document.
# A document moves through these states during ingestion.
# ============================================================
class DocumentStatus(str, Enum):
    """
    The processing lifecycle of a document.

    pending    → The document was uploaded but processing hasn't started yet.
    processing → The system is currently parsing, chunking, and embedding.
    ready      → Processing succeeded; the document is available for RAG queries.
    failed     → Processing failed; see error_message for details.
    """
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


# ============================================================
# FileType enum
#
# The supported document types in the MVP.
# TODO: Add "txt", "html", "csv" in future iterations.
# ============================================================
class FileType(str, Enum):
    """Supported document file formats."""
    PDF = "pdf"
    DOCX = "docx"


# ============================================================
# Document model
#
# Represents a row in the `documents` table.
# This is used by DocumentService and passed to API routes.
# ============================================================
class Document(BaseModel):
    """
    Internal representation of a document record.

    Maps 1:1 to the `documents` database table.
    Returned by DocumentService methods.
    """

    id: UUID
    tenant_id: str              # Which tenant owns this document
    filename: str               # Original uploaded filename (e.g., "contract.pdf")
    file_type: FileType         # "pdf" or "docx"
    file_path: str              # Absolute or relative path to the file on disk
    status: DocumentStatus      # Current processing state

    error_message: str | None = None    # Set if processing failed
    page_count: int | None = None       # Populated after successful processing
    chunk_count: int | None = None      # Number of text chunks produced

    created_at: datetime
    processed_at: datetime | None = None  # Set when processing completes

    class Config:
        # Allow creating a Document from a dict with extra fields
        # (e.g., from a Supabase query response)
        from_attributes = True


# ============================================================
# DocumentChunk model
#
# Represents a row in the `document_chunks` table.
# One document produces many chunks.
# ============================================================
class DocumentChunk(BaseModel):
    """
    Internal representation of a document chunk.

    A chunk is a small piece of text (~500 tokens) extracted
    from a document. It carries an embedding vector used for
    RAG retrieval.
    """

    id: UUID
    document_id: UUID           # Links back to the parent document
    tenant_id: str              # Denormalized for fast queries (avoids JOINs)
    content: str                # The actual text of this chunk
    chunk_index: int            # Position within the document (0-based)

    # The embedding is a list of floats (1536 dimensions)
    # It's None when the chunk is first created and populated
    # by the embedding step of document processing.
    embedding: list[float] | None = None

    created_at: datetime

    class Config:
        from_attributes = True


# ============================================================
# CreateDocumentRequest
#
# Data needed to create a new document record in the DB.
# Used internally by DocumentService.upload_document().
# ============================================================
class CreateDocumentRequest(BaseModel):
    """Input to create a new document record."""
    tenant_id: str
    filename: str
    file_type: FileType
    file_path: str
