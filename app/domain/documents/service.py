# ============================================================
# app/domain/documents/service.py — Document ingestion service
#
# WHY THIS FILE EXISTS:
# The DocumentService is the "brain" of the document domain.
# It orchestrates the entire ingestion pipeline:
#
#   Upload file → Save to disk → Create DB record
#   → Parse text (PDF/DOCX) → Split into chunks
#   → Generate embeddings → Save chunks to DB
#   → Mark document as "ready"
#
# All parsing, chunking, and embedding logic lives in separate
# modules. This service just coordinates them in the right order.
#
# DESIGN PRINCIPLE:
# The service never knows how PDF parsing works internally.
# It calls parse_pdf() and gets back text. This means we can
# swap out the parser later (e.g., add OCR) without changing
# the service at all.
# ============================================================

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path

from supabase import Client

from app.core.config import settings
from app.core.logging import get_logger
from app.core.errors import NotFoundError, DocumentProcessingError, TenantIsolationError

from app.domain.documents.models import Document, DocumentChunk, DocumentStatus, FileType
from app.domain.documents.parser_pdf import parse_pdf
from app.domain.documents.parser_docx import parse_docx
from app.domain.documents.parser_txt import parse_txt
from app.domain.documents.chunking import chunk_text
from app.domain.documents.embeddings import embed_texts

# Logger for this module
logger = get_logger(__name__)


class DocumentService:
    """
    Handles document upload, processing, and retrieval.

    This service manages the full document ingestion pipeline:
    from raw file bytes to searchable embedded chunks in the DB.

    Usage:
        db = get_supabase_client()
        service = DocumentService(db)
        doc = await service.upload_document(
            tenant_id="acme",
            filename="contract.pdf",
            file_bytes=b"...",
        )
        service.process_document(doc.id)
    """

    def __init__(self, db: Client) -> None:
        """
        Initialize the service with a Supabase client.

        Args:
            db: An initialized Supabase Client from app.db.supabase
        """
        self.db = db
        # Ensure the uploads directory exists at startup
        os.makedirs(settings.upload_dir, exist_ok=True)

    # ----------------------------------------------------------
    # upload_document
    # Saves the file to disk and creates a DB record.
    # Does NOT start processing — call process_document() separately.
    # ----------------------------------------------------------
    def upload_document(
        self,
        tenant_id: str,
        filename: str,
        file_bytes: bytes,
        country: str | None = None,
    ) -> Document:
        """
        Save an uploaded file to disk and create its database record.

        This method only handles storage — it does NOT parse or embed.
        Call process_document() after this to run the full pipeline.

        Args:
            tenant_id: The tenant that owns this document.
            filename:  The original filename (e.g., "contract.pdf").
            file_bytes: The raw file content as bytes.
            country:   Normalized country slug this document applies to
                       (e.g. "costa-rica"). None means global (all countries).

        Returns:
            The newly created Document record with status="pending".

        Raises:
            DocumentProcessingError: If the file type is not supported.
        """
        logger.info(f"Uploading document: {filename} for tenant: {tenant_id}, country: {country}")

        # --- Determine file type from extension ---
        # We only support PDF and DOCX in the MVP
        extension = Path(filename).suffix.lower().lstrip(".")
        if extension == "pdf":
            file_type = FileType.PDF
        elif extension == "docx":
            file_type = FileType.DOCX
        elif extension == "txt":
            file_type = FileType.TXT
        else:
            raise DocumentProcessingError(
                f"Unsupported file type: .{extension}. "
                "Only PDF, DOCX, and TXT files are supported."
            )

        # --- Create tenant-scoped directory ---
        # Structure: ./uploads/{tenant_id}/
        # This keeps each tenant's files physically separate on disk
        tenant_upload_dir = Path(settings.upload_dir) / tenant_id
        tenant_upload_dir.mkdir(parents=True, exist_ok=True)

        # --- Generate a unique filename to avoid collisions ---
        # Prefix the original filename with a UUID to handle duplicate uploads
        unique_filename = f"{uuid.uuid4().hex}_{filename}"
        file_path = str(tenant_upload_dir / unique_filename)

        # --- Save file to disk ---
        with open(file_path, "wb") as f:
            f.write(file_bytes)

        logger.debug(f"File saved to: {file_path}")

        # --- Create the document record in the database ---
        # Status starts as "pending" — process_document() will update it
        now = datetime.now(timezone.utc).isoformat()
        doc_data = {
            "tenant_id": tenant_id,
            "filename": filename,           # Store original filename for display
            "file_type": file_type.value,   # "pdf" or "docx"
            "file_path": file_path,         # Full path on disk
            "status": DocumentStatus.PENDING.value,
            "country": country,             # None = global, slug = country-specific
            "created_at": now,
        }

        result = self.db.table("documents").insert(doc_data).execute()

        if not result.data:
            raise DocumentProcessingError(
                "Failed to create document record in database"
            )

        # Convert the DB row dict into our Document model
        doc = Document(**result.data[0])

        logger.info(f"Document uploaded successfully: id={doc.id}, status=pending")
        return doc

    # ----------------------------------------------------------
    # process_document
    # Runs the full ingestion pipeline for an uploaded document.
    # ----------------------------------------------------------
    def process_document(self, document_id: str) -> Document:
        """
        Run the full processing pipeline for an uploaded document.

        Pipeline:
          1. Fetch document record from DB
          2. Set status to "processing"
          3. Parse text from file (PDF or DOCX)
          4. Split text into chunks
          5. Generate embeddings for all chunks
          6. Save all chunks to document_chunks table
          7. Set status to "ready"

        On any failure: set status to "failed" and store error_message.

        Args:
            document_id: UUID string of the document to process.

        Returns:
            The updated Document record with status="ready".

        Raises:
            NotFoundError: If the document doesn't exist.
            DocumentProcessingError: If parsing or embedding fails.
        """
        logger.info(f"Starting processing pipeline for document: {document_id}")

        # --- Fetch the document record ---
        doc = self.get_document(document_id)
        tenant_id = doc.tenant_id

        # --- Mark as "processing" ---
        self._update_status(document_id, DocumentStatus.PROCESSING)

        try:
            # --- Step 1: Parse the file ---
            # Dispatch to the correct parser based on file type
            if doc.file_type == FileType.PDF:
                raw_text, page_count = parse_pdf(doc.file_path)
            elif doc.file_type == FileType.DOCX:
                raw_text, page_count = parse_docx(doc.file_path)
            elif doc.file_type == FileType.TXT:
                raw_text, page_count = parse_txt(doc.file_path)
            else:
                raise DocumentProcessingError(f"Unsupported file type: {doc.file_type}")

            logger.debug(f"Parsed {len(raw_text)} characters from {doc.filename}")

            # --- Step 2: Split text into chunks ---
            chunks_text = chunk_text(raw_text)
            chunk_count = len(chunks_text)

            if chunk_count == 0:
                raise DocumentProcessingError("Document produced no text chunks after parsing")

            logger.debug(f"Split into {chunk_count} chunks")

            # --- Step 3: Generate embeddings for all chunks ---
            # This makes one API call to OpenAI with all chunks batched
            logger.debug(f"Generating embeddings for {chunk_count} chunks...")
            embeddings = embed_texts(chunks_text)

            # Sanity check: one embedding per chunk
            if len(embeddings) != chunk_count:
                raise DocumentProcessingError(
                    f"Embedding count mismatch: got {len(embeddings)}, expected {chunk_count}"
                )

            # --- Step 4: Save chunks to the database ---
            # Build the list of rows to insert
            now = datetime.now(timezone.utc).isoformat()
            chunk_rows = [
                {
                    "document_id": str(document_id),
                    "tenant_id": tenant_id,
                    "content": chunk_text_item,
                    "chunk_index": idx,
                    "embedding": embedding,  # List of 1536 floats
                    "country": doc.country,  # Propagated from parent document
                    "created_at": now,
                }
                for idx, (chunk_text_item, embedding) in enumerate(
                    zip(chunks_text, embeddings)
                )
            ]

            # Insert all chunks in a single DB operation
            # This is much faster than inserting one-by-one
            self.db.table("document_chunks").insert(chunk_rows).execute()
            logger.debug(f"Saved {chunk_count} chunks to database")

            # --- Step 5: Update document status to "ready" ---
            now_ts = datetime.now(timezone.utc).isoformat()
            self.db.table("documents").update({
                "status": DocumentStatus.READY.value,
                "page_count": page_count,
                "chunk_count": chunk_count,
                "processed_at": now_ts,
            }).eq("id", str(document_id)).execute()

            logger.info(
                f"Document {document_id} processed successfully: "
                f"{page_count} pages, {chunk_count} chunks"
            )

            # Return the updated document record
            return self.get_document(document_id)

        except Exception as e:
            # --- Handle any failure: mark document as "failed" ---
            logger.error(f"Document processing failed for {document_id}: {e}", exc_info=True)

            self.db.table("documents").update({
                "status": DocumentStatus.FAILED.value,
                "error_message": str(e),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", str(document_id)).execute()

            # Re-raise so the API layer can return an appropriate error response
            if isinstance(e, (NotFoundError, DocumentProcessingError)):
                raise
            raise DocumentProcessingError(
                f"Document processing failed: {e}",
                resource_id=str(document_id),
            )

    # ----------------------------------------------------------
    # list_documents
    # Returns all documents belonging to a tenant.
    # ----------------------------------------------------------
    def list_documents(self, tenant_id: str) -> list[Document]:
        """
        List all documents for a given tenant.

        Args:
            tenant_id: The tenant's identifier.

        Returns:
            List of Document records, ordered by creation date (newest first).
        """
        result = (
            self.db.table("documents")
            .select("*")
            .eq("tenant_id", tenant_id)          # CRITICAL: filter by tenant
            .order("created_at", desc=True)       # Newest first
            .execute()
        )

        # Convert each DB row dict to a Document model
        return [Document(**row) for row in result.data]

    # ----------------------------------------------------------
    # get_document
    # Returns a single document by ID, with tenant isolation check.
    # ----------------------------------------------------------
    def get_document(self, document_id: str, tenant_id: str | None = None) -> Document:
        """
        Retrieve a single document by its ID.

        If tenant_id is provided, also verifies the document belongs
        to that tenant (raises TenantIsolationError if not).

        Args:
            document_id: UUID of the document to fetch.
            tenant_id:   If provided, verify ownership.

        Returns:
            The Document record.

        Raises:
            NotFoundError: If no document with that ID exists.
            TenantIsolationError: If the document belongs to a different tenant.
        """
        result = (
            self.db.table("documents")
            .select("*")
            .eq("id", str(document_id))
            .execute()
        )

        if not result.data:
            raise NotFoundError(
                f"Document not found: {document_id}",
                resource_id=str(document_id),
            )

        doc = Document(**result.data[0])

        # Tenant isolation check — if tenant_id is provided, verify ownership
        if tenant_id and doc.tenant_id != tenant_id:
            logger.warning(
                f"Tenant isolation violation: tenant {tenant_id} "
                f"attempted to access document {document_id} "
                f"owned by tenant {doc.tenant_id}"
            )
            raise TenantIsolationError(
                "Access denied: this document belongs to a different tenant"
            )

        return doc

    # ----------------------------------------------------------
    # _update_status (private helper)
    # ----------------------------------------------------------
    def _update_status(self, document_id: str, status: DocumentStatus) -> None:
        """Update only the status field of a document record."""
        self.db.table("documents").update(
            {"status": status.value}
        ).eq("id", str(document_id)).execute()
