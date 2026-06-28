# ============================================================
# app/domain/documents/service.py — Document ingestion service
#
# WHY THIS FILE EXISTS:
# The DocumentService is the "brain" of the document domain.
# It orchestrates the entire ingestion pipeline:
#
#   Upload file → Save to Supabase Storage → Create DB record
#   → Parse text (PDF/DOCX) → Split into chunks
#   → Generate embeddings → Save chunks to DB
#   → Mark document as "ready"
#
# All parsing, chunking, and embedding logic lives in separate
# modules. This service just coordinates them in the right order.
#
# WHY SUPABASE STORAGE (not local disk):
# Sandbox runs on Railway with ephemeral disks — a container
# restart erases local files. Production target (Cloud Run) has
# the same constraint. Storing originals in the Supabase bucket
# `rag-documents` means uploads survive restarts, the re-ingest
# script can pull objects directly, and the sidecar stays
# horizontally scalable.
# ============================================================

import os
import re
import tempfile
import unicodedata
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
from app.domain.documents.chunking import chunk_text_with_sections
from app.domain.documents.embeddings import embed_texts

logger = get_logger(__name__)


def _sanitize_storage_filename(filename: str) -> str:
    """
    Make a filename safe for Supabase Storage keys.
    Strips accents, replaces spaces with underscores, removes chars
    that aren't alphanumeric, hyphens, underscores, or dots.
    """
    nfkd = unicodedata.normalize("NFKD", filename)
    ascii_str = nfkd.encode("ascii", "ignore").decode("ascii")
    safe = re.sub(r"[^\w.\-]", "_", ascii_str)
    safe = re.sub(r"_+", "_", safe).strip("_")
    return safe or "document"


# MIME types used when uploading to Storage. Required by the
# Supabase storage API for correct content negotiation when files
# are downloaded later (signed URLs, browser preview, etc.).
_MIME_BY_EXTENSION = {
    "pdf": "application/pdf",
    "docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "txt": "text/plain; charset=utf-8",
}


class DocumentService:
    """
    Handles document upload, processing, and retrieval.

    This service manages the full document ingestion pipeline:
    from raw file bytes to searchable embedded chunks in the DB.
    Originals are stored in Supabase Storage so the sidecar
    remains stateless and horizontally scalable.

    Usage:
        db = get_supabase_client()
        service = DocumentService(db)
        doc = service.upload_document(
            tenant_id="acme",
            filename="contract.pdf",
            file_bytes=b"...",
        )
        service.process_document(doc.id)
    """

    def __init__(self, db: Client) -> None:
        self.db = db
        self._bucket = settings.storage_bucket

    # ----------------------------------------------------------
    # upload_document
    # ----------------------------------------------------------
    def upload_document(
        self,
        tenant_id: str,
        filename: str,
        file_bytes: bytes,
        country: str | None = None,
        framework: str | None = None,
    ) -> Document:
        """
        Save an uploaded file to Supabase Storage and create its
        database record. Does NOT parse or embed — call
        process_document() after this.

        Args:
            tenant_id:  The tenant that owns this document.
            filename:   The original filename (e.g., "contract.pdf").
            file_bytes: The raw file content as bytes.
            country:    Normalized country slug this document applies
                        to (e.g. "costa-rica"). None = global.
            framework:  Normative framework slug (e.g. "iso_42001").
                        None = framework-agnostic.

        Returns:
            The newly created Document record with status="pending".

        Raises:
            DocumentProcessingError: If the file type is not supported
                or the Storage upload fails.
        """
        logger.info(
            f"Uploading document: {filename} for tenant: {tenant_id}, "
            f"country: {country}, framework: {framework}, "
            f"size: {len(file_bytes)} bytes"
        )

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

        # Insert the document row first so we have a stable UUID to
        # use in the Storage object path. The status stays "pending"
        # until the bytes are confirmed in Storage.
        now = datetime.now(timezone.utc).isoformat()
        doc_data = {
            "tenant_id": tenant_id,
            "filename": filename,
            "file_type": file_type.value,
            "file_path": "",  # Filled in after we know the document UUID
            "status": DocumentStatus.PENDING.value,
            "country": country,
            "framework": framework,
            "created_at": now,
        }
        result = self.db.table("documents").insert(doc_data).execute()
        if not result.data:
            raise DocumentProcessingError(
                "Failed to create document record in database"
            )
        doc_row = result.data[0]
        document_id = doc_row["id"]

        # Storage path scheme: {tenant_id}/{document_id}/{safe_filename}
        # Using the document UUID as a directory means we never
        # collide even when two documents share a filename, and we
        # can delete a document's storage tree by prefix.
        # Supabase Storage rejects keys with spaces / accents — sanitize.
        safe_filename = _sanitize_storage_filename(filename)
        storage_path = f"{tenant_id}/{document_id}/{safe_filename}"

        try:
            self._storage_upload(
                storage_path=storage_path,
                file_bytes=file_bytes,
                content_type=_MIME_BY_EXTENSION.get(extension, "application/octet-stream"),
            )
        except Exception as exc:
            # Roll back the DB row so a failed upload doesn't leave
            # an orphan "pending" document the user can't recover.
            logger.error(
                f"Storage upload failed for document {document_id}: {exc}",
                exc_info=True,
            )
            try:
                self.db.table("documents").delete().eq("id", document_id).execute()
            except Exception as cleanup_exc:
                logger.warning(
                    f"Failed to clean up DB row after upload error: {cleanup_exc}"
                )
            raise DocumentProcessingError(
                f"Failed to upload file to Storage: {exc}"
            ) from exc

        # Persist the storage path now that the upload succeeded.
        self.db.table("documents").update({"file_path": storage_path}).eq(
            "id", document_id
        ).execute()
        doc_row["file_path"] = storage_path

        doc = Document(**doc_row)
        logger.info(
            f"Document uploaded successfully: id={doc.id}, "
            f"storage_path={storage_path}, status=pending"
        )
        return doc

    # ----------------------------------------------------------
    # process_document
    # ----------------------------------------------------------
    def process_document(self, document_id: str) -> Document:
        """
        Run the full processing pipeline for an uploaded document.

        Pipeline:
          1. Fetch document record from DB
          2. Set status to "processing"
          3. Download bytes from Supabase Storage to a temp file
          4. Parse text from file (PDF/DOCX/TXT)
          5. Split text into chunks
          6. Generate embeddings for all chunks
          7. Save all chunks to document_chunks table
          8. Set status to "ready"

        On any failure: set status to "failed" and store error_message.
        """
        logger.info(f"Starting processing pipeline for document: {document_id}")

        doc = self.get_document(document_id)
        tenant_id = doc.tenant_id

        self._update_status(document_id, DocumentStatus.PROCESSING)

        tmp_file: str | None = None
        try:
            # --- Step 1: pull bytes from Storage to a temp file ---
            # Parsers are filesystem-based (pypdf opens a path); a
            # /tmp file is cheap and respects Cloud Run's writable
            # /tmp area. Cleaned up in the finally clause.
            file_bytes = self._storage_download(doc.file_path)
            suffix = f".{doc.file_type.value}"
            fd, tmp_file = tempfile.mkstemp(suffix=suffix, prefix="aicore_")
            with os.fdopen(fd, "wb") as f:
                f.write(file_bytes)
            logger.debug(
                f"Downloaded {len(file_bytes)} bytes from "
                f"storage[{doc.file_path}] → {tmp_file}"
            )

            # --- Step 2: Parse the file ---
            if doc.file_type == FileType.PDF:
                raw_text, page_count = parse_pdf(tmp_file)
            elif doc.file_type == FileType.DOCX:
                raw_text, page_count = parse_docx(tmp_file)
            elif doc.file_type == FileType.TXT:
                raw_text, page_count = parse_txt(tmp_file)
            else:
                raise DocumentProcessingError(f"Unsupported file type: {doc.file_type}")

            logger.debug(f"Parsed {len(raw_text)} characters from {doc.filename}")

            # --- Step 3: Section-aware chunking ---
            chunks_with_sections = chunk_text_with_sections(raw_text)
            chunk_count = len(chunks_with_sections)
            if chunk_count == 0:
                raise DocumentProcessingError(
                    "Document produced no text chunks after parsing"
                )
            logger.debug(
                f"Split into {chunk_count} chunks "
                f"({sum(1 for c in chunks_with_sections if c.section_ref)} with section_ref)"
            )

            # --- Step 4: Embeddings ---
            logger.debug(f"Generating embeddings for {chunk_count} chunks...")
            chunk_texts = [c.content for c in chunks_with_sections]
            embeddings = embed_texts(chunk_texts)
            if len(embeddings) != chunk_count:
                raise DocumentProcessingError(
                    f"Embedding count mismatch: got {len(embeddings)}, expected {chunk_count}"
                )

            # --- Step 5: Persist chunks (replace any from prior run) ---
            self.db.table("document_chunks").delete().eq(
                "document_id", str(document_id)
            ).execute()

            now = datetime.now(timezone.utc).isoformat()
            chunk_rows = [
                {
                    "document_id": str(document_id),
                    "tenant_id": tenant_id,
                    "content": chunk.content,
                    "chunk_index": idx,
                    "section_ref": chunk.section_ref,
                    "embedding": embedding,
                    "country": doc.country,
                    "framework": doc.framework,
                    "created_at": now,
                }
                for idx, (chunk, embedding) in enumerate(
                    zip(chunks_with_sections, embeddings)
                )
            ]
            _CHUNK_BATCH_SIZE = 50
            for batch_start in range(0, len(chunk_rows), _CHUNK_BATCH_SIZE):
                batch = chunk_rows[batch_start : batch_start + _CHUNK_BATCH_SIZE]
                self.db.table("document_chunks").insert(batch).execute()
            logger.debug(f"Saved {chunk_count} chunks to database")

            # --- Step 6: Mark ready ---
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
            return self.get_document(document_id)

        except Exception as e:
            logger.error(
                f"Document processing failed for {document_id}: {e}",
                exc_info=True,
            )
            self.db.table("documents").update({
                "status": DocumentStatus.FAILED.value,
                "error_message": str(e),
                "processed_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", str(document_id)).execute()

            if isinstance(e, (NotFoundError, DocumentProcessingError)):
                raise
            raise DocumentProcessingError(
                f"Document processing failed: {e}",
                resource_id=str(document_id),
            )

        finally:
            if tmp_file and os.path.exists(tmp_file):
                try:
                    os.unlink(tmp_file)
                except OSError as cleanup_exc:
                    logger.debug(
                        f"Tmp file cleanup failed (non-fatal): {cleanup_exc}"
                    )

    # ----------------------------------------------------------
    # list_documents
    # ----------------------------------------------------------
    def list_documents(self, tenant_id: str) -> list[Document]:
        """List all documents for a given tenant, newest first."""
        result = (
            self.db.table("documents")
            .select("*")
            .eq("tenant_id", tenant_id)
            .order("created_at", desc=True)
            .execute()
        )
        return [Document(**row) for row in result.data]

    # ----------------------------------------------------------
    # get_document
    # ----------------------------------------------------------
    def get_document(self, document_id: str, tenant_id: str | None = None) -> Document:
        """
        Retrieve a single document by ID. If tenant_id is provided,
        verifies ownership and raises TenantIsolationError otherwise.
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
        self.db.table("documents").update(
            {"status": status.value}
        ).eq("id", str(document_id)).execute()

    # ----------------------------------------------------------
    # Supabase Storage helpers
    #
    # The Supabase Python SDK's storage client surfaces different
    # exception shapes across versions; we normalise by raising
    # DocumentProcessingError so process_document() can report a
    # consistent error path.
    # ----------------------------------------------------------
    def _storage_upload(
        self,
        storage_path: str,
        file_bytes: bytes,
        content_type: str,
    ) -> None:
        """Upload bytes to the configured bucket. Overwrites if present."""
        # upsert=true so a retried upload after a partial write
        # replaces the previous object instead of failing.
        self.db.storage.from_(self._bucket).upload(
            path=storage_path,
            file=file_bytes,
            file_options={
                "content-type": content_type,
                "upsert": "true",
            },
        )

    def _storage_download(self, storage_path: str) -> bytes:
        """Download bytes for a storage object. Raises on missing file."""
        if not storage_path:
            raise DocumentProcessingError(
                "Document has no storage path — cannot process"
            )
        try:
            return self.db.storage.from_(self._bucket).download(storage_path)
        except Exception as exc:
            raise DocumentProcessingError(
                f"Storage download failed for '{storage_path}': {exc}"
            ) from exc
