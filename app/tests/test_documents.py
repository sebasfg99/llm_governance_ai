# ============================================================
# app/tests/test_documents.py — Document service unit tests
# ============================================================

import pytest
import os
import tempfile
from unittest.mock import MagicMock, patch

from app.domain.documents.chunking import chunk_text


# ============================================================
# Chunking tests (pure Python — no mocks needed)
# ============================================================

class TestChunking:
    """Tests for the fixed-size chunking algorithm."""

    def test_basic_chunking(self):
        """A long text should be split into multiple chunks."""
        # Create a text with 150 words
        text = " ".join([f"word{i}" for i in range(150)])
        chunks = chunk_text(text, chunk_size=50, chunk_overlap=10)

        assert len(chunks) > 1
        # Each chunk should have at most 50 words
        for chunk in chunks:
            assert len(chunk.split()) <= 50

    def test_overlap_is_present(self):
        """Adjacent chunks should share some words (the overlap)."""
        words = [f"w{i}" for i in range(100)]
        text = " ".join(words)
        chunks = chunk_text(text, chunk_size=20, chunk_overlap=5)

        if len(chunks) >= 2:
            # The last words of chunk 0 should appear at the start of chunk 1
            chunk0_words = set(chunks[0].split())
            chunk1_words = set(chunks[1].split())
            overlap = chunk0_words & chunk1_words
            assert len(overlap) > 0, "Adjacent chunks should have overlapping words"

    def test_short_text_produces_single_chunk(self):
        """Text shorter than chunk_size should produce exactly one chunk."""
        text = "This is a short text with only a few words."
        chunks = chunk_text(text, chunk_size=500)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_empty_text_returns_empty_list(self):
        """Empty input should return an empty list, not an error."""
        chunks = chunk_text("")
        assert chunks == []

    def test_whitespace_only_returns_empty_list(self):
        """Whitespace-only input should return an empty list."""
        chunks = chunk_text("   \n\t  ")
        assert chunks == []


# ============================================================
# Grader tests (from evals domain — pure Python)
# ============================================================

class TestGraders:
    """Tests for the heuristic quality graders."""

    def test_grounding_score_high_overlap(self):
        """Response using words from context should have a high grounding score."""
        from app.domain.evals.graders import grounding_score

        context = ["The contract requires payment within 30 days of invoice receipt."]
        response = "Payment must be made within 30 days after receiving the invoice."

        score = grounding_score(response, context)
        assert score > 0.1, f"Expected high grounding score, got {score}"

    def test_grounding_score_zero_for_unrelated(self):
        """Response with no overlap with context should have near-zero score."""
        from app.domain.evals.graders import grounding_score

        context = ["The sky is blue and clouds are white."]
        response = "Machine learning models require large datasets for training."

        score = grounding_score(response, context)
        assert score < 0.15

    def test_length_ok_normal_response(self):
        """A normal-length response should pass the length check."""
        from app.domain.evals.graders import length_ok

        response = " ".join(["word"] * 100)  # 100 words
        assert length_ok(response) is True

    def test_length_ok_too_short(self):
        """A very short response should fail the length check."""
        from app.domain.evals.graders import length_ok

        response = "I don't know."
        assert length_ok(response, min_words=10) is False

    def test_has_citations_with_numbered(self):
        """Response with [1] style citations should be detected."""
        from app.domain.evals.graders import has_citations

        response = "According to the contract [1], payment is due within 30 days."
        assert has_citations(response) is True

    def test_has_citations_without(self):
        """Response with no citations should return False."""
        from app.domain.evals.graders import has_citations

        response = "The payment is due within 30 days."
        assert has_citations(response) is False


# ============================================================
# DocumentService upload test (with mocked DB and filesystem)
# ============================================================

class TestDocumentService:
    """Tests for the DocumentService upload and processing logic."""

    def test_upload_creates_db_record_and_file(self, tmp_path):
        """upload_document should save the file and insert a DB record."""
        from app.domain.documents.service import DocumentService

        # Mock the Supabase client
        mock_db = MagicMock()
        mock_db.table.return_value.insert.return_value.execute.return_value.data = [
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "tenant_id": "test-tenant",
                "filename": "test.pdf",
                "file_type": "pdf",
                "file_path": str(tmp_path / "test-tenant" / "test.pdf"),
                "status": "pending",
                "error_message": None,
                "page_count": None,
                "chunk_count": None,
                "created_at": "2024-01-01T00:00:00+00:00",
                "processed_at": None,
            }
        ]

        # Patch the upload_dir setting to use tmp_path
        with patch("app.domain.documents.service.settings") as mock_settings:
            mock_settings.upload_dir = str(tmp_path)
            mock_settings.chunk_size = 500
            mock_settings.chunk_overlap = 50

            service = DocumentService(mock_db)
            doc = service.upload_document(
                tenant_id="test-tenant",
                filename="test.pdf",
                file_bytes=b"%PDF fake content",
            )

        assert doc.tenant_id == "test-tenant"
        assert doc.filename == "test.pdf"
        assert doc.status.value == "pending"

    def test_unsupported_file_type_raises_error(self, tmp_path):
        """Uploading an unsupported file type should raise DocumentProcessingError."""
        from app.domain.documents.service import DocumentService
        from app.core.errors import DocumentProcessingError

        mock_db = MagicMock()

        with patch("app.domain.documents.service.settings") as mock_settings:
            mock_settings.upload_dir = str(tmp_path)

            service = DocumentService(mock_db)

            with pytest.raises(DocumentProcessingError) as exc_info:
                service.upload_document(
                    tenant_id="test-tenant",
                    filename="spreadsheet.xlsx",
                    file_bytes=b"fake content",
                )

            assert "xlsx" in str(exc_info.value).lower() or "unsupported" in str(exc_info.value).lower()
