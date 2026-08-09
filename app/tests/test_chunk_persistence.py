# ============================================================
# app/tests/test_chunk_persistence.py — Durable chunk writes
#
# Regression tests for the incident of 2026-07-29, where a
# seed_documents.py --force run lost two documents to transient
# network errors mid-insert:
#
#   EU AI Act (493 chunks) kept exactly 200 — four batches of 50,
#   the fifth dying on "Server disconnected".
#   GDPR (189 chunks) kept zero after a "Broken pipe".
#
# Both had processed cleanly in June. Three defects combined:
#
#   1. no retry, so one dropped connection ended the run;
#   2. the previous chunks were DELETED before their replacements
#      were written, so a failure destroyed the working version;
#   3. chunk_count was only written on the success path, leaving
#      both documents advertising 493 and 189 while holding 200
#      and 0 rows.
#
# The tests below pin each of those three.
# ============================================================

import pytest

from app.core.errors import DocumentProcessingError
from app.domain.documents.service import (
    DocumentService,
    _is_transient_network_error,
)


# ------------------------------------------------------------
# A minimal in-memory stand-in for the Supabase client
#
# MagicMock cannot express this flow — the assertions are about
# what SURVIVES in a table across a partial failure, so the fake
# keeps real state and applies real inserts and deletes.
# ------------------------------------------------------------
class FakeAPIError(Exception):
    """Mimics postgrest's APIError, matched by class NAME in the code."""


APIError = FakeAPIError
FakeAPIError.__name__ = "APIError"


class _Result:
    def __init__(self, data, count=None):
        self.data = data
        self.count = count


class _Query:
    def __init__(self, table, op, payload=None, count_mode=None):
        self.table = table
        self.op = op
        self.payload = payload
        self.count_mode = count_mode
        self.filters = []

    def eq(self, field, value):
        self.filters.append(("eq", field, value))
        return self

    def in_(self, field, values):
        self.filters.append(("in", field, list(values)))
        return self

    def single(self):
        return self

    def _matches(self, row):
        for kind, field, value in self.filters:
            if kind == "eq" and str(row.get(field)) != str(value):
                return False
            if kind == "in" and row.get(field) not in value:
                return False
        return True

    def execute(self):
        rows = self.table.rows
        if self.op == "select":
            hits = [r for r in rows if self._matches(r)]
            return _Result(list(hits), count=len(hits) if self.count_mode else None)
        if self.op == "insert":
            self.table.on_insert(self.payload)
            rows.extend(dict(r) for r in self.payload)
            return _Result(list(self.payload))
        if self.op == "delete":
            kept = [r for r in rows if not self._matches(r)]
            removed = len(rows) - len(kept)
            self.table.rows[:] = kept
            return _Result([], count=removed)
        if self.op == "update":
            for row in rows:
                if self._matches(row):
                    row.update(self.payload)
            return _Result([])
        raise AssertionError(f"unsupported op {self.op}")


class FakeTable:
    def __init__(self, rows=None):
        self.rows = rows or []
        self.insert_calls = 0
        self._fail_plan = {}

    def fail_insert_on(self, call_number, exc):
        """Make the Nth insert call raise `exc`, once. Models a blip
        that the retry is expected to ride out."""
        self._fail_plan[call_number] = exc

    def fail_insert_from(self, call_number, exc):
        """Make every insert call from the Nth onward raise `exc`.
        Models the incident itself: the connection went and stayed
        broken, so no number of retries recovers."""
        self._fail_from = (call_number, exc)

    def on_insert(self, payload):
        self.insert_calls += 1
        persistent = getattr(self, "_fail_from", None)
        if persistent and self.insert_calls >= persistent[0]:
            raise persistent[1]
        exc = self._fail_plan.get(self.insert_calls)
        if exc is not None:
            raise exc

    def select(self, *_a, **kw):
        return _Query(self, "select", count_mode=kw.get("count"))

    def insert(self, payload):
        return _Query(self, "insert", payload=payload)

    def delete(self):
        return _Query(self, "delete")

    def update(self, payload):
        return _Query(self, "update", payload=payload)


class FakeDB:
    def __init__(self, documents, chunks=None):
        self.tables = {
            "documents": FakeTable(documents),
            "document_chunks": FakeTable(chunks or []),
        }

    def table(self, name):
        return self.tables[name]


DOC_ID = "dc497191-e052-4ea7-8526-d43a6804d4e7"


def _document_row(**overrides):
    row = {
        "id": DOC_ID,
        "tenant_id": "global",
        "filename": "OJ_L_202401689_ES_TXT.pdf",
        "file_type": "pdf",
        "file_path": f"global/{DOC_ID}/OJ_L_202401689_ES_TXT.pdf",
        "status": "ready",
        "country": None,
        "error_message": None,
        "page_count": 144,
        "chunk_count": 189,
        "created_at": "2026-06-13T16:52:18+00:00",
        "processed_at": "2026-06-13T16:55:00+00:00",
    }
    row.update(overrides)
    return row


def _old_chunks(n):
    """A previously-good generation of chunks, as June left them."""
    return [
        {
            "id": f"old-{i}",
            "document_id": DOC_ID,
            "tenant_id": "global",
            "content": f"contenido previo {i}",
            "chunk_index": i,
        }
        for i in range(n)
    ]


@pytest.fixture
def service_factory(monkeypatch):
    """
    Build a DocumentService whose parse/embed steps are stubbed, so a
    test only has to describe how the DATABASE behaves.
    """
    def _build(db, n_chunks=10):
        monkeypatch.setattr(
            "app.domain.documents.service.DocumentService._storage_download",
            lambda self, path: b"%PDF fake",
        )
        monkeypatch.setattr(
            "app.domain.documents.service.parse_pdf",
            lambda path: ("texto " * 4000, 144),
        )
        monkeypatch.setattr(
            "app.domain.documents.service.chunk_text_with_sections",
            lambda text: [
                type("C", (), {"content": f"nuevo {i}", "section_ref": None,
                               "is_reference_list": False,
                               "is_front_matter": False})()
                for i in range(n_chunks)
            ],
        )
        monkeypatch.setattr(
            "app.domain.documents.service.embed_texts",
            lambda texts: [[0.1] * 1536 for _ in texts],
        )
        # No real sleeping between retries.
        monkeypatch.setattr("app.domain.documents.service.time.sleep", lambda s: None)
        return DocumentService(db)

    return _build


# ------------------------------------------------------------
# 1. Which errors are worth retrying
# ------------------------------------------------------------
class TestTransientErrorDetection:
    @pytest.mark.parametrize(
        "exc",
        [
            Exception("Server disconnected"),          # the EU AI Act failure
            Exception("[Errno 32] Broken pipe"),       # the GDPR failure
            Exception("Connection reset by peer"),
            Exception("The read operation timed out"),
            ConnectionError("connection aborted"),
            BrokenPipeError(32, "Broken pipe"),
        ],
    )
    def test_dropped_connections_are_transient(self, exc):
        """
        Expected input:  the transport failures actually observed.
        Expected output: True — worth retrying.
        """
        assert _is_transient_network_error(exc) is True

    def test_server_rejections_are_not_retried(self):
        """
        Expected input:  an APIError, i.e. the server answered and said no.
        Expected output: False. A rejected payload fails identically every
                         time; retrying only delays the real error and
                         triples the write attempts.
        """
        assert _is_transient_network_error(FakeAPIError("duplicate key")) is False

    def test_unrelated_errors_are_not_retried(self):
        assert _is_transient_network_error(ValueError("bad chunk")) is False


# ------------------------------------------------------------
# 2. Retry
# ------------------------------------------------------------
class TestInsertRetry:
    def test_transient_failure_is_retried_and_succeeds(self, service_factory):
        """
        Expected input:  the 5th insert batch fails once with the exact
                         error that truncated the EU AI Act, then recovers.
        Expected output: all chunks persisted, document ready. Under the
                         old code this run ended at 200 of 493.
        """
        db = FakeDB([_document_row()])
        db.tables["document_chunks"].fail_insert_on(5, Exception("Server disconnected"))
        service = service_factory(db, n_chunks=250)  # 5 batches of 50

        service.process_document(DOC_ID)

        assert len(db.tables["document_chunks"].rows) == 250
        doc = db.tables["documents"].rows[0]
        assert doc["status"] == "ready"
        assert doc["chunk_count"] == 250

    def test_retries_are_bounded(self, service_factory):
        """
        Expected input:  a batch that fails transiently on every attempt.
        Expected output: the failure surfaces rather than looping forever.
        """
        db = FakeDB([_document_row()])
        db.tables["document_chunks"].fail_insert_from(1, Exception("timed out"))
        service = service_factory(db, n_chunks=10)

        with pytest.raises(DocumentProcessingError):
            service.process_document(DOC_ID)

        # 3 attempts on the single batch, then give up.
        assert db.tables["document_chunks"].insert_calls == 3


# ------------------------------------------------------------
# 3. A failed run must not destroy the working version
# ------------------------------------------------------------
class TestFailedRunIsNonDestructive:
    def test_previous_chunks_survive_a_mid_insert_failure(self, service_factory):
        """
        Expected input:  a document with 189 good chunks; reprocessing
                         dies partway through writing the new generation.
        Expected output: the 189 originals are still there and no new
                         chunks remain. This is the GDPR case, which the
                         old delete-first order took to zero permanently.
        """
        db = FakeDB([_document_row()], chunks=_old_chunks(189))
        db.tables["document_chunks"].fail_insert_from(3, Exception("Broken pipe"))
        service = service_factory(db, n_chunks=250)

        with pytest.raises(DocumentProcessingError):
            service.process_document(DOC_ID)

        remaining = db.tables["document_chunks"].rows
        assert len(remaining) == 189
        assert all(r["id"].startswith("old-") for r in remaining), (
            "a failed reprocess must leave the previous generation intact"
        )

    def test_successful_run_retires_the_previous_generation(self, service_factory):
        """
        Expected input:  a document with 189 old chunks, reprocessed cleanly.
        Expected output: exactly the 10 new chunks — no duplicates left over
                         from the old generation.
        """
        db = FakeDB([_document_row()], chunks=_old_chunks(189))
        service = service_factory(db, n_chunks=10)

        service.process_document(DOC_ID)

        rows = db.tables["document_chunks"].rows
        assert len(rows) == 10
        assert not any(r["id"].startswith("old-") for r in rows)


# ------------------------------------------------------------
# 4. chunk_count must never overstate reality
# ------------------------------------------------------------
class TestChunkCountTruthfulness:
    def test_failure_rewrites_chunk_count_to_the_real_value(self, service_factory):
        """
        Expected input:  a document advertising chunk_count=189 whose
                         reprocess fails with no previous chunks left.
        Expected output: status failed AND chunk_count corrected to 0.
                         Leaving the stale 189 is precisely what made the
                         corpus look healthy while GDPR was empty.
        """
        db = FakeDB([_document_row(chunk_count=189)], chunks=[])
        db.tables["document_chunks"].fail_insert_from(1, Exception("Server disconnected"))
        service = service_factory(db, n_chunks=10)

        with pytest.raises(DocumentProcessingError):
            service.process_document(DOC_ID)

        doc = db.tables["documents"].rows[0]
        assert doc["status"] == "failed"
        assert doc["chunk_count"] == 0

    def test_chunk_count_records_verified_rows_not_intended_rows(
        self, service_factory
    ):
        """
        Expected input:  a clean run of 120 chunks.
        Expected output: chunk_count equals the rows actually counted back
                         from the database, not the chunker's intent.
        """
        db = FakeDB([_document_row(chunk_count=None)])
        service = service_factory(db, n_chunks=120)

        service.process_document(DOC_ID)

        doc = db.tables["documents"].rows[0]
        assert doc["chunk_count"] == len(db.tables["document_chunks"].rows) == 120
