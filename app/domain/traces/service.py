# ============================================================
# app/domain/traces/service.py — Trace logging service
#
# WHY THIS FILE EXISTS:
# Every AI response must be fully traceable. This service writes
# a complete record of each request-response cycle to the DB.
#
# A trace is written AFTER the LLM responds, BEFORE returning
# to the caller. It must always succeed — if tracing fails,
# we log the error but still return the AI response to the user.
# ============================================================

import json
from datetime import datetime, timezone

from supabase import Client

from app.core.logging import get_logger
from app.core.errors import NotFoundError
from app.domain.traces.models import Trace, CreateTraceRequest

logger = get_logger(__name__)


class TraceService:
    """
    Writes and reads AI request-response trace records.

    Usage:
        service = TraceService(db)
        trace = service.create_trace(CreateTraceRequest(...))
        print(f"Trace saved: {trace.id}")
    """

    def __init__(self, db: Client) -> None:
        self.db = db

    def create_trace(self, request: CreateTraceRequest) -> Trace:
        """
        Write a complete trace record to the database.

        Called synchronously after every AI response.
        If this fails, the failure is logged but not raised —
        the AI response still reaches the user.

        Args:
            request: All the data about this AI request/response cycle.

        Returns:
            The persisted Trace record with its generated ID.
        """
        logger.debug(
            f"Creating trace: tenant={request.tenant_id}, "
            f"user={request.user_id}, model={request.model_provider}/{request.model_name}"
        )

        result = self.db.table("traces").insert({
            "tenant_id": request.tenant_id,
            "user_id": request.user_id,
            "user_query": request.user_query,
            "prompt_text": request.prompt_text,
            "use_case": request.use_case,
            "model_provider": request.model_provider,
            "model_name": request.model_name,
            # Store the list of chunk IDs as JSON
            "retrieved_chunk_ids": json.dumps(request.retrieved_chunk_ids),
            "retrieved_content": request.retrieved_content,
            "response_text": request.response_text,
            "response_mode": request.response_mode,
            "latency_ms": request.latency_ms,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }).execute()

        if not result.data:
            raise RuntimeError("Failed to persist trace record")

        trace = Trace(**result.data[0])
        logger.debug(f"Trace created: id={trace.id}")
        return trace

    def get_trace(self, trace_id: str) -> Trace:
        """
        Retrieve a single trace by its ID.

        Args:
            trace_id: UUID string of the trace.

        Returns:
            The Trace record.

        Raises:
            NotFoundError: If no trace with that ID exists.
        """
        result = (
            self.db.table("traces")
            .select("*")
            .eq("id", trace_id)
            .execute()
        )

        if not result.data:
            raise NotFoundError(f"Trace not found: {trace_id}", resource_id=trace_id)

        return Trace(**result.data[0])

    def list_traces(
        self,
        tenant_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[Trace]:
        """
        List traces for a tenant, newest first.

        Args:
            tenant_id: Filter to this tenant's traces only.
            limit:     Max number of records to return.
            offset:    Skip this many records (for pagination).

        Returns:
            List of Trace records.
        """
        result = (
            self.db.table("traces")
            .select("*")
            .eq("tenant_id", tenant_id)            # Tenant isolation
            .order("created_at", desc=True)         # Newest first
            .range(offset, offset + limit - 1)      # Pagination
            .execute()
        )

        return [Trace(**row) for row in result.data]
