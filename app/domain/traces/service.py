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
            "project_id": request.project_id,
            "thread_id": request.thread_id,
            "user_query": request.user_query,
            "prompt_text": request.prompt_text,
            "use_case": request.use_case,
            "model_provider": request.model_provider,
            "model_name": request.model_name,
            "retrieved_chunk_ids": request.retrieved_chunk_ids,
            "retrieved_content": request.retrieved_content,
            "response_text": request.response_text,
            "response_mode": request.response_mode,
            "latency_ms": request.latency_ms,
            "created_at": datetime.now(timezone.utc).isoformat(),
            # Migration 006 governance columns. The DB trigger
            # will backfill client_id from tenant_id when None.
            "prompt_tokens": request.prompt_tokens,
            "completion_tokens": request.completion_tokens,
            "cost_estimate_usd": request.cost_estimate_usd,
            "risk_level": request.risk_level,
            "requires_disclaimer": request.requires_disclaimer,
            "client_id": request.client_id or request.tenant_id,
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

    def list_project_traces(
        self,
        project_id: str,
        tenant_id: str,
        use_cases: list[str] | None = None,
        limit: int = 10,
    ) -> list[Trace]:
        """
        Return the most recent traces for a project, optionally filtered by
        use_case. Used by AIService to reconstruct reasoning when the user
        asks "why did you do X?".

        We deliberately keep this cheap (no embedding / semantic match) in
        the v1: recency + use_case filter is enough to give the LLM the
        relevant context for a narrative justification. A semantic variant
        can be added later if recall proves insufficient.
        """
        query = (
            self.db.table("traces")
            .select("*")
            .eq("tenant_id", tenant_id)
            .eq("project_id", project_id)
            .order("created_at", desc=True)
            .limit(limit)
        )
        if use_cases:
            query = query.in_("use_case", use_cases)
        result = query.execute()
        return [Trace(**row) for row in (result.data or [])]
