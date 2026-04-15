# ============================================================
# app/schemas/common.py — Shared API response schemas
# ============================================================

from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """Standard error response shape for all API errors."""
    error: str
    message: str
    resource_id: str | None = None
    detail: dict | None = None


class PaginatedResponse(BaseModel):
    """Wrapper for paginated list responses."""
    total: int
    limit: int
    offset: int
    items: list
