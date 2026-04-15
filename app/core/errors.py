# ============================================================
# app/core/errors.py — Custom exception hierarchy + handlers
#
# WHY THIS FILE EXISTS:
# We define our own exceptions instead of using raw Python
# exceptions so that:
#   1. Every error carries an HTTP status code
#   2. Error responses have a consistent JSON shape
#   3. We can raise domain-specific errors anywhere in the
#      code and the API layer handles them uniformly
#
# HOW IT WORKS:
# - Domain services raise AppError subclasses (e.g., NotFoundError)
# - FastAPI catches them via the registered exception_handler
# - The handler converts them to clean JSON responses
#
# HOW TO USE:
#   from app.core.errors import NotFoundError
#   raise NotFoundError("Document not found", resource_id="doc_123")
#
# REGISTERING HANDLERS (done in main.py):
#   from app.core.errors import register_error_handlers
#   register_error_handlers(app)
# ============================================================

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


# ============================================================
# Base exception class
# All our custom exceptions inherit from this.
# ============================================================

class AppError(Exception):
    """
    Base class for all application-level errors.

    Carries an HTTP status_code so the error handler can
    return the correct HTTP response without extra logic.

    Attributes:
        message:     Human-readable description of what went wrong
        status_code: HTTP status code to return (default 500)
        resource_id: Optional ID of the resource involved (for debugging)
        detail:      Optional dict with extra structured context
    """

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        resource_id: str | None = None,
        detail: dict | None = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.resource_id = resource_id
        self.detail = detail or {}


# ============================================================
# Specific error subclasses
# Add more as the domain grows.
# ============================================================

class NotFoundError(AppError):
    """
    Raised when a requested resource does not exist.
    Maps to HTTP 404 Not Found.

    Example:
        raise NotFoundError("Document not found", resource_id="doc_abc123")
    """

    def __init__(self, message: str = "Resource not found", resource_id: str | None = None) -> None:
        super().__init__(message, status_code=404, resource_id=resource_id)


class ValidationError(AppError):
    """
    Raised when input data fails business-logic validation.
    (Not to be confused with Pydantic's own ValidationError.)
    Maps to HTTP 422 Unprocessable Entity.

    Example:
        raise ValidationError("Prompt injection detected in user query")
    """

    def __init__(self, message: str = "Validation failed", detail: dict | None = None) -> None:
        super().__init__(message, status_code=422, detail=detail)


class UpstreamError(AppError):
    """
    Raised when an external API (OpenAI, Gemini, Supabase) fails.
    Maps to HTTP 502 Bad Gateway.

    Example:
        raise UpstreamError("OpenAI API returned an error", detail={"provider": "openai"})
    """

    def __init__(self, message: str = "Upstream service error", detail: dict | None = None) -> None:
        super().__init__(message, status_code=502, detail=detail)


class AuthenticationError(AppError):
    """
    Raised when a request is missing or has invalid credentials.
    Maps to HTTP 401 Unauthorized.

    Example:
        raise AuthenticationError("Invalid or expired token")
    """

    def __init__(self, message: str = "Authentication required") -> None:
        super().__init__(message, status_code=401)


class TenantIsolationError(AppError):
    """
    Raised when a tenant attempts to access another tenant's data.
    Maps to HTTP 403 Forbidden.

    This is a critical security error — always log it.

    Example:
        raise TenantIsolationError("Access denied: document belongs to a different tenant")
    """

    def __init__(self, message: str = "Access denied") -> None:
        super().__init__(message, status_code=403)


class DocumentProcessingError(AppError):
    """
    Raised when document parsing or embedding fails.
    Maps to HTTP 422 Unprocessable Entity.

    Example:
        raise DocumentProcessingError("Could not parse PDF", resource_id="doc_123")
    """

    def __init__(self, message: str = "Document processing failed", resource_id: str | None = None) -> None:
        super().__init__(message, status_code=422, resource_id=resource_id)


# ============================================================
# FastAPI exception handlers
# These convert our AppError exceptions into JSON HTTP responses.
# ============================================================

async def app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    """
    Handles any AppError (and all its subclasses) raised in route handlers.

    Returns a consistent JSON error response shape:
    {
        "error": "not_found",
        "message": "Document not found",
        "resource_id": "doc_123",  (if applicable)
        "detail": {}               (any extra context)
    }
    """
    # Build the response body
    body: dict = {
        # Convert the HTTP status to a snake_case error code
        # e.g. 404 → "not_found", 422 → "unprocessable_entity"
        "error": _status_to_error_code(exc.status_code),
        "message": exc.message,
    }

    # Only include optional fields if they have values
    if exc.resource_id:
        body["resource_id"] = exc.resource_id
    if exc.detail:
        body["detail"] = exc.detail

    return JSONResponse(status_code=exc.status_code, content=body)


def _status_to_error_code(status_code: int) -> str:
    """
    Convert an HTTP status code to a readable snake_case error code.
    Used in the error response body to give clients a stable identifier.
    """
    codes = {
        400: "bad_request",
        401: "unauthorized",
        403: "forbidden",
        404: "not_found",
        422: "unprocessable_entity",
        500: "internal_server_error",
        502: "bad_gateway",
    }
    # Default to "error" if we don't have a mapping
    return codes.get(status_code, "error")


def register_error_handlers(app: FastAPI) -> None:
    """
    Register all custom exception handlers on the FastAPI app.

    Call this in main.py during app initialization:
        from app.core.errors import register_error_handlers
        register_error_handlers(app)
    """
    # Register the base AppError handler — it covers all subclasses too
    app.add_exception_handler(AppError, app_error_handler)  # type: ignore[arg-type]
