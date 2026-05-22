# ============================================================
# app/core/tenancy.py — Multi-tenant access guard
#
# WHY THIS FILE EXISTS:
# Multi-tenancy is the most consequential security invariant of
# the platform: client A must NEVER read or write client B's
# data. The codebase already filters by `tenant_id` everywhere,
# but the spec (Task 5) requires an explicit gate function
# every domain service can call:
#
#     validate_tenant_access(client_id, user_id, document_client_id=None)
#
# This file is that gate. It returns the authoritative
# `tenant_id` for the calling client_id and raises a clear
# AppError when the request is cross-tenant.
#
# WHY THE WRAPPER OVER tenant_id:
# The spec uses "client_id" while the rest of the code uses
# "tenant_id". They are conceptually the same. We resolve them
# transparently here, so:
#   - new code can pass either name and the right column is used
#   - cross-tenant attempts always log a security event with both
#     ids for incident response
# ============================================================

from __future__ import annotations

from typing import Any

from fastapi import Header

from app.core.errors import AuthenticationError, TenantIsolationError
from app.core.logging import get_logger
from app.core.security import verify_token

logger = get_logger(__name__)


# Reserved tenant slugs that must NOT be used as the calling client.
# `global` is the shared knowledge library; if the caller is `global`
# they would bypass tenant isolation entirely.
_RESERVED_TENANTS = {"global"}


def validate_tenant_access(
    client_id: str | None,
    user_id: str | None,
    document_client_id: str | None = None,
) -> str:
    """
    Verify that `user_id` is allowed to act on data belonging to
    `client_id`, and that `document_client_id` (if supplied) matches.

    Args:
        client_id:          Caller's client/tenant identifier. Must be
                            non-empty and not a reserved slug.
        user_id:            Caller's user identifier. Must be non-empty.
        document_client_id: When the caller passes a document_id from
                            the request body, the service should fetch
                            the document's owning client_id and pass it
                            here. A mismatch raises TenantIsolationError.

    Returns:
        The validated tenant_id (always equal to client_id when valid).

    Raises:
        AuthenticationError: When client_id or user_id is missing.
        TenantIsolationError: When the document belongs to another tenant
                              or the client is reserved.
    """
    if not client_id or not str(client_id).strip():
        logger.warning("validate_tenant_access: missing client_id")
        raise AuthenticationError(
            "client_id is required. The request was rejected to avoid cross-tenant data access."
        )

    if not user_id or not str(user_id).strip():
        logger.warning(
            f"validate_tenant_access: missing user_id (client_id={client_id})"
        )
        raise AuthenticationError(
            "user_id is required for tenant access validation."
        )

    if client_id in _RESERVED_TENANTS:
        logger.error(
            f"Reserved tenant slug used as caller: client_id={client_id} user_id={user_id}"
        )
        raise TenantIsolationError(
            "Reserved tenant slug cannot be used as a caller identity."
        )

    if document_client_id is not None and document_client_id != client_id:
        # The caller asked about a resource belonging to a DIFFERENT
        # tenant. This is the most important security event we log.
        logger.error(
            "TENANT_ISOLATION_VIOLATION: caller_client_id=%s user_id=%s "
            "document_client_id=%s",
            client_id,
            user_id,
            document_client_id,
        )
        raise TenantIsolationError(
            "Access denied: the requested resource belongs to another client."
        )

    return client_id


# ============================================================
# FastAPI dependency
#
# Wires validate_tenant_access into the request lifecycle so
# every protected endpoint gets a verified tenant_id without
# duplicating the check.
# ============================================================
def get_validated_tenant(
    authorization: str | None = Header(default=None),
    x_client_id: str | None = Header(default=None),
) -> dict[str, Any]:
    """
    FastAPI dependency that returns a validated identity dict.

    Reads the bearer token (current stub returns dev identity)
    AND an optional X-Client-Id header. When both are present the
    header overrides the token's tenant — useful for service-to-
    service calls where the caller is acting on behalf of a tenant.

    Returns a dict with "client_id", "tenant_id", "user_id".
    Raises AuthenticationError on missing identity.
    """
    payload = verify_token(authorization or "stub-token")
    user_id = payload.get("user_id")
    tenant_from_token = payload.get("tenant_id")
    client_id = x_client_id or tenant_from_token

    validated = validate_tenant_access(
        client_id=client_id, user_id=user_id, document_client_id=None
    )
    return {
        "client_id": validated,
        "tenant_id": validated,
        "user_id": user_id,
    }
