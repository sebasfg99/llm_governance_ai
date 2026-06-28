# ============================================================
# app/api/deps.py — FastAPI dependency injection functions
#
# WHY THIS FILE EXISTS:
# FastAPI uses a "Depends()" system to inject shared resources
# into route handlers. Instead of duplicating auth logic and
# DB access in every single route, we define them once here
# and inject them where needed.
#
# HOW IT WORKS:
# A dependency is just a Python function. FastAPI calls it
# automatically before calling the route handler, and passes
# the return value as an argument to the route.
#
# EXAMPLE USAGE IN A ROUTE:
#   from app.api.deps import get_current_user, get_db
#
#   @router.get("/documents")
#   def list_docs(
#       current_user: dict = Depends(get_current_user),
#       db = Depends(get_db),
#   ):
#       tenant_id = current_user["tenant_id"]
#       ...
# ============================================================

from fastapi import Depends, Header, HTTPException, status
from supabase import Client

from app.core.security import (
    verify_service_key,
    verify_supabase_jwt,
    TokenPayload,
)
from app.core.errors import AuthenticationError
from app.core.logging import get_logger
from app.core.tenancy import get_validated_tenant, validate_tenant_access
from app.db.supabase import get_supabase_client

logger = get_logger(__name__)


def get_db() -> Client:
    """
    FastAPI dependency that provides the Supabase database client.

    The client is a module-level singleton — the same instance
    is reused across all requests (no re-connection overhead).
    """
    return get_supabase_client()


def get_current_user(
    authorization: str | None = Header(default=None),
    x_tenant_id: str | None = Header(default=None, alias="X-Tenant-Id"),
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
) -> TokenPayload:
    """
    Authenticate the incoming request via either auth path.

    PATH A — Service key + trusted headers (GovAI Edge Function proxy):
        Authorization: Bearer <AI_CORE_SERVICE_KEY>
        X-Tenant-Id:   <institution_id::text>
        X-User-Id:     <auth.users.id>

      The Edge Function has already validated the end-user's Supabase
      JWT. The sidecar trusts the X-* headers iff the service key
      matches. Both headers are required — anything else is a
      misconfigured proxy and we 401 immediately rather than
      silently downgrade to anonymous.

    PATH B — Supabase JWT (direct caller):
        Authorization: Bearer <supabase-jwt>

      tenant_id is read from app_metadata.tenant_id. Used by ops
      scripts and any future caller that does not sit behind the
      Edge Function.

    Returns:
        {"tenant_id": str, "user_id": str}

    Raises:
        HTTPException 401: If neither path successfully authenticates.
    """
    # Path A: service-key + X-* headers
    if verify_service_key(authorization):
        if not x_tenant_id or not x_user_id:
            logger.warning(
                "Service-key auth succeeded but required headers missing "
                f"(tenant_id={bool(x_tenant_id)}, user_id={bool(x_user_id)})"
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Service-key auth requires X-Tenant-Id and X-User-Id headers",
            )
        return {"tenant_id": x_tenant_id, "user_id": x_user_id}

    # Path B: Supabase JWT
    jwt_claims = verify_supabase_jwt(authorization)
    if jwt_claims is not None:
        return jwt_claims

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Unauthorized",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ============================================================
# Tenant-validated identity (Task 5)
#
# get_validated_tenant() is implemented in app/core/tenancy.py.
# We re-export it here so route modules can import it from the
# usual deps location:
#
#   from app.api.deps import get_validated_tenant
#
# The dependency raises AuthenticationError or TenantIsolationError
# automatically when the caller does not pass a valid client_id.
# ============================================================
__all__ = [
    "get_db",
    "get_current_user",
    "get_validated_tenant",
    "validate_tenant_access",
]
