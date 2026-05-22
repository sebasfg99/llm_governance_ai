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
#
# TODO: Add get_current_tenant() once tenants table is populated
# TODO: Add rate_limit() dependency for production throttling
# ============================================================

from fastapi import Depends, Header
from supabase import Client

from app.core.security import verify_token, extract_token_from_header, TokenPayload
from app.core.errors import AuthenticationError
from app.core.logging import get_logger
from app.core.tenancy import get_validated_tenant, validate_tenant_access
from app.db.supabase import get_supabase_client

# Logger for this module
logger = get_logger(__name__)


def get_db() -> Client:
    """
    FastAPI dependency that provides the Supabase database client.

    The client is a module-level singleton — the same instance
    is reused across all requests (no re-connection overhead).

    Returns:
        The initialized Supabase Client instance.

    Usage in a route:
        @router.get("/something")
        def my_route(db: Client = Depends(get_db)):
            result = db.table("documents").select("*").execute()
    """
    # get_supabase_client() returns the already-initialized singleton
    return get_supabase_client()


def get_current_user(
    authorization: str | None = Header(default=None)
) -> TokenPayload:
    """
    FastAPI dependency that authenticates the incoming request.

    Reads the Authorization header, extracts the Bearer token,
    verifies it, and returns the caller's identity.

    Args:
        authorization: Injected by FastAPI from the "Authorization" request header.
                       Value format: "Bearer <token>"

    Returns:
        A dict with "tenant_id" and "user_id" keys.

    Raises:
        AuthenticationError: If the header is missing or the token is invalid.

    NOTE: In the current MVP stub, any token (or no token) returns
    the dev identity. The real validation is in security.verify_token().
    """
    # For the MVP stub, we skip token extraction and just return dev identity.
    # This means ALL requests succeed regardless of the Authorization header.
    # TODO: Remove this bypass when real auth is implemented.
    if True:  # TODO: Change to "if authorization" when real auth is added
        return verify_token(authorization or "stub-token")

    # The code below is the REAL implementation pattern — currently unreachable.
    # It will be activated when the stub above is removed.
    try:
        # Extract the raw token string from "Bearer <token>"
        token = extract_token_from_header(authorization)
    except ValueError as e:
        # Re-raise as our application's AuthenticationError
        raise AuthenticationError(str(e))

    # Validate the token and extract identity claims
    try:
        return verify_token(token)
    except Exception as e:
        logger.warning(f"Token verification failed: {e}")
        raise AuthenticationError("Invalid or expired token")


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
