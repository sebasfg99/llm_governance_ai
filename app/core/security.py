# ============================================================
# app/core/security.py — Authentication and token verification
#
# WHY THIS FILE EXISTS:
# Every API request needs to identify who is calling and which
# tenant they belong to. This file is the single place where
# token validation happens.
#
# CURRENT STATE (MVP):
# The verify_token function is STUBBED — it always returns the
# same hardcoded identity (dev-tenant / dev-user). This lets us
# build and test all features without a live Supabase auth setup.
#
# WHAT TO REPLACE THIS WITH IN PRODUCTION:
# Supabase issues JWTs (JSON Web Tokens) when users log in.
# Each JWT contains claims like:
#   - sub: the user's unique ID
#   - email: user's email
#   - user_metadata.tenant_id: the tenant this user belongs to
#
# To validate a Supabase JWT:
#   1. Get the JWT secret from Supabase Settings → API → JWT Secret
#   2. Use python-jose or PyJWT to decode and verify the token
#   3. Extract tenant_id and user_id from the claims
#
# TODO: Replace stub with real Supabase JWT validation
# TODO: Add token expiry check
# TODO: Add tenant_id claim extraction from JWT metadata
# ============================================================

from app.core.logging import get_logger

# Get a logger specific to this module
logger = get_logger(__name__)

# ============================================================
# Type alias for the identity extracted from a token.
# This dict is passed to route handlers via FastAPI Depends.
# ============================================================
# Shape: {"tenant_id": str, "user_id": str}
TokenPayload = dict[str, str]


def verify_token(token: str) -> TokenPayload:
    """
    Verify an authentication token and return the caller's identity.

    In the MVP, this always returns a hardcoded dev identity so
    all endpoints work without a real authentication setup.

    Args:
        token: The Bearer token from the Authorization header.
               Currently ignored in the stub implementation.

    Returns:
        A dict containing tenant_id and user_id extracted from the token.

    Raises:
        AuthenticationError: (future) If the token is invalid or expired.

    TODO: Implement real Supabase JWT validation:
        import jwt  # from PyJWT library
        from app.core.config import settings
        payload = jwt.decode(token, settings.supabase_jwt_secret, algorithms=["HS256"])
        return {
            "tenant_id": payload["user_metadata"]["tenant_id"],
            "user_id": payload["sub"],
        }
    """
    # Log that we're using the stub — this will remind developers during testing
    logger.debug(
        "Auth stub active — returning dev identity. "
        "Replace verify_token() in security.py before going to production."
    )

    # Return hardcoded development identity
    # All requests will appear to come from "dev-tenant" / "dev-user"
    # until real JWT validation is implemented
    return {
        "tenant_id": "dev-tenant",  # The tenant this user belongs to
        "user_id": "dev-user",      # The individual user making the request
    }


def extract_token_from_header(authorization: str | None) -> str:
    """
    Extract the Bearer token from an Authorization header value.

    Expected format: "Bearer <token>"

    Args:
        authorization: The raw Authorization header value.

    Returns:
        The token string (without the "Bearer " prefix).

    Raises:
        ValueError: If the header is missing or malformed.

    Example:
        token = extract_token_from_header("Bearer eyJhbGciOiJIUzI1NiJ9...")
    """
    if not authorization:
        raise ValueError("Authorization header is missing")

    # The header must start with "Bearer " (case-sensitive per RFC 6750)
    if not authorization.startswith("Bearer "):
        raise ValueError("Authorization header must use Bearer scheme")

    # Split on the first space and return the second part (the token itself)
    parts = authorization.split(" ", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError("Authorization header is malformed")

    return parts[1]
