# ============================================================
# app/core/security.py — Authentication and token verification
#
# WHY THIS FILE EXISTS:
# Every API request needs to identify who is calling and which
# tenant they belong to. This file is the single place where
# token validation happens.
#
# TWO AUTH PATHS RUN SIDE BY SIDE:
#
#   Path A — Service key + trusted headers
#     Used by the Supabase Edge Function proxy. The Edge Function
#     has already validated the end user's Supabase JWT, so the
#     sidecar trusts X-Tenant-Id / X-User-Id headers when the
#     Authorization header carries the shared AI_CORE_SERVICE_KEY.
#
#   Path B — Supabase JWT
#     Used by direct clients that pass a Supabase auth JWT. The
#     sidecar decodes the JWT using SUPABASE_JWT_SECRET and
#     extracts tenant_id from app_metadata.
#
# Path A is the production default for GovAI's Edge Function
# integration. Path B exists so any future caller (mobile, ops
# scripts, partner integrations) can hit the sidecar without an
# Edge Function in front.
# ============================================================

import secrets
from typing import Any

import jwt

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# ============================================================
# Type alias for the identity extracted from a token.
# This dict is passed to route handlers via FastAPI Depends.
# ============================================================
# Shape: {"tenant_id": str, "user_id": str}
TokenPayload = dict[str, str]


def verify_service_key(authorization: str | None) -> bool:
    """
    Check if the Authorization header carries the shared service
    key issued to the GovAI Edge Function proxy.

    Uses `secrets.compare_digest` to avoid timing-attack leaks
    of the key length / prefix.

    Args:
        authorization: Raw `Authorization` request header value.

    Returns:
        True iff the header matches AI_CORE_SERVICE_KEY exactly.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return False

    configured = settings.ai_core_service_key
    if not configured:
        # Service-key auth disabled when the secret isn't configured.
        # We log once so deployments missing the env var don't fail
        # silently in a confusing way.
        logger.debug("Service-key auth skipped: AI_CORE_SERVICE_KEY is empty")
        return False

    token = authorization.removeprefix("Bearer ").strip()
    return secrets.compare_digest(token, configured)


def verify_supabase_jwt(authorization: str | None) -> TokenPayload | None:
    """
    Decode a Supabase JWT and return the caller's identity claims.

    Returns None on any failure (missing header, wrong scheme,
    expired, bad signature, missing tenant). Callers should treat
    None as "not authenticated via this path".

    Tenant ID is read from `app_metadata.tenant_id`, which is set
    by the GovAI Supabase auth hooks when an institution member
    logs in.

    Args:
        authorization: Raw `Authorization` request header value.

    Returns:
        {"tenant_id": str, "user_id": str} or None.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return None

    if not settings.supabase_jwt_secret:
        logger.debug("Supabase JWT auth skipped: SUPABASE_JWT_SECRET is empty")
        return None

    token = authorization.removeprefix("Bearer ").strip()

    try:
        payload: dict[str, Any] = jwt.decode(
            token,
            settings.supabase_jwt_secret,
            algorithms=["HS256"],
            audience="authenticated",
        )
    except jwt.PyJWTError as exc:
        logger.warning(f"Supabase JWT verification failed: {exc}")
        return None

    user_id = payload.get("sub")
    app_metadata = payload.get("app_metadata") or {}
    tenant_id = app_metadata.get("tenant_id") or payload.get("tenant_id")

    if not user_id or not tenant_id:
        logger.warning(
            "Supabase JWT missing required claims "
            f"(sub={bool(user_id)}, tenant_id={bool(tenant_id)})"
        )
        return None

    return {"tenant_id": str(tenant_id), "user_id": str(user_id)}


def extract_token_from_header(authorization: str | None) -> str:
    """
    Extract the Bearer token from an Authorization header value.

    Kept for backwards compatibility with callers that want the
    raw token (tests, debug tooling). Production auth paths above
    do this inline.

    Args:
        authorization: The raw Authorization header value.

    Returns:
        The token string (without the "Bearer " prefix).

    Raises:
        ValueError: If the header is missing or malformed.
    """
    if not authorization:
        raise ValueError("Authorization header is missing")
    if not authorization.startswith("Bearer "):
        raise ValueError("Authorization header must use Bearer scheme")

    parts = authorization.split(" ", 1)
    if len(parts) != 2 or not parts[1]:
        raise ValueError("Authorization header is malformed")

    return parts[1]


# Kept for backwards compatibility with tests / older callers that
# imported `verify_token` directly. New code should call
# `verify_service_key` / `verify_supabase_jwt` from app.api.deps.
def verify_token(token: str) -> TokenPayload:
    """
    Legacy adapter — verifies a raw token against the Supabase JWT
    path only. Service-key auth requires the header context and
    must use `verify_service_key` instead.

    Raises:
        ValueError: If the token cannot be verified as a JWT.
    """
    claims = verify_supabase_jwt(f"Bearer {token}")
    if claims is None:
        raise ValueError("Unable to verify token via Supabase JWT")
    return claims
