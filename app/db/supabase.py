# ============================================================
# app/db/supabase.py — Supabase database client
#
# WHY THIS FILE EXISTS:
# The Supabase client is initialized ONCE here and reused
# across the entire application. This is called the
# "singleton pattern" — creating one shared instance is more
# efficient than reconnecting on every request.
#
# WHAT THE SUPABASE CLIENT CAN DO:
#   - Query Postgres tables (like an ORM, but simpler)
#   - Call Postgres functions via RPC (used for vector search)
#   - Upload/download files in Supabase Storage
#   - Manage auth users (not used directly in this service)
#
# HOW OTHER MODULES USE THIS:
#   # As a direct import (in service classes):
#   from app.db.supabase import get_supabase_client
#   db = get_supabase_client()
#   result = db.table("documents").select("*").execute()
#
#   # As a FastAPI dependency (in route handlers):
#   from app.api.deps import get_db
#   @router.get("/documents")
#   def list_docs(db = Depends(get_db)):
#       ...
#
# TODO: Add connection health check on startup
# TODO: Consider connection pooling for high-traffic production use
# ============================================================

from supabase import create_client, Client
from app.core.config import settings
from app.core.logging import get_logger

# Logger for this module
logger = get_logger(__name__)

# ============================================================
# Module-level singleton
#
# _client is initialized to None and populated on first call
# to get_supabase_client(). This is "lazy initialization" —
# the connection is made only when first needed.
# ============================================================
_client: Client | None = None


def get_supabase_client() -> Client:
    """
    Return the initialized Supabase client instance.

    Uses lazy initialization: the client is created on the
    first call and reused for all subsequent calls.

    Returns:
        An initialized supabase.Client instance ready to use.

    Raises:
        RuntimeError: If the client cannot be initialized
                      (e.g., missing SUPABASE_URL or SUPABASE_SERVICE_KEY).

    Example:
        db = get_supabase_client()
        result = db.table("tenants").select("*").execute()
        tenants = result.data  # List of dicts
    """
    global _client  # Access the module-level variable

    # If client hasn't been created yet, create it now
    if _client is None:
        logger.info("Initializing Supabase client...")

        # Validate that required settings are present before attempting connection
        if not settings.supabase_url:
            raise RuntimeError("SUPABASE_URL environment variable is not set")
        if not settings.supabase_service_key:
            raise RuntimeError("SUPABASE_SERVICE_KEY environment variable is not set")

        # Create the Supabase client
        # create_client() sets up:
        #   - A PostgREST client for DB queries
        #   - An Auth client for user management
        #   - A Storage client for file operations
        _client = create_client(
            supabase_url=settings.supabase_url,
            supabase_key=settings.supabase_service_key,
        )

        logger.info(f"Supabase client initialized for: {settings.supabase_url}")

    return _client
