# ============================================================
# app/api/routes/health.py — Health check endpoint
#
# WHY: A health check is the simplest endpoint in the system.
# Load balancers, Docker, and Kubernetes use it to know
# whether the app is running and ready to accept requests.
#
# A healthy response does NOT mean all dependencies (DB, LLM)
# are working — it only means the process is alive.
# TODO: Add a "deep" health check that pings Supabase + OpenAI.
# ============================================================

from fastapi import APIRouter

# Create a router — a mini-application that groups related routes.
# We'll register this router in main.py with a prefix like "/health".
router = APIRouter()


@router.get(
    "",  # Empty string because the prefix in main.py already includes "/health"
    summary="Health check",
    description="Returns 200 OK if the application process is running.",
    tags=["Health"],  # Groups this endpoint in the /docs Swagger UI
)
def health_check() -> dict:
    """
    Simple liveness check.

    Returns a JSON object confirming the service is up.
    Useful for Docker healthchecks and load balancer probes.
    """
    return {
        "status": "ok",
        "version": "0.1.0",  # TODO: Read from a VERSION file or pyproject.toml
        "service": "ai-core",
    }
