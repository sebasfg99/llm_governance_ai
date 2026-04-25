# ============================================================
# app/main.py — FastAPI application entry point
#
# WHY THIS FILE EXISTS:
# This is the top-level file that creates the FastAPI app
# and wires everything together:
#   - Registers all API routers (routes)
#   - Registers error handlers
#   - Sets up middleware (CORS)
#   - Exposes the "app" object that uvicorn starts
#
# HOW TO RUN:
#   uvicorn app.main:app --reload
#   "app.main" = the file app/main.py
#   "app"      = the FastAPI() instance defined below
#
# WHAT TO IMPROVE LATER:
#   - Add rate limiting middleware
#   - Add request ID middleware for distributed tracing
#   - Add Prometheus metrics endpoint
# ============================================================

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Import core error registration
from app.core.errors import register_error_handlers

# Import all API route modules
from app.api.routes import (
    ai,
    documents,
    evals,
    feedback,
    health,
    policies,
    projects,
    prompts,
    risk,
    threads,
)

# ============================================================
# Create the FastAPI application instance
#
# title, description, version appear in the auto-generated
# API docs at http://localhost:8000/docs
# ============================================================
app = FastAPI(
    title="ai-core",
    description=(
        "Centralized AI engine for multi-tenant SaaS applications. "
        "Provides document ingestion, RAG-based retrieval, LLM routing, "
        "traceability, feedback collection, and evaluation."
    ),
    version="0.1.0",
    # Disable the /redoc endpoint if you only want /docs
    # redoc_url=None,
)

# ============================================================
# CORS Middleware
#
# CORS (Cross-Origin Resource Sharing) controls which domains
# can call this API from a browser.
#
# For development: allow all origins ("*")
# For production: replace "*" with your frontend's actual domain,
# e.g. ["https://app.yourproduct.com"]
#
# TODO: Read allowed origins from settings.ALLOWED_ORIGINS
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # Allow all origins in development
    allow_credentials=True,       # Allow cookies and auth headers
    allow_methods=["*"],          # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],          # Allow all request headers
)

# ============================================================
# Register API routers
#
# Each router groups related endpoints under a common prefix.
# We'll add more routers here as we build each domain phase.
# ============================================================

# Register custom error handlers (AppError → clean JSON responses)
register_error_handlers(app)

# --- Health check: GET /health ---
app.include_router(health.router, prefix="/health", tags=["Health"])

# --- Documents: upload, process, list, get ---
# POST /documents/upload
# POST /documents/{id}/process
# GET  /documents
# GET  /documents/{id}
app.include_router(documents.router, prefix="/documents", tags=["Documents"])

# --- AI queries: the central RAG endpoint ---
# POST /ai/query
app.include_router(ai.router, prefix="/ai", tags=["AI"])

# --- Prompt library: versioned system prompts ---
# GET  /prompts
# POST /prompts
app.include_router(prompts.router, prefix="/prompts", tags=["Prompts"])

# --- Feedback: user ratings on AI responses ---
# POST /feedback
app.include_router(feedback.router, prefix="/feedback", tags=["Feedback"])

# --- Evaluations: automatic quality checks ---
# GET /evals/{trace_id}
app.include_router(evals.router, prefix="/evals", tags=["Evaluations"])

# --- Projects: central "expediente" that risk, policies and chats attach to ---
# GET    /projects
# POST   /projects
# GET    /projects/{id}
# PATCH  /projects/{id}
app.include_router(projects.router, prefix="/projects", tags=["Projects"])

# --- Threads: conversation memory (L1) ---
# GET  /threads                      list the caller's threads
# POST /threads                      create an empty thread
# GET  /threads/{id}                 get a single thread
# GET  /threads/{id}/messages        replay messages in chronological order
app.include_router(threads.router, prefix="/threads", tags=["Threads"])

# --- Risk: intake interview (B2) + risk orchestrator (B3) ---
# POST /risk/{project_id}/intake
# POST /risk/{project_id}/analyze
app.include_router(risk.router, prefix="/risk", tags=["Risk"])

# --- Policies: policy generation (C1) + controls (C2) ---
# POST /policies/generate
# POST /policies/{project_id}/generate
# POST /policies/{project_id}/controls
app.include_router(policies.router, prefix="/policies", tags=["Policies"])


# ============================================================
# Global exception handler
#
# Catches any unhandled exception that bubbles up through the
# app and returns a clean JSON error response instead of a
# raw 500 Internal Server Error.
#
# Once app/core/errors.py is implemented (Phase 2), we'll
# register specific handlers for AppError subclasses here.
# ============================================================
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Catch-all handler for any exception not handled elsewhere.
    Returns a 500 JSON response instead of crashing or showing a stack trace.
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_server_error",
            "detail": "An unexpected error occurred. Please try again later.",
        },
    )


# ============================================================
# Startup event
#
# Code here runs once when the app starts.
# Useful for: DB connection checks, loading model weights, etc.
# ============================================================
@app.on_event("startup")
async def on_startup() -> None:
    """
    Application startup hook.

    Currently just prints a confirmation message.
    TODO: Add Supabase connectivity check once db/supabase.py is built.
    """
    print("ai-core starting up...")
    print("API docs available at: http://localhost:8000/docs")


# ============================================================
# Shutdown event
#
# Code here runs once when the app shuts down (Ctrl+C, SIGTERM).
# Useful for: closing DB connections, flushing log buffers, etc.
# ============================================================
@app.on_event("shutdown")
async def on_shutdown() -> None:
    """Application shutdown hook."""
    print("ai-core shutting down...")
