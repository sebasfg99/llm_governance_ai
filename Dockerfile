# ============================================================
# Dockerfile — Container image for the ai-core FastAPI app
#
# WHY: Packages the app into a portable, reproducible image
# that can run anywhere Docker is available (local, Railway,
# Cloud Run, CI).
#
# BUILD:  docker build -t aicore .
# RUN (local):       docker run -p 8000:8000 --env-file .env aicore
# RUN (cloud-run):   gcloud run deploy ai-core --source . --port 8080
#
# CLOUD-RUN / RAILWAY COMPATIBILITY:
#   - $PORT is injected by the host (Railway: 8080, Cloud Run: 8080).
#     The CMD below honors it via ${PORT:-8000} so local docker
#     still binds 8000 when PORT is unset.
#   - The image is stateless: there is no UPLOADS volume. Document
#     originals live in Supabase Storage (see app/domain/documents/
#     service.py); nothing in this image needs durable disk.
#   - /health (liveness) and /ready (readiness) are both wired in
#     app/main.py so the probes match what the platform expects.
#
# STAGES:
#   1. builder — installs Python deps (cached layer)
#   2. runtime — copies app code and runs uvicorn
# ============================================================

# ---- Stage 1: builder ----------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /app

# Copy only requirements first so the dep layer is cached when
# only application source changes.
COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: runtime ----------------------------------------
FROM python:3.11-slim AS runtime

# No .pyc files in containers (they're recompiled on every run).
ENV PYTHONDONTWRITEBYTECODE=1

# Force unbuffered stdout so Cloud Run / Railway logs surface
# the moment the app prints them.
ENV PYTHONUNBUFFERED=1

# Default port. Overridden by the platform's $PORT in production
# (Cloud Run + Railway both inject it). Local docker runs ignore
# it because we bind via `-p 8000:8000`.
ENV PORT=8000

WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code only — no uploads/ dir, no .env
COPY app/ ./app/

# Expose the default port. Cloud Run / Railway ignore EXPOSE and
# rely on $PORT instead, but Docker desktop + local CI surfaces
# expect it.
EXPOSE 8000

# Run uvicorn against the platform-provided port. We use sh -c
# so $PORT expansion happens at container start, not image build.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
