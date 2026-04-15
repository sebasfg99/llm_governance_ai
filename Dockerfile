# ============================================================
# Dockerfile — Container image for the ai-core FastAPI app
#
# WHY: Packages the app into a portable, reproducible image
# that can run anywhere Docker is available (local, cloud, CI).
#
# BUILD:  docker build -t aicore .
# RUN:    docker run -p 8000:8000 --env-file .env aicore
#
# STAGES:
#   1. builder — installs Python deps (cached layer)
#   2. runtime — copies app code and runs uvicorn
#
# Using multi-stage keeps the final image small by not
# including build tools in the production image.
# ============================================================

# ---- Stage 1: builder ----------------------------------------
# Install dependencies in a separate stage so Docker can cache
# this layer. If requirements.txt hasn't changed, Docker skips
# this step entirely on rebuilds — much faster.
FROM python:3.11-slim AS builder

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first — this is the cache optimization.
# Docker rebuilds from here only if requirements.txt changes.
COPY requirements.txt .

# Install all Python dependencies
# --no-cache-dir: don't store pip cache in the image (saves space)
# --upgrade: ensure pip itself is up to date
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt


# ---- Stage 2: runtime ----------------------------------------
# Start fresh from the slim base image.
# Copy only what we need: installed packages + app code.
FROM python:3.11-slim AS runtime

# Prevent Python from writing .pyc files (not needed in containers)
ENV PYTHONDONTWRITEBYTECODE=1

# Prevent Python output from being buffered
# (ensures logs appear immediately in docker logs)
ENV PYTHONUNBUFFERED=1

# Set working directory
WORKDIR /app

# Copy installed Python packages from the builder stage
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY app/ ./app/

# Create the uploads directory so it exists in the container
# In production, mount a volume here instead of relying on this dir
RUN mkdir -p /app/uploads

# Expose the port FastAPI/uvicorn will listen on
EXPOSE 8000

# Start the app using uvicorn
# --host 0.0.0.0: listen on all interfaces (required inside Docker)
# --port 8000: the port we exposed above
# For production, remove --reload and add --workers 2 or more
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
