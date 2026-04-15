# ============================================================
# app/core/config.py — Centralized application configuration
#
# WHY THIS FILE EXISTS:
# All configuration values (API keys, URLs, feature flags) are
# read from environment variables in ONE place. This means:
#   - You never hardcode secrets in source code
#   - Changing a setting only requires updating .env
#   - Every other module imports from here — one source of truth
#
# HOW IT WORKS:
# pydantic-settings reads each field from environment variables
# automatically. The field name maps to the env var name.
# Example: SUPABASE_URL in .env → settings.supabase_url in code
#
# HOW TO USE IN OTHER FILES:
#   from app.core.config import settings
#   print(settings.supabase_url)
#
# TODO: Add field validators for URL format, key prefix checks, etc.
# ============================================================

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment variables.

    pydantic-settings automatically reads each field from the
    matching environment variable (case-insensitive).
    If a required variable is missing and has no default,
    the app will fail fast at startup with a clear error.
    """

    # ----------------------------------------------------------
    # Supabase connection
    # ----------------------------------------------------------

    # The URL of your Supabase project (e.g. https://xyz.supabase.co)
    supabase_url: str

    # The service role key — has full DB access, bypasses RLS.
    # NEVER expose this to the browser or end users.
    # TODO: In production, validate this starts with "eyJ" (JWT format)
    supabase_service_key: str

    # ----------------------------------------------------------
    # LLM provider API keys
    # ----------------------------------------------------------

    # OpenAI API key — used for both GPT completions and embeddings
    openai_api_key: str

    # Google Gemini API key — used for Gemini completions
    # Optional: the app won't crash if missing, but Gemini calls will fail
    gemini_api_key: str = ""

    # ----------------------------------------------------------
    # LLM Router defaults
    # These tell the router which provider and model to use by default.
    # ----------------------------------------------------------

    # Which LLM provider to use: "openai" or "gemini"
    llm_default_provider: str = "openai"

    # The specific model name for the chosen provider
    # OpenAI: gpt-4o, gpt-4o-mini, gpt-3.5-turbo
    # Gemini: gemini-1.5-pro, gemini-1.5-flash, gemini-2.0-flash
    llm_default_model: str = "gpt-4o-mini"

    # ----------------------------------------------------------
    # Embedding model settings
    # ----------------------------------------------------------

    # The OpenAI embedding model to use for converting text → vectors
    # text-embedding-3-small: 1536 dimensions, fast, cheap
    # text-embedding-3-large: 3072 dimensions, higher quality, more expensive
    embedding_model: str = "text-embedding-3-small"

    # Number of dimensions the embedding model produces.
    # Must match the vector(N) column type in the database.
    # If you change the model, you must also update the DB schema.
    embedding_dimensions: int = 1536

    # ----------------------------------------------------------
    # File storage
    # ----------------------------------------------------------

    # Local directory where uploaded documents are saved.
    # Files are organized as: {upload_dir}/{tenant_id}/{filename}
    # TODO: Replace with an S3 bucket path in production
    upload_dir: str = "./uploads"

    # ----------------------------------------------------------
    # Chunking settings for RAG document processing
    # ----------------------------------------------------------

    # How many tokens (approx. words) per text chunk
    chunk_size: int = 500

    # How many tokens overlap between adjacent chunks.
    # Overlap prevents important context from being split across chunk boundaries.
    chunk_overlap: int = 50

    # How many chunks to retrieve per RAG query
    retrieval_top_k: int = 5

    # ----------------------------------------------------------
    # Application settings
    # ----------------------------------------------------------

    # Logging verbosity: DEBUG, INFO, WARNING, ERROR, CRITICAL
    log_level: str = "INFO"

    # Deployment environment — used in logging and feature flags
    app_env: str = "development"

    # ----------------------------------------------------------
    # pydantic-settings configuration
    # ----------------------------------------------------------
    model_config = SettingsConfigDict(
        # Read from a .env file if it exists (silently ignored if not)
        env_file=".env",
        # Allow extra fields in .env without raising an error
        extra="ignore",
        # Make field names case-insensitive (SUPABASE_URL == supabase_url)
        case_sensitive=False,
    )


# ============================================================
# Singleton instance
#
# We create ONE Settings instance at module load time.
# All other modules import this single object:
#   from app.core.config import settings
#
# This is the standard pattern — it's loaded once and cached.
# ============================================================
settings = Settings()
