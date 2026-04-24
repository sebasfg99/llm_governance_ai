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
    chunk_size: int = 300

    # How many tokens overlap between adjacent chunks.
    # Overlap prevents important context from being split across chunk boundaries.
    chunk_overlap: int = 30

    # How many chunks to retrieve per RAG query
    retrieval_top_k: int = 5

    # Minimum cosine similarity score to include a chunk in the LLM context.
    # Chunks below this threshold are discarded before prompt assembly.
    # Range: 0.0 (no filter) – 1.0 (exact match). Default: 0.45
    similarity_threshold: float = 0.45

    # Number of candidate chunks to fetch from pgvector before threshold filtering.
    # Fetching more candidates than top_k ensures there are enough chunks above the
    # threshold to fill the final top_k slots, AND gives the per-document diversity
    # cap enough pool to pull chunks from multiple documents. Default: 30 (6x top_k).
    retrieval_candidate_k: int = 30

    # Diversity cap: maximum number of chunks a single document can contribute
    # to the final top-k list. Prevents one document from monopolising the
    # context window (e.g. a Spanish doc outranking every English doc on a
    # Spanish query). Set to 0 or None to disable the cap.
    # With cap=1 the top-k list will contain k *distinct* documents, which is
    # the strongest defense against cross-language ranking bias and gives
    # lower-similarity but topically-relevant documents a fair shot at context.
    retrieval_max_chunks_per_document: int = 1

    # Cross-lingual query expansion. When True, every user query is translated
    # into both Spanish and English (one gpt-4o-mini call, ~$0.0001 per query)
    # and the vector search runs with both variants. Results are merged by
    # chunk_id keeping the max similarity, which closes the cross-lingual gap
    # between Spanish queries and English documents (OWASP, ISO, etc.).
    retrieval_multilingual: bool = True

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
