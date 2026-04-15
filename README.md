# ai-core — Centralized AI Engine

A production-oriented MVP of a centralized AI engine for multi-tenant SaaS applications. Uses RAG (Retrieval-Augmented Generation) to answer questions from your private documents.

```
                        ┌─────────────────────────────────────────────┐
                        │                 ai-core API                  │
                        │                                               │
  Chatbot / App ───────►│  POST /ai/query                              │
                        │    1. Load system prompt                      │
                        │    2. Retrieve relevant chunks (RAG)          │
                        │    3. Apply guardrails                        │
                        │    4. Call LLM (OpenAI or Gemini)             │
                        │    5. Save trace + eval                       │
                        │    6. Return response + trace_id              │
                        └──────────────┬──────────────────────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              │                        │                        │
       ┌──────▼──────┐         ┌───────▼──────┐        ┌───────▼──────┐
       │  Supabase   │         │   OpenAI     │        │   Gemini     │
       │  Postgres   │         │   GPT + EMB  │        │   (optional) │
       │  pgvector   │         └──────────────┘        └──────────────┘
       └─────────────┘
```

---

## Prerequisites

- Python 3.11+
- Docker Desktop (for local Postgres with pgvector)
- A Supabase account (free tier works for development)
- An OpenAI API key

---

## 1. Supabase Setup (first time only)

### Create a Supabase project

1. Go to [supabase.com](https://supabase.com) and sign up for a free account
2. Click **New project**
3. Choose a name, set a database password, and select a region close to you
4. Wait ~2 minutes for provisioning

### Get your credentials

1. In your project dashboard, click **Settings** → **API**
2. Copy:
   - **Project URL** → this is your `SUPABASE_URL`
   - **service_role** key (under "Project API keys") → this is your `SUPABASE_SERVICE_KEY`

> **Warning**: The service_role key has full database access. Never expose it in client-side code or commit it to git.

### Run the database migration

1. In your Supabase project, click **SQL Editor** in the left sidebar
2. Click **New query**
3. Copy the entire contents of `app/db/migrations/001_initial.sql`
4. Paste it into the SQL editor and click **Run**

You should see: `Success. No rows returned`

### Enable pgvector (if not already enabled)

The migration script runs `CREATE EXTENSION IF NOT EXISTS vector;` automatically. Verify it worked:

```sql
SELECT * FROM pg_extension WHERE extname = 'vector';
```

You should see one row. If not, go to **Database** → **Extensions** and enable `vector`.

---

## 2. Local Development Setup

### Clone and install dependencies

```bash
# Clone the repository
git clone <your-repo-url>
cd llm_ai_governance_project

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configure environment variables

```bash
# Copy the example file
cp .env.example .env

# Edit .env and fill in your values
# Required: SUPABASE_URL, SUPABASE_SERVICE_KEY, OPENAI_API_KEY
# Optional: GEMINI_API_KEY (only needed if using Gemini)
```

### Start local Postgres with pgvector (optional)

Use this if you want a fully local setup instead of Supabase cloud:

```bash
# Start Postgres with pgvector in Docker
docker compose up -d

# Verify it's running
docker compose ps

# Connect and apply migrations
psql -h localhost -U postgres -d aicore -f app/db/migrations/001_initial.sql
```

> If using local Docker, set `SUPABASE_URL=http://localhost:8000` and connect directly via psql.
> For full Supabase-compatible local dev, see [Supabase Local Development](https://supabase.com/docs/guides/cli/local-development).

### Seed default system prompts

```bash
PYTHONPATH=. python scripts/seed_prompts.py
```

This inserts the 4 default use-case prompts: `default_rag`, `document_summary`, `risk_analysis`, `response_draft`.

### Start the API server

```bash
uvicorn app.main:app --reload
```

The API is now running at `http://localhost:8000`.

- **Interactive docs**: http://localhost:8000/docs
- **Health check**: http://localhost:8000/health

---

## 3. API Usage Examples

### Upload a document

```bash
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@contract.pdf"
```

Response:
```json
{
  "id": "doc_uuid_here",
  "filename": "contract.pdf",
  "status": "pending",
  "message": "Document uploaded. Call /documents/{id}/process to start processing."
}
```

### Process the document (parse + embed)

```bash
curl -X POST http://localhost:8000/documents/{doc_id}/process
```

Response:
```json
{
  "id": "doc_uuid_here",
  "status": "ready",
  "chunk_count": 42,
  "page_count": 15,
  "message": "Document processed successfully: 42 chunks created."
}
```

### Ask a question (RAG query)

```bash
curl -X POST http://localhost:8000/ai/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the termination clauses in this contract?",
    "use_case": "default_rag",
    "top_k": 5
  }'
```

Response:
```json
{
  "response_text": "According to Section 8, either party may terminate...",
  "trace_id": "trace_uuid_here",
  "model_provider": "openai",
  "model_name": "gpt-4o-mini",
  "latency_ms": 1240,
  "use_case": "default_rag",
  "retrieved_chunk_count": 5,
  "retrieved_chunks": [...]
}
```

### Submit feedback

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "trace_id": "trace_uuid_here",
    "rating": 5,
    "comment": "Very accurate and well-cited."
  }'
```

### Check evaluation results

```bash
curl http://localhost:8000/evals/{trace_id}
```

Response:
```json
{
  "trace_id": "trace_uuid_here",
  "grounding_score": 0.342,
  "length_ok": true,
  "has_citations": false,
  "passed": true
}
```

### List / create prompts

```bash
# List all prompts
curl http://localhost:8000/prompts

# Create a custom prompt
curl -X POST http://localhost:8000/prompts \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Legal Review Prompt",
    "version": 1,
    "use_case": "legal_review",
    "content": "You are a legal document reviewer...",
    "is_active": true
  }'
```

---

## 4. Project Structure

```
llm_ai_governance_project/
├── app/
│   ├── main.py              ← FastAPI app + router registration
│   ├── api/
│   │   ├── routes/          ← Thin HTTP handlers (call domain services)
│   │   └── deps.py          ← Shared FastAPI dependencies (auth, db)
│   ├── core/
│   │   ├── config.py        ← All settings from .env (single source of truth)
│   │   ├── security.py      ← Auth token verification (stub in MVP)
│   │   ├── logging.py       ← Structured logger factory
│   │   └── errors.py        ← AppError hierarchy + HTTP exception handlers
│   ├── domain/
│   │   ├── ai_service.py    ← Central orchestrator (THE most important file)
│   │   ├── documents/       ← Upload, parse, chunk, embed
│   │   ├── retrieval/       ← pgvector similarity search
│   │   ├── llm/             ← Router, OpenAI client, Gemini client, guardrails
│   │   ├── prompts/         ← Versioned system prompt library
│   │   ├── traces/          ← Full audit log of every AI request
│   │   ├── feedback/        ← User ratings on AI responses
│   │   └── evals/           ← Heuristic quality graders
│   ├── db/
│   │   ├── supabase.py      ← Supabase client singleton
│   │   └── migrations/
│   │       └── 001_initial.sql  ← Complete DB schema
│   └── tests/
├── notebooks/
│   └── finetune_dataset_builder.ipynb  ← Export fine-tuning datasets
├── scripts/
│   ├── seed_prompts.py          ← Insert default system prompts (run once)
│   ├── backfill_embeddings.py   ← Re-embed chunks after model changes
│   └── export_finetune_data.py  ← CLI fine-tuning export
├── .env.example
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```

---

## 5. Running Tests

```bash
# Run all tests with verbose output
PYTHONPATH=. pytest app/tests/ -v

# Run a specific test file
PYTHONPATH=. pytest app/tests/test_router.py -v

# Run with coverage report
PYTHONPATH=. pytest app/tests/ --cov=app --cov-report=term-missing
```

Tests use mocks for all external dependencies (Supabase, OpenAI, Gemini).
No API keys needed to run tests.

---

## 6. Fine-Tuning Workflow

Fine-tuning teaches the model HOW to respond, not WHAT to know.
Use it after collecting 50+ high-quality traces.

1. **Collect traces**: Use the API normally. Every query creates a trace.
2. **Collect feedback**: Rate responses via `POST /feedback`.
3. **Export dataset**: Run the Jupyter notebook or the CLI script:
   ```bash
   PYTHONPATH=. python scripts/export_finetune_data.py \
     --use-case default_rag \
     --min-rating 4 \
     --output finetune.jsonl
   ```
4. **Upload to OpenAI**: Follow [OpenAI fine-tuning docs](https://platform.openai.com/docs/guides/fine-tuning).
5. **Update config**: Set `LLM_DEFAULT_MODEL` in `.env` to your fine-tuned model ID.

---

## 7. Production Checklist

Before going to production, complete these items:

- [ ] **Replace auth stub**: Implement real Supabase JWT validation in `app/core/security.py`
- [ ] **Add Row Level Security**: Enable RLS policies in Supabase for all tables
- [ ] **Replace local file storage**: Implement S3/GCS storage backend in `DocumentService`
- [ ] **Set ALLOWED_ORIGINS**: Replace `*` in `main.py` CORS config with your frontend domain
- [ ] **Set APP_ENV=production** in environment
- [ ] **Set LOG_LEVEL=WARNING** to reduce log volume
- [ ] **Add rate limiting**: Protect `/ai/query` from abuse
- [ ] **Add async support**: Migrate DB calls to async for better throughput

---

## 8. Key Design Decisions

| Decision | Rationale |
|---|---|
| RAG for documents | Correct approach for "answer from documents" — not fine-tuning |
| pgvector in Supabase | One platform for DB + vectors; no separate vector DB needed |
| Sync processing in MVP | Simpler than queues/workers; add async job queue post-MVP |
| App-level tenant isolation | Simpler than DB RLS for MVP; add RLS before production |
| Rule-based evals only | Free, fast, zero API cost; upgrade to LLM-judge when needed |
| Config-based LLM routing | Explicit and debuggable; add per-request routing post-MVP |
| Auth stub | Unblocks development; real JWT validation is a clear TODO |
