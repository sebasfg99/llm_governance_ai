# Gobernanza, Auditoría y Seguridad RAG

Esta carpeta documenta los módulos de gobernanza añadidos al sistema RAG. Todo el contenido está en español, alineado con el equipo de cumplimiento.

## Índice

| # | Módulo | Documento |
| - | ------ | --------- |
| 01 | Evaluación de respuestas (8 métricas) | [01_evaluation.md](./01_evaluation.md) |
| 02 | Adherencia al contexto y detección de alucinaciones | [02_context_adherence.md](./02_context_adherence.md) |
| 03 | Risk scoring por respuesta (HIGH / MEDIUM / LOW) | [03_risk_scoring.md](./03_risk_scoring.md) |
| 04 | Guardrails de respuesta (PII, disclaimers, abstención) | [04_guardrails.md](./04_guardrails.md) |
| 05 | Observabilidad (cost, latencia, métricas) | [05_observability.md](./05_observability.md) |
| 06 | Seguridad multi-tenant | [06_multi_tenant_security.md](./06_multi_tenant_security.md) |

## Diagrama del pipeline

```
client_id + query
        │
        ▼
validate_tenant_access  ──►  AuthorizationError si falla
        │
        ▼
Input Guardrails (prompt injection)
        │
        ▼
Hybrid Retrieval (pgvector + tsvector + RRF)
        │
        ▼
Reranker (feature reweighting)
        │
        ▼
LLM Router (OpenAI / Gemini)
        │
        ▼
LLM-as-Judge → 8 métricas de evaluación
        │
        ▼
validate_response_guardrails (PII, disclaimer, IDK, adherencia)
        │
        ▼
compute_risk (LOW / MEDIUM / HIGH)
        │
        ▼
Persistencia: traces + evaluations + response_risks
        │
        ▼
Respuesta (safe_response, trace_id, risk_level)
```

## Endpoints principales

- `POST /ai/query` — pipeline completo, devuelve la respuesta segura.
- `GET  /metrics/rag?client_id=...&since=...&until=...` — métricas agregadas.
- `GET  /evals/{trace_id}` — evaluación individual.

## Scripts de soporte

- `scripts/export_finetune_data.py` — exporta solo trazas con `risk_level=low` y `hallucination=false`, formato OpenAI fine-tuning.
- `notebooks/rag_results_dashboard.ipynb` — dashboard exploratorio con últimas 100 trazas.

## Tablas relevantes

- `traces` (extendida) — almacena cost_estimate_usd, prompt_tokens, completion_tokens, risk_level, requires_disclaimer, client_id.
- `evaluations` (extendida) — 8 métricas + judge_raw_json.
- `response_risks` (nueva) — risk_level, confidence_score, retrieval_confidence, risk_reasons.
- `rag_metrics_v` (vista) — agregaciones por client_id y día para `/metrics/rag`.

Migración SQL: [`app/db/migrations/006_rag_governance.sql`](../../app/db/migrations/006_rag_governance.sql).
