# 05 · Observabilidad

Toda llamada al LLM produce un registro completo y auditable. La observabilidad se compone de **tres tablas** y **un endpoint agregado**.

## Qué se registra

| Campo | Tabla | Descripción |
| ----- | ----- | ----------- |
| `id` | `traces` | UUID de la traza. |
| `client_id` / `tenant_id` | `traces` | Cliente que originó la llamada (alias mantenido por trigger). |
| `user_id` | `traces` | Usuario humano (cuando aplique). |
| `user_query` | `traces` | Pregunta original. |
| `prompt_text` | `traces` | Prompt completo enviado al LLM (system + contexto + query). |
| `retrieved_chunk_ids` | `traces` | UUIDs de los chunks recuperados. |
| `retrieved_content` | `traces` | Texto concatenado de los chunks (para debugging). |
| `response_text` | `traces` | Respuesta segura entregada al cliente. |
| `model_provider` / `model_name` | `traces` | OpenAI/Gemini + modelo. |
| `prompt_tokens`, `completion_tokens` | `traces` | Tokens consumidos (cuando el provider los reporta). |
| `cost_estimate_usd` | `traces` | Costo en USD calculado por `app/domain/observability/cost.py`. |
| `latency_ms` | `traces` | Tiempo end-to-end del LLM. |
| `risk_level`, `requires_disclaimer` | `traces` | Eco del risk para queries rápidas. |
| 8 métricas de gobernanza | `evaluations` | Ver doc 01. |
| `judge_raw_json` | `evaluations` | JSON crudo del juez para auditoría. |
| `risk_level`, `confidence_score`, `risk_reasons` | `response_risks` | Ver doc 03. |

## Tabla de precios

`app/domain/observability/cost.py` mantiene `MODEL_PRICING` (USD por 1M tokens). Versión: `2025.04.refresh`.

```
gpt-4o-mini      input 0.15  output 0.60
gpt-4o           input 2.50  output 10.00
gemini-1.5-flash input 0.075 output 0.30
text-embedding-3-small  input 0.02
...
```

Modelos no listados usan un fallback conservador (`1.00 / 3.00`). El log marca el fallback en DEBUG para detectar tabla obsoleta.

Refrescar trimestralmente:

1. Editar `MODEL_PRICING` con los precios públicos vigentes.
2. Bumpear `PRICING_VERSION`.
3. Documentar en este archivo.

## Logger unificado

`app/domain/observability/logger.py` expone:

```python
ObservabilityLogger(db).log_llm_call(
    client_id=...,
    user_id=...,
    query=...,
    prompt_text=...,
    use_case=...,
    chunks=...,
    response_text=...,
    model_provider=...,
    model_name=...,
    prompt_tokens=...,
    completion_tokens=...,
    latency_ms=...,
    evaluation=...,
    guardrails=...,
    risk=...,
)
```

Encadena trace → evaluation → response_risk en una sola transacción lógica. Cada paso se envuelve en try/except: una falla en evaluation NO impide guardar el trace, una falla en response_risks NO oculta los datos del trace.

## Endpoint de métricas

`GET /metrics/rag?client_id=...&since=...&until=...&use_case=...`

Devuelve:

```json
{
  "client_id": "acme",
  "window": {"since": "...", "until": "..."},
  "totals": {
    "calls": 124,
    "cost_usd": 0.81,
    "prompt_tokens": 95400,
    "completion_tokens": 18200,
    "evaluations": 124,
    "risks": 121
  },
  "averages": {
    "retrieval_precision": 0.71,
    "groundedness_score": 0.43,
    "context_adherence_score": 0.92,
    "citation_score": 0.66,
    "completeness_score": 0.81,
    "instruction_adherence_score": 0.94,
    "cost_usd": 0.0065
  },
  "latency_ms": {"avg": 612, "p50": 480, "p95": 1450, "max": 3120},
  "risk_distribution": {"low": 102, "medium": 18, "high": 4, "unknown": 0},
  "hallucination_rate": 0.04,
  "should_have_said_idk_rate": 0.10,
  "use_cases": {"chatbot": 80, "risk_analysis": 44},
  "models": {"gpt-4o-mini": 124}
}
```

La ventana por defecto es 30 días. El endpoint deniega si `client_id` falta, y advierte (sin bloquear) si el `client_id` query difiere del token (porque hoy el token es un stub).

## Vista SQL agregada

`rag_metrics_v` (en migración 006) contiene un row por `(client_id, día)`. Útil para Prometheus/Grafana cuando el dashboard se haga real:

```sql
SELECT * FROM rag_metrics_v
WHERE client_id = 'acme'
  AND day >= NOW() - INTERVAL '30 days'
ORDER BY day DESC;
```

## Retención

Las trazas y evaluaciones son auditoría regulatoria. Recomendación de retención:

| Dato | Retención mínima | Razón |
| ---- | ---------------- | ----- |
| `traces.user_query`, `prompt_text`, `response_text` | 12 meses | Auditoría regulatoria, drift analysis |
| `evaluations.judge_raw_json` | 6 meses | Reproducibilidad del judge |
| `response_risks` | 12 meses | Mismo SLA que traces |
| Métricas agregadas (`rag_metrics_v`) | 24 meses | Tendencias largas |

Implementar con `pg_cron` o limpieza programada cuando el volumen lo justifique.

## Notebook de soporte

`notebooks/rag_results_dashboard.ipynb` carga los últimos 100 traces, hace EDA visual y exporta `summary.csv` + `summary.png`. Útil para QBR o pre-mortem antes de cambios de prompt.
