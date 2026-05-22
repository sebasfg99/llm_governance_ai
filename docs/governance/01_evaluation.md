# 01 · Evaluación de respuestas RAG

Cada respuesta del LLM se evalúa con **ocho métricas** que producen una huella objetiva, auditable y comparable entre modelos y prompts.

Punto de entrada:

```python
from app.domain.evals.evaluator import evaluate_response

evaluate_response(
    query=...,
    chunks=[...],          # ChunkResult o dicts
    response=...,
    citations=...,         # opcional
    instructions=...,      # opcional (system prompt)
    use_llm_judge=True,
    response_mode="text",  # o "structured"
)
```

Devuelve un dict con las ocho métricas más metadatos del juez.

## Las 8 métricas

| Métrica | Tipo | Cómo se calcula | Quién la usa |
| ------- | ---- | --------------- | ------------ |
| `retrieval_precision` | float ∈ [0,1] | Fracción de chunks recuperados con `similarity ≥ 0.45`. | Risk scoring, dashboards |
| `groundedness_score` | float ∈ [0,1] | Jaccard entre tokens de respuesta y tokens de los chunks. | Fallback heurístico cuando el judge falla |
| `context_adherence_score` | float ∈ [0,1] | (claims soportados por chunks) / (claims totales) — calculado por LLM-as-judge | Guardrails (umbral 0.7), risk scoring, fine-tune export (≥0.9) |
| `citation_score` | float ∈ [0,1] | Mezcla 50/50 de presencia (¿hay citas?) y trazabilidad (¿resuelven a chunks reales?) | Risk scoring |
| `completeness_score` | float ∈ [0,1] | Cobertura de keywords del query + adecuación de longitud + IDK guard. | Risk scoring |
| `instruction_adherence_score` | float ∈ [0,1] | Penaliza JSON inválido, idioma incorrecto, longitud fuera de rango. | Risk scoring |
| `hallucination` | bool | True cuando el judge marca al menos una afirmación como `is_hallucination`. | Guardrails (rewrite), risk HIGH, exclusión de fine-tune |
| `should_have_said_idk` | bool | True cuando los chunks no contienen información suficiente y la respuesta no abstuvo. | Guardrails (rewrite suave), risk MEDIUM |

## LLM-as-Judge

El juez es una segunda llamada a `gpt-4o-mini` con `temperature=0` y `response_format=json_object`. Recibe:

- El query original.
- Los chunks numerados `[1] ... [N]`.
- La respuesta del LLM principal.
- (Opcional) las instrucciones del sistema.

Y devuelve estrictamente este JSON (definido en `app/domain/evals/llm_judge.py`):

```json
{
  "claims": [
    {
      "text": "...",
      "supported": true,
      "supporting_chunk_indexes": [1, 2],
      "is_hallucination": false
    }
  ],
  "should_have_said_idk": false,
  "reasoning": "..."
}
```

### Costo y latencia

- ~$0.0002 por llamada con `gpt-4o-mini`.
- ~150–400 ms adicionales.

Se puede desactivar con `EVAL_USE_LLM_JUDGE=false` en el `.env` o por llamada con `evaluate_response(use_llm_judge=False)`. Cuando está desactivado, `context_adherence_score` cae al `groundedness_score` heurístico.

## Persistencia

`EvalService.evaluate_trace(...)` escribe las 8 métricas en la tabla `evaluations` (columnas añadidas por la migración `006_rag_governance.sql`), junto al JSON crudo del juez en `judge_raw_json` para auditoría.

## Casos de prueba

`app/tests/test_evaluator.py` cubre:

- Respuesta limpia → 8 métricas en rango y sin alucinación.
- Respuesta con alucinación → `hallucination=True`, `context_adherence < 1`.
- Abstención correcta sin chunks → `completeness_score=1.0`.
- JSON inválido en modo structured → `instruction_adherence < 1`.
- Idioma equivocado → `instruction_adherence < 1`.

## Fórmulas resumen

```
retrieval_precision         = #chunks(sim ≥ 0.45) / #chunks
groundedness_score          = |query_tokens ∩ chunk_tokens| / |query_tokens ∪ chunk_tokens|
context_adherence_score     = #claims_supported / #claims_total   (judge)
citation_score              = 0.5 * presencia + 0.5 * trazabilidad
completeness_score          = 0.6 * keyword_coverage + 0.4 * length_quality
instruction_adherence_score = 1.0 − Σ penalizaciones (json, idioma, longitud)
hallucination               = ANY claim.is_hallucination          (judge)
should_have_said_idk        = judge.should_have_said_idk
```
