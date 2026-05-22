# 03 · Risk scoring por respuesta

Cada respuesta del LLM recibe un nivel de riesgo: **`low` | `medium` | `high`**. El nivel alimenta la respuesta al cliente (UI), los dashboards (`/metrics/rag`) y el filtro de exportación a fine-tuning.

Punto de entrada:

```python
from app.domain.response_risk import compute_risk

risk = compute_risk(
    evaluation,    # output de evaluate_response()
    guardrails,    # output de validate_response_guardrails()
    metadata,      # dict con use_case, chunks, etc.
)
```

Devuelve:

```python
{
    "risk_level":           "low" | "medium" | "high",
    "confidence_score":     float in [0,1],
    "risk_reasons":         list[str],
    "retrieval_confidence": float in [0,1],
}
```

## Matriz de decisión

| Nivel | Condición | Razón típica |
| ----- | --------- | ------------ |
| **HIGH** | `hallucination=True` | `hallucination_detected` |
| **HIGH** | `context_adherence_score < 0.7` | `low_context_adherence` |
| **HIGH** | `groundedness_score < 0.10` (con chunks) | `low_groundedness` |
| **HIGH** | `citation_score == 0` (con chunks) | `no_citations` |
| **HIGH** | PII detectado en la respuesta | `pii_exposure` |
| **HIGH** | Caso de uso legal sin disclaimer | `legal_output_missing_disclaimer` |
| **HIGH** | Cualquier guardrail con `severity=high` | `guardrail_high_severity:<code>` |
| **MEDIUM** | `0.7 ≤ context_adherence_score < 0.9` | `partial_context_adherence` |
| **MEDIUM** | `completeness_score < 0.5` | `incomplete_answer` |
| **MEDIUM** | `0 < citation_score < 0.2` | `weak_citations` |
| **MEDIUM** | `should_have_said_idk=True` | `should_have_abstained` |
| **MEDIUM** | `instruction_adherence_score < 0.7` | `low_instruction_adherence` |
| **LOW** | Ninguno de los anteriores | `all_signals_clean` |

Los códigos están en `app/domain/response_risk/scorer.py` para auditoría.

## confidence_score

Agregado en [0, 1]. Cuanto más alto, más confianza en que la respuesta es segura.

```
base = 0.45 * context_adherence
     + 0.20 * groundedness
     + 0.20 * instruction_adherence
     + 0.15 * retrieval_confidence

confidence_score = base * factor(level)
```

con `factor(low)=1.0, factor(medium)=0.7, factor(high)=0.4`.

Esto garantiza que toda respuesta HIGH tenga `confidence_score < 0.5`, lo que las dashboards usan para alertas automatizadas.

## retrieval_confidence

```
retrieval_confidence = mean(similarity) * (1 - exp(-n/3))
```

El amortiguador exponencial penaliza retrieval con muy pocos chunks: 1 chunk a 0.85 → 0.24, 5 chunks promedio 0.75 → 0.61.

## Persistencia

Tabla `response_risks` (1-1 con `traces` vía `trace_id`):

```sql
id                     UUID
trace_id               UUID
tenant_id              TEXT
client_id              TEXT
risk_level             TEXT CHECK (in low/medium/high)
confidence_score       NUMERIC(5,4)
retrieval_confidence   NUMERIC(5,4)
risk_reasons           JSONB
created_at             TIMESTAMPTZ
```

`traces.risk_level` es una columna desnormalizada para queries rápidas en dashboards.

## Casos cubiertos por tests

`app/tests/test_response_risk.py` valida:

- Cada uno de los 6 disparadores HIGH.
- Los 5 disparadores MEDIUM.
- LOW solo cuando todas las señales son limpias.
- `confidence_score` siempre en [0, 1].
- HIGH siempre tiene confidence menor que LOW.

## Mapa de uso

| Consumidor | Cómo usa el risk |
| ---------- | ---------------- |
| `validate_response_guardrails` | Bloquea/rewriting cuando HIGH. |
| `/metrics/rag` | Distribución `low/medium/high/unknown` por client. |
| `scripts/export_finetune_data.py` | Solo exporta `risk_level=low`. |
| UI cliente | Pinta badges de riesgo (verde/amarillo/rojo). |
