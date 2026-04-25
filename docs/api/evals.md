# Endpoint de Evaluaciones

Tras cada llamada al LLM, el sistema corre una **evaluación heurística
automática** (sin llamar a otro LLM) y la almacena en la tabla `evaluations`.
Este endpoint expone esos resultados.

---

## `GET /evals/{trace_id}`

```
GET /evals/{trace_id}
Authorization: Bearer <token>
```

### Response

```json
{
  "trace_id": "1d4a-...",
  "grounding_score": 0.72,
  "length_ok": true,
  "has_citations": true,
  "passed": true,
  "grader_details": {
    "word_overlap_with_chunks": 0.72,
    "min_words": 30,
    "max_words": 800,
    "actual_words": 245
  },
  "created_at": "2026-04-25T15:30:08Z"
}
```

### Campos

| Campo | Descripción |
|---|---|
| `grounding_score` | 0.0–1.0. Mide solapamiento de palabras entre la respuesta y los chunks recuperados. Valor bajo = posible hallucination. |
| `length_ok` | `true` si la respuesta cae dentro del rango razonable de palabras (configurable). |
| `has_citations` | `true` si la respuesta hace referencia a chunks (heurística por menciones / formato structured). |
| `passed` | `true` si superó todos los graders. |
| `grader_details` | Diccionario con detalles por grader. |

### Errores

| HTTP | Causa |
|---|---|
| 404 | `trace_id` no existe o aún no tiene evaluación. |

---

## Ejemplo

```bash
curl http://localhost:8000/evals/1d4a-... \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## Cómo se calcula

La evaluación corre dentro de `AIService.query()` después de persistir la
trace, en un `try/except` no bloqueante: si los graders fallan no afecta la
respuesta del usuario. El código vive en `app/domain/evals/`.

Para más detalle (umbrales, lógica por grader, cómo agregar un nuevo grader)
ver [`../testing.md`](../testing.md#evaluación-automática).
