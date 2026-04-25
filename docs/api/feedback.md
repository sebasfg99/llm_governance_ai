# Endpoint de Feedback

Captura ratings (1–5) y comentarios opcionales del usuario sobre las
respuestas del LLM. Cada feedback está atado a una `trace_id` específica.

---

## `POST /feedback`

```
POST /feedback
Content-Type: application/json
Authorization: Bearer <token>
```

### Request

```jsonc
{
  "trace_id": "uuid",        // requerido — id devuelto por /ai/query
  "rating": 5,                // 1..5
  "comment": "Respuesta muy precisa y bien citada."  // opcional, ≤ 2000 chars
}
```

### Response

```json
{
  "id": "uuid",
  "trace_id": "uuid",
  "rating": 5,
  "comment": "...",
  "created_at": "2026-04-25T15:31:30Z",
  "message": "Thank you for your feedback!"
}
```

### Errores

| HTTP | Causa |
|---|---|
| 422 | `rating` fuera de rango (debe ser 1..5). |
| 404 | `trace_id` no existe. |
| 409 | Ya enviaste feedback para esa trace (UNIQUE `trace_id, user_id`). |

---

## Ejemplos

**curl**

```bash
curl -X POST http://localhost:8000/feedback \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "trace_id": "1d4a-...",
    "rating": 4,
    "comment": "Buena pero le faltó citar la cláusula 8.4"
  }'
```

**JavaScript**

```javascript
async function submitFeedback(traceId, rating, comment) {
  return fetch("/feedback", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: "Bearer YOUR_TOKEN",
    },
    body: JSON.stringify({ trace_id: traceId, rating, comment }),
  }).then((r) => r.json());
}
```

---

## Por qué importa

- **Identificar respuestas malas** para revisión y ajuste de prompt.
- **Construir dataset** de respuestas buenas para fine-tuning (ver
  `scripts/export_finetune_data.py`).
- **Métricas de calidad** a lo largo del tiempo, por `use_case` y por modelo.

> Un patrón común en la UI: mostrar pulgar arriba/abajo bajo cada respuesta del
> chat. Pulgar arriba = `rating=5`, pulgar abajo = `rating=1` con un input
> opcional para `comment`.
