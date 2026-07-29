# Testing y Debugging

Esta guía cubre **cómo testear `ai-core`** (suite automática + tests manuales),
**cómo validar la calidad del retrieval** y **cómo debuggear problemas
comunes** (malas respuestas, falta de contexto, hallucinations).

---

## 1. Suite de tests automática

### 1.1 Estructura

Los tests viven en `app/tests/`:

```
app/tests/
├── test_router.py        # LLM router + guardrails
├── test_documents.py     # Upload, chunking, embeddings (mocked)
└── test_ai.py            # AIService end-to-end con mocks
```

Usa **pytest** + **pytest-asyncio** + **httpx.TestClient** para integración
de FastAPI.

### 1.2 Correr los tests

```bash
# Todo
pytest -v

# Un archivo
pytest app/tests/test_ai.py -v

# Un test específico
pytest app/tests/test_ai.py::test_ai_query_happy_path -v

# Con cobertura (requiere pip install pytest-cov)
pytest --cov=app --cov-report=term-missing
```

### 1.3 Variables de entorno para tests

Los tests importan `app.core.config.settings`, así que necesitan que **alguna
configuración esté disponible**. Lo más simple es mantener el `.env` de
desarrollo y hacer que los tests **mockeen** las llamadas externas (OpenAI,
Supabase) en lugar de hablar con servicios reales.

> Tip: para CI, exporta variables ficticias:
>
> ```bash
> SUPABASE_URL=http://fake \
> SUPABASE_SERVICE_KEY=fake \
> OPENAI_API_KEY=sk-fake \
> pytest
> ```

### 1.4 Patrón de mock

`test_ai.py` ilustra cómo mockear todo el stack:

```python
from unittest.mock import MagicMock, patch

def test_ai_query():
    with patch("app.domain.ai_service.PromptService") as mock_prompt, \
         patch("app.domain.ai_service.RetrievalService") as mock_retrieval, \
         patch("app.domain.ai_service.LLMRouter") as mock_router, \
         patch("app.domain.ai_service.TraceService") as mock_trace, \
         patch("app.domain.ai_service.EvalService"), \
         patch("app.domain.ai_service.ProjectService"), \
         patch("app.domain.ai_service.ThreadService"):

        mock_prompt.return_value.get_active_prompt.return_value = MagicMock(
            content="System prompt", name="chatbot", version=1
        )
        # ... etc
```

> **Por qué mockear y no integrar contra Supabase real:** los tests deben
> correr en segundos sin red, sin keys, sin fallos por rate-limit.

---

## 2. Tests manuales (smoke)

Tras una migración o cambio grande, ejecutar a mano:

```bash
# 1. El servidor levanta
uvicorn app.main:app --reload &
sleep 2 && curl http://localhost:8000/health

# Autenticación local Path A
KEY=$(grep '^AI_CORE_SERVICE_KEY=' .env | cut -d= -f2-)

# 2. La biblioteca está sembrada
curl -H "Authorization: Bearer $KEY" \
  -H "X-Tenant-Id: dev-tenant" \
  -H "X-User-Id: dev-user" \
  http://localhost:8000/documents | jq '.total'

# 3. Una query simple funciona
curl -X POST http://localhost:8000/ai/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $KEY" \
  -H "X-Tenant-Id: dev-tenant" \
  -H "X-User-Id: dev-user" \
  -d '{"query":"¿Qué es ISO 42001?","use_case":"chatbot"}' | jq '.response_text'
```

---

## 3. Validación de calidad del retrieval

El retrieval es el componente más crítico de RAG. Si los chunks que pasan al
LLM son irrelevantes, **ninguna mejora del prompt** lo va a salvar. Estos son
los métodos para validarlo.

### 3.1 Inspección manual

Cada respuesta de `/ai/query` incluye `retrieved_chunks` con un preview de
200 chars. Léelos y verifica:

- ¿Son **on-topic** con la pregunta?
- ¿La similitud es razonable? (≥ 0.6 es buen indicador; 0.45–0.6 es marginal;
  < 0.45 ya está descartado por threshold).
- ¿Vienen de **distintos documentos** (efecto del cap de diversidad)?

```bash
curl -X POST http://localhost:8000/ai/query \
  -H "Authorization: Bearer $KEY" \
  -H "X-Tenant-Id: dev-tenant" \
  -H "X-User-Id: dev-user" \
  -H "Content-Type: application/json" \
  -d '{"query":"...","use_case":"chatbot"}' \
  | jq '.retrieved_chunks[] | {similarity, doc: .document_id, preview: .content_preview}'
```

### 3.2 Logs de retrieval

`RetrievalService` emite logs `INFO` y `DEBUG` con info útil. Activa
`LOG_LEVEL=DEBUG` en `.env`:

```
INFO Retrieving top-5 chunks (candidate_k=30) for tenant=dev-tenant, ...
DEBUG Searching with 2 query variant(s): ['ISO 42001', 'ISO 42001']
DEBUG Variant 'ISO 42001' → tenant=12 global=18
DEBUG Merged candidate pool: 24 unique chunks across 2 variant(s)
INFO Threshold filter (0.45): discarded 12 low-similarity chunks, 12 remaining
INFO Retrieval complete: returning 5 chunks (variants: 2, unique_candidates: 24, threshold_discarded: 12)
```

### 3.3 Eval set propio

Para validar mejoras (cambio de embeddings, threshold, chunking) construye un
**eval set** de 20–50 preguntas con su respuesta esperada (o el doc esperado):

```python
# eval/cases.json
[
  {
    "query": "¿Qué requiere ISO 42001 sobre el comité de gobernanza?",
    "expected_doc": "ISO 42001 2023.pdf",
    "expected_keywords": ["comité", "gobernanza", "responsabilidad"]
  },
  ...
]
```

Y un script que llama `/ai/query` y verifica:

- ¿`expected_doc` aparece en `retrieved_chunks[].document_id`?
- ¿La respuesta menciona los `expected_keywords`?

Esto te da una **métrica** comparable entre cambios.

### 3.4 Heurística automática (`evaluations`)

Cada trace tiene su evaluación. Para revisar la calidad agregada:

```sql
-- En Supabase SQL Editor
SELECT
  use_case,
  COUNT(*) total,
  AVG(grounding_score) avg_grounding,
  SUM(CASE WHEN passed THEN 1 ELSE 0 END)::float / COUNT(*) pass_rate
FROM evaluations e
JOIN traces t ON e.trace_id = t.id
WHERE t.created_at > NOW() - INTERVAL '7 days'
GROUP BY use_case;
```

---

## 4. Debugging de malas respuestas

### Síntoma A — "El bot no sabe nada del tema"

**Causa probable:** retrieval no encontró chunks relevantes (debajo del
threshold).

**Cómo confirmarlo:**

1. Mirar `retrieved_chunk_count` en la respuesta. Si es `0`, RAG falló.
2. Mirar logs: ¿salió "No chunks above similarity threshold"?
3. Verificar que el documento esperado existe (`GET /documents`) y está
   `ready` con `chunk_count > 0`.

**Soluciones:**

- ¿La query está en otro idioma que los docs? Asegurar
  `RETRIEVAL_MULTILINGUAL=true`.
- ¿El threshold es muy alto? Bajar a 0.35 temporalmente y ver si pasan chunks
  útiles.
- ¿La pregunta es muy genérica? Reformular con keywords específicos.
- ¿`framework_filter` está excluyendo el doc correcto? Quitarlo para verificar.

### Síntoma B — "El bot inventa cosas (hallucinations)"

**Causa probable:** los chunks recuperados no respaldan la afirmación, pero
el LLM rellenó con conocimiento de entrenamiento.

**Cómo confirmarlo:**

1. Mirar `evaluations.grounding_score` para esa trace. Si es < 0.5 → señal de
   hallucination.
2. Leer los chunks (campo `retrieved_content` en la trace) y la respuesta
   lado a lado: ¿la respuesta dice cosas que **no están** en los chunks?

**Soluciones:**

- Reforzar el "GROUNDING CONTRACT" en el system prompt: "Si los documentos no
  contienen la respuesta, dilo en lugar de inferir."
- Bajar `top_k` (menos chunks dispersos = más enfoque).
- Subir el threshold (chunks dudosos engañan al LLM).
- Cambiar el modelo a `gpt-4o` (mejor adherencia a instrucciones que
  `gpt-4o-mini`).

### Síntoma C — "El bot olvida lo que dijo antes"

**Causa probable:** el cliente no está reusando `thread_id`.

**Cómo confirmarlo:**

1. En la respuesta de `/ai/query`, mirar `thread_id`. Si cambia entre
   request, el cliente no lo está reusando.
2. `GET /threads/{thread_id}/messages` debe mostrar todos los turnos.

**Soluciones:**

- Persistir `thread_id` en localStorage / Redis / DB.
- Verificar que el `use_case` es conversacional (`chatbot`, `risk_intake`, …);
  los no-conversacionales (`executive_summary`) **no** crean thread por
  defecto.

### Síntoma D — "El JSON estructurado viene mal formado"

**Causa probable:** el LLM ignoró las instrucciones del prompt de output.

**Cómo confirmarlo:**

```python
import json
data = resp.json()
try:
    parsed = json.loads(data["response_text"])
except json.JSONDecodeError as e:
    print("LLM output:", data["response_text"][:500])
    print("Error:", e)
```

**Soluciones:**

- Asegurar `response_mode="structured"` en el request.
- Cambiar a `gpt-4o` (mejor adherencia a JSON).
- Usar el fallback documentado en el prompt: si la primera parse falla, hacer
  un segundo call pidiendo "convierte la respuesta anterior a JSON válido".
- A futuro: usar **Function Calling** / **JSON mode** del API de OpenAI (TODO
  en `llm/openai_client.py`).

### Síntoma E — "Latencias >10 s para chatbot"

**Causa probable:** modelo grande, top_k muy alto, o muchas variantes
multilingual + RPC lento.

**Cómo confirmarlo:** mirar `latency_ms` (LLM solo). El tiempo total incluye
embedding + RAG + persistencia.

**Soluciones:**

- Bajar a `LLM_DEFAULT_MODEL=gpt-4o-mini` para chatbot.
- Bajar `top_k` (menos contexto = menos tokens = menos latencia).
- Desactivar `RETRIEVAL_MULTILINGUAL` si tu corpus es monolingüe.
- Mirar el índice ivfflat: `lists` muy alto puede degradar latencia.

---

## 5. Debugging de errores HTTP

### `422 prompt_injection_detected`

El query disparó una regex de `app/domain/llm/guardrails.py`. La respuesta
incluye `matched_pattern`. Si es **falso positivo** (un usuario legítimo
preguntando algo que parece inyección), ajustar la regex:

```python
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|...)\s+instructions", re.IGNORECASE),
    # Hacer más específico para evitar falsos positivos
]
```

### `502 upstream_error` con OpenAI

Causas frecuentes:
- `OPENAI_API_KEY` inválida o sin créditos.
- Rate-limit de la cuenta.
- Timeout de red.

Verificar:

```bash
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer YOUR_OPENAI_API_KEY" | jq '.data[0]'
```

### `502 upstream_error` con Supabase

Causas:
- Migraciones SQL no aplicadas (los RPCs `match_chunks` no existen).
- `SUPABASE_SERVICE_KEY` rotada o equivocada.
- Función `match_chunks` con firma ambigua (ver issue conocido en migración
  004 — solución: `DROP FUNCTION` antes de `CREATE OR REPLACE`).

Test rápido:

```sql
SELECT * FROM match_chunks(
    array_fill(0::float, ARRAY[1536])::vector,
    'global'::text,
    1::int,
    NULL::text[]
);
```

Si devuelve un error y no una row vacía, hay un problema de firma.

---

## 6. Activar logs detallados

```env
# .env
LOG_LEVEL=DEBUG
```

Y reiniciar uvicorn. Verás logs como:

```
DEBUG app.domain.retrieval.service: Embedding 1 text(s) using text-embedding-3-small
DEBUG app.domain.retrieval.service: Variant 'ISO 42001' → tenant=8 global=14
DEBUG app.domain.ai_service: Final prompt assembled: 4823 chars
INFO app.domain.llm.router: LLM response received: 612 chars, 1840ms
```

Para producción, dejar en `INFO`. `DEBUG` produce mucho output.

---

## 7. Inspeccionar traces en Supabase

Cualquier respuesta puede ser auditada:

```sql
SELECT
  id, created_at, use_case, model_name, latency_ms,
  user_query,
  LEFT(response_text, 200) AS response_preview
FROM traces
WHERE created_at > NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC
LIMIT 20;
```

Y para una trace específica, ver el prompt completo y los chunks:

```sql
SELECT prompt_text, retrieved_chunk_ids, retrieved_content
FROM traces WHERE id = '1d4a-...';
```

---

## 8. Evaluación automática

`app/domain/evals/graders.py` corre 3 graders heurísticos por trace. Para
añadir un grader nuevo:

```python
def grade_completeness(response: str, query: str, chunks: list[str]) -> dict:
    # Tu lógica
    return {"completeness_score": 0.85, "passed": True}
```

Y registrarlo en `EvalService.evaluate_trace()`. Los graders deben ser
**rápidos y deterministas**: si necesitas LLM-as-judge, eso es otro endpoint
opt-in (no parte del flujo principal).

---

## 9. Checklist de pre-deploy

- [ ] `pytest -v` todo verde.
- [ ] `seed_prompts.py` y `seed_documents.py` corren sin error.
- [ ] El smoke test (`/health` + una query) responde en < 5 s.
- [ ] No hay `OPENAI_API_KEY` ni otras secrets logueadas.
- [ ] `LOG_LEVEL=INFO` (no `DEBUG`) en producción.
- [ ] El threshold de similitud está afinado para el corpus real (no para los
      docs de ejemplo).
- [ ] CORS restringido al dominio del frontend real.
- [ ] Token stub (`security.py`) reemplazado por validación real de JWT
      (cuando aplique).

---

## Ver también

- [Pipeline RAG completo](rag/full_pipeline.md)
- [Decisiones y trade-offs](decisions/tradeoffs.md)

---

## Checklist manual: `country` + `industry` en todas las rutas LLM

### Pre-condiciones

1. Aplicar migraciones `008_industry_codes.sql` **y `010_industry_filter_fix.sql`**
   en Supabase (Python sidecar DB). Sin la 010 el filtro por industria no filtra
   nada — cualquier escenario de esta checklist pasaría en falso.
   Ver [`rag/industry_codes.md`](rag/industry_codes.md).
2. Colocar SUGEF 10-07 y SUGEF 18-16 en `docs/admin/Ciberseguridad/Costa Rica/`.
3. Re-ejecutar: `PYTHONPATH=. python scripts/seed_documents.py`
4. Confirmar en DB: `SELECT filename, industry_codes, country FROM document_chunks WHERE filename ILIKE '%SUGEF 10%' LIMIT 5;`
   → Esperado: `industry_codes = '{bank,cooperative}'`, `country = 'costa-rica'`

### Escenarios (5 × 3 preguntas)

| Escenario | Pregunta de prueba | Criterio de éxito |
|-----------|-------------------|-------------------|
| CR + cooperative | ¿Qué es el gobierno corporativo? | ≥1 chunk local CR (SUGEF 24-22/CONASSIF) + ≥1 global (ISO/NIST) |
| CR + cooperative | ¿Qué controles aplican a decisiones automatizadas de crédito? | SUGEF 10-07 + NIST/ISO (requiere PDF indexado) |
| CR + bank | ¿Qué obligaciones de ciberseguridad aplican? | CONASSIF / SUGEF 18-16 + global OWASP/ISO |
| PA + bank | ¿Qué normativa de ciberseguridad aplica en Panamá? | Chunks PA locales + globales; sin chunks CR |
| GT + fintech | ¿Qué dice ISO 42001 sobre comité de gobernanza? | Solo globales; sin SUGEF/CONASSIF (fintech no aplica) |

**Verificar siempre:**
- `retrieved_chunk_count > 0`
- Mix de chunks locales + globales en la respuesta
- Sin citas de regulaciones inventadas
- Log del sidecar muestra: `country=<slug>, industry=<code>`

### Verificación payload Edge Function

En devtools (Network → `agent-chat`), confirmar que el body enviado tiene:
```json
{ "country": "costa-rica", "industry": "cooperative", "include_global": true }
```

### Test panel de admin

1. Admin → Prompts → seleccionar prompt → "Probar este prompt"
2. Verificar que aparecen los selectores **País** e **Industria**
3. Confirmar que "Enviar prueba" está deshabilitado hasta seleccionar ambos
4. Seleccionar CR + cooperative → enviar pregunta → verificar respuesta contextualizada

### Guard de locale incompleto (UI)

1. Institución sin country/industry → abrir agente → el primer mensaje muestra aviso
2. Ir a Configuración → banner de advertencia rojo visible
3. CO edita país e industria → guarda → banner desaparece
