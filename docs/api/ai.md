# `POST /ai/query` — Consulta RAG central

El endpoint principal del sistema. Recibe una pregunta del usuario, ejecuta el
pipeline RAG completo (guardrails → recuperación → ensamblaje de prompt →
LLM → trazas → evaluación) y devuelve la respuesta junto con metadatos de
auditoría.

---

## Descripción

`POST /ai/query` es la entrada única para interactuar con el LLM. Soporta seis
**use cases**, cada uno con su system prompt versionado en la tabla `prompts`:

| `use_case` | Para qué sirve |
|---|---|
| `chatbot` | Q&A conversacional con citas (caso más común) |
| `executive_summary` | Resumen ejecutivo one-shot de un proyecto / documento |
| `risk_intake` | Entrevista conversacional para llenar el `project_profile` |
| `risk_analysis` | Generación estructurada del registro de riesgos (típicamente vía `/risk/{id}/analyze`) |
| `policy_generation` | Borrador de política organizacional |
| `controls_generation` | Derivación de controles desde el registro de riesgos |

> Aunque `risk_analysis`, `policy_generation` y `controls_generation` aceptan
> llamadas directas a `/ai/query`, el camino recomendado es usar los endpoints
> dedicados (`/risk/...`, `/policies/...`) que aplican validación adicional y
> persisten resultados al expediente del proyecto.

---

## Método y URL

```
POST /ai/query
```

Headers:

```
Content-Type: application/json
Authorization: Bearer <token>
```

---

## Request body

```jsonc
{
  "query": "string (1..4000 chars, requerido)",
  "use_case": "chatbot | executive_summary | policy_generation | risk_intake | risk_analysis | controls_generation",
  "top_k": 5,                          // 1..20, chunks a recuperar
  "include_global": true,               // buscar en biblioteca global (tenant=global)
  "response_mode": "text",              // "text" o "structured" (JSON)
  "stream": false,                      // true => respuesta SSE token a token
  "country": "costa-rica",              // opcional, filtra retrieval por país
  "project_id": "uuid",                 // opcional, hidrata memoria L2
  "thread_id": "uuid",                  // opcional, continúa conversación L1
  "framework_filter": ["iso_42001"]     // opcional, restringe RAG a frameworks
}
```

### Campos relevantes

| Campo | Tipo | Default | Descripción |
|---|---|---|---|
| `query` | string | — | La pregunta del usuario. Mínimo 1, máximo 4000 caracteres. |
| `use_case` | enum | `chatbot` | Selecciona qué system prompt usar. |
| `top_k` | int | 5 | Cuántos chunks pasar al LLM. Más = más contexto pero más tokens. |
| `include_global` | bool | true | Si `false`, solo busca en los documentos del tenant. |
| `response_mode` | enum | `text` | `text` devuelve prosa; `structured` devuelve un JSON con `answer`, `citations`, `confidence`. |
| `stream` | bool | false | Si `true`, la respuesta llega como Server-Sent Events (`text/event-stream`). **Incompatible con `response_mode="structured"`** (devuelve 422). Ver [Streaming SSE](#streaming-sse). |
| `country` | string | null | Slug de país (ej. `costa-rica`, `panama`). Restringe los chunks `country=NULL` + el país pedido. |
| `project_id` | UUID | null | Si se pasa, el orquestador inyecta un bloque “Memoria del proyecto” en el prompt. |
| `thread_id` | UUID | null | Continúa una conversación existente. Si se omite y el `use_case` es conversacional (`chatbot`, `risk_intake`, `policy_generation`), el servidor crea un thread nuevo. |
| `framework_filter` | string[] | null | Restringe el retrieval a chunks con `framework` en esta lista (o `framework=NULL`). Slugs válidos: `iso_42001`, `nist_airmf`, `eu_ai_act`, `gdpr`, `owasp_llm_top10`, `mitre_atlas`, `pci_dss`, `cybersecurity`, `corporate_governance`, `local_data_protection`. |

---

## Response body

```jsonc
{
  "response_text": "string",            // respuesta del LLM (texto o JSON serializado)
  "trace_id": "uuid|null",              // id de la trace de auditoría
  "thread_id": "uuid|null",             // thread asociado (siempre presente en use_case conversacional)
  "project_id": "uuid|null",            // proyecto si fue pasado
  "model_provider": "openai|gemini",
  "model_name": "gpt-4o-mini",
  "latency_ms": 1840,
  "use_case": "chatbot",
  "retrieved_chunk_count": 5,
  "retrieved_chunks": [
    {
      "chunk_id": "uuid",
      "document_id": "uuid",
      "similarity": 0.87,
      "content_preview": "primeros 200 caracteres del chunk..."
    }
  ],
  "used_reasoning_memory": false        // true si el orquestador detectó "¿por qué...?" y cargó traces previas
}
```

### Modo `structured`

Cuando `response_mode = "structured"`, el campo `response_text` contiene un JSON
serializado con esta forma:

```json
{
  "answer": "<respuesta en prosa>",
  "citations": [
    {
      "chunk_id": "uuid",
      "document_name": "...",
      "section": "...",
      "quote": "..."
    }
  ],
  "confidence": "high | medium | low",
  "suggested_followups": ["..."]
}
```

El cliente debe parsear este string con `JSON.parse(...)` /
`json.loads(...)`. Si el LLM no logra producir JSON válido, el system prompt
le instruye devolver un objeto de fallback con `confidence="low"`.

---

## Citaciones y atribución de fuente

Antes de pasar los chunks al LLM, el orquestador hidrata `{filename,
framework}` para cada `document_id` y los inyecta en el prompt junto al
`section_ref` que el chunker section-aware tagueó en cada chunk:

```
[Chunk N | Fuente: <filename> | Section: <ref> | Framework: <slug> | similarity: <float>]
```

`Section:` se omite cuando el documento no tiene marcadores formales
(ej. el NIST AI RMF Playbook). `Framework:` se omite cuando el documento
es framework-agnostic (`framework=NULL`).

Esto permite que el modelo:

- Cumpla la regla `[Fuente: <documento>]` del system prompt al citar.
- Cite **a nivel de cláusula** (`ISO 42001, Cláusula 6.1.2`) cuando el
  chunk pertenece a una sección formal — ver
  [ADR-013](../decisions/design_choices.md#adr-013--chunking-section-aware--hnsw).
- Distinga ISO 42001 vs NIST vs EU AI Act cuando varios marcos aplican.
- Evite refusals falsos en preguntas específicas de framework.

El system prompt activo del use case `chatbot` (versión 4 — *"Chatbot
Documental v4 (semantic inference)"*) aplica además **interpretación
semántica obligatoria**: si el usuario pregunta por "comité de gobernanza"
y los documentos hablan de "estructura organizativa de gobernanza" o
"roles y responsabilidades", el modelo los considera **conceptualmente
equivalentes** y responde directamente, en lugar de declarar "no
encontrado".

Detalle completo en
[`../rag/full_pipeline.md#7-ensamblaje-del-prompt`](../rag/full_pipeline.md#7-ensamblaje-del-prompt)
y en las
[ADR-011](../decisions/design_choices.md#adr-011--atribución-de-fuente-en-chunks)
/
[ADR-012](../decisions/design_choices.md#adr-012--prompt-v4-del-chatbot-con-inferencia-semántica).

---

## Streaming SSE

Pasar `"stream": true` en el body cambia el `Content-Type` de la
respuesta de `application/json` a `text/event-stream`. Los tokens del
LLM llegan a medida que se generan, lo que mejora la percepción de
latencia para `risk_analysis` y `policy_generation` (que tardan 8-40 s
en modo bloqueante).

### Formato de los eventos

```
event: token
data: {"text": "Hola"}

event: token
data: {"text": " mundo"}

event: token
data: {"text": "."}

event: done
data: {"trace_id":"...","thread_id":"...","project_id":null,
       "model_provider":"openai","model_name":"gpt-4o-mini",
       "latency_ms":1842,"use_case":"chatbot",
       "retrieved_chunk_count":5,
       "retrieved_chunks":[{"chunk_id":"...","section_ref":"Cláusula 6.1.2", ...}],
       "used_reasoning_memory":false,"completed":true}
```

- Cada `event: token` empuja un fragmento incremental del texto. El
  cliente debe **concatenarlos en orden** para reconstruir la
  respuesta.
- El `event: done` cierra el stream y trae la metadata que el modo
  bloqueante devolvía en el body JSON (`trace_id`, `latency_ms`, los
  chunks recuperados, etc.).
- Si la generación falla a mitad de stream, el servidor emite
  `event: error` con `{"message": "...", "detail": "..."}` antes del
  `done` (que llega con `completed: false`).

### Reglas y compatibilidad

- **`stream=true` con `response_mode="structured"` → 422.** Streaming
  JSON parcial no es parseable y no aporta UX. El validador del schema
  rechaza la combinación antes del pipeline.
- **`stream=false` (default)** mantiene el contrato JSON tradicional.
  Los clientes existentes no necesitan cambios.
- **Tracing y evaluations** se ejecutan al cierre del stream con el
  texto completo acumulado server-side, igual que en modo bloqueante.

### Ejemplo curl

`curl -N` desactiva el buffer del cliente para que veas los eventos
llegar en tiempo real:

```bash
curl -N -X POST http://localhost:8000/ai/query \
  -H 'Authorization: Bearer dev-token' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "¿Qué dice ISO 42001 sobre el análisis de riesgos de IA?",
    "use_case": "chatbot",
    "stream": true,
    "framework_filter": ["iso_42001"]
  }'
```

### Ejemplo JavaScript (browser / Next.js)

```javascript
const response = await fetch("/ai/query", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({
    query: "¿Qué dice ISO 42001 sobre el análisis de riesgos?",
    use_case: "chatbot",
    stream: true,
  }),
});

const reader = response.body.getReader();
const decoder = new TextDecoder();
let buffer = "";

while (true) {
  const { done, value } = await reader.read();
  if (done) break;
  buffer += decoder.decode(value, { stream: true });

  // SSE separa eventos con "\n\n"
  const events = buffer.split("\n\n");
  buffer = events.pop() ?? "";

  for (const evt of events) {
    const eventLine = evt.match(/^event:\s*(\S+)/m)?.[1];
    const dataLine = evt.match(/^data:\s*(.+)$/m)?.[1];
    if (!eventLine || !dataLine) continue;
    const payload = JSON.parse(dataLine);

    if (eventLine === "token") appendToUI(payload.text);
    if (eventLine === "done") finalizeUI(payload);  // payload.trace_id, payload.latency_ms, ...
    if (eventLine === "error") showError(payload);
  }
}
```

Detalle completo en
[`../rag/full_pipeline.md#8bis-streaming-sse`](../rag/full_pipeline.md#8bis-streaming-sse)
y la decisión de diseño en
[ADR-014](../decisions/design_choices.md#adr-014--streaming-sse-via-stream-param).

---

## Ejemplos

### curl

```bash
curl -X POST http://localhost:8000/ai/query \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -d '{
    "query": "¿Qué requiere ISO 42001 sobre el comité de gobernanza?",
    "use_case": "chatbot",
    "framework_filter": ["iso_42001"],
    "response_mode": "structured"
  }'
```

### Python

```python
import json
import requests

resp = requests.post(
    "http://localhost:8000/ai/query",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={
        "query": "Resúmeme los principales riesgos de un sistema generativo público",
        "use_case": "chatbot",
        "framework_filter": ["nist_airmf", "owasp_llm_top10"],
        "top_k": 8,
        "response_mode": "structured",
    },
    timeout=60,
)
resp.raise_for_status()
data = resp.json()

structured = json.loads(data["response_text"])
print("Respuesta:", structured["answer"])
print("Confianza:", structured["confidence"])
for cite in structured["citations"]:
    print(f"  - {cite['document_name']}: {cite['quote'][:120]}")
```

### JavaScript (fetch)

```javascript
const resp = await fetch("http://localhost:8000/ai/query", {
  method: "POST",
  headers: {
    "Content-Type": "application/json",
    Authorization: "Bearer YOUR_TOKEN",
  },
  body: JSON.stringify({
    query: "¿Qué controles aplican a un modelo que procesa datos personales?",
    use_case: "chatbot",
    framework_filter: ["gdpr", "iso_42001"],
    response_mode: "structured",
  }),
});

if (!resp.ok) {
  throw new Error(`HTTP ${resp.status}: ${await resp.text()}`);
}

const data = await resp.json();
const structured = JSON.parse(data.response_text);
console.log(structured.answer);
console.log("Citas:", structured.citations);
```

---

## Continuar una conversación (memoria L1)

Para mantener el hilo, guarda el `thread_id` que el servidor devuelve en la
primera llamada y reúsalo en las siguientes:

```javascript
let threadId = null;

async function ask(query) {
  const r = await fetch("/ai/query", {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: "Bearer YOUR_TOKEN" },
    body: JSON.stringify({ query, use_case: "chatbot", thread_id: threadId }),
  }).then((r) => r.json());

  threadId = r.thread_id;
  return r.response_text;
}

await ask("¿Qué es ISO 42001?");
await ask("¿Y cómo se relaciona con el NIST AI RMF?"); // recuerda el contexto
```

---

## Memoria del proyecto (L2) y "¿por qué?" (L3)

- **L2:** pasa `project_id`. El orquestador inyecta un bloque “Contexto del
  proyecto” con perfil + tier EU AI Act + resumen de riesgos. El LLM puede
  referirse a esos datos en su respuesta.
- **L3 ligero:** si el `query` contiene patrones tipo *"¿por qué…?"*,
  *"justifica…"*, *"explica por qué…"*, **y** existe `project_id`, el sistema
  carga las últimas 5 traces de análisis (`risk_analysis`, `risk_intake`,
  `policy_generation`, `controls_generation`) del mismo proyecto y las inyecta
  como “Análisis previos”. La respuesta lo señala con
  `used_reasoning_memory=true`.

Detalle del ensamblaje del prompt en
[`../rag/full_pipeline.md`](../rag/full_pipeline.md#7-ensamblaje-del-prompt).

---

## Errores comunes

| HTTP | Caso | Cómo evitarlo |
|---|---|---|
| **422** | Inyección de prompt detectada (regex match contra patrones tipo "ignore previous instructions"). | Reformular la pregunta. Patrones detectados en `app/domain/llm/guardrails.py`. |
| **422** | Body inválido (ej. `top_k=50` que excede `le=20`, `query=""`). | Validar contra el schema antes de enviar. |
| **401** | `Authorization` faltante. | Incluir `Authorization: Bearer ...` (en MVP cualquier valor sirve). |
| **502** | OpenAI/Gemini rate limit, key inválida o timeout. | Reintentar con backoff. Verificar `OPENAI_API_KEY`. |
| **502** | Falla en el RPC de pgvector (`match_chunks`). | Verificar que las migraciones SQL están aplicadas y que `embedding` tiene la dimensión correcta (1536). |
| **500** | Excepción no manejada. | Revisar logs del servidor (`uvicorn` con `LOG_LEVEL=DEBUG`). |

---

## Latencia esperada

| Use case | Latencia típica | Notas |
|---|---|---|
| `chatbot` (text) | 1.2 – 3 s | Con `gpt-4o-mini` y `top_k=5` |
| `chatbot` (structured) | 1.5 – 4 s | JSON estructurado tarda un poco más |
| `risk_analysis` | 8 – 20 s | Llamada multi-step, `gpt-4o` recomendado |
| `policy_generation` | 20 – 40 s | Prompt largo, output extenso |

La latencia se devuelve en `latency_ms` (solo el tiempo del LLM, no incluye
embedding ni RAG).

---

## Ver también

- [Pipeline RAG completo](../rag/full_pipeline.md)
- [Guía de integración](integration_guide.md)
- [Endpoints de Riesgo](risk.md)
- [Endpoints de Políticas](policies.md)
