# Trade-offs y Deuda Técnica

Este documento es **transparente sobre lo que sabemos que no es perfecto**
en el sistema. Cada decisión tiene un costo. Aquí los listamos para que
cualquiera que se incorpore al proyecto entienda el "por qué" detrás del
estado actual y qué planes hay para evolucionar.

---

## 1. Postgres + pgvector vs. Vector DB dedicado

### Trade-off

| | Postgres + pgvector (actual) | Pinecone / Weaviate |
|---|---|---|
| **Coste** | ✅ Una sola DB | ❌ Servicio separado, $$$ |
| **Operaciones** | ✅ Backups y monitoreo unificados | ❌ Más infra |
| **Filtros SQL** | ✅ Joins triviales con metadata | ⚠️ Limitados |
| **Recall a >1M vectores** | ✅ HNSW (migración 005) | ✅ Optimizado |
| **Latencia p99** | ⚠️ ~80-200ms con `ef_search=100` | ✅ <50ms |
| **Hybrid search (BM25 + vector)** | ⚠️ Requiere trabajo manual | ✅ Built-in |

### Estado

✅ **Aceptable hasta ~500K chunks**. Hoy estamos en ~10K. Margen amplio.
HNSW activado en migración 005 (ver
[ADR-013](design_choices.md#adr-013--chunking-section-aware--hnsw)).

### Plan de evolución

- **Corto plazo:** ✅ migrado de `ivfflat` a `hnsw` (m=16,
  ef_construction=64, ef_search=100 runtime).
- **Mediano plazo:** monitorear recall y latencia en producción; subir
  `ef_search` a 200 si hay caídas.
- **Largo plazo (>1M chunks):** migrar a vector DB dedicado solo si la
  latencia se vuelve un blocker.

---

## 2. Detección de "¿por qué?" con regex

### Trade-off

L3 (memoria de razonamiento) se activa con un **detector heurístico
regex** sobre la pregunta del usuario.

- ✅ **Cero costo** — microsegundos, sin llamada externa.
- ⚠️ **Falsos negativos** — preguntas en español sin "¿por qué" pueden
  no activarlo (ej: "Explícame esa decisión").
- ⚠️ **Falsos positivos** — preguntas que mencionan "por qué" sin
  realmente pedir explicación de algo previo.

### Estado

✅ **Aceptable para MVP.** El campo `used_reasoning_memory` en la respuesta
permite al frontend (y a los tests) saber si se activó.

### Plan de evolución

Reemplazar el regex por un **clasificador semántico ligero** (embedding
similarity contra anchor sentences) o, en última instancia, una llamada a
GPT-4o-mini con función de routing.

---

## 3. Stub de autenticación en producción

### Deuda técnica conocida

`app/core/security.py:verify_token` devuelve siempre `dev-tenant`. Esto
significa que **hoy cualquier request al API es aceptado**.

### ⛔ Riesgo

**NO desplegar a producción** sin reemplazar este stub.

### Plan

Antes de exponer el API públicamente:

1. Activar Supabase Auth (o Auth0).
2. Reemplazar `verify_token` con validación real de JWT (verificar firma,
   expiración, claims).
3. Mapear `tenant_id` desde el JWT (en lugar de hardcodear).
4. Activar **Row Level Security (RLS)** en todas las tablas. Política base:
   `tenant_id = (auth.jwt() -> 'app_metadata' ->> 'tenant_id')::uuid`.
5. Agregar tests de auth en `app/tests/test_auth.py` (no existe aún).

---

## 4. Re-ranking es un stub

### Trade-off

`app/domain/retrieval/ranking.py:rerank_chunks` actualmente solo ordena
por `similarity` descendente. **No hay un re-ranker real** (Cohere
Rerank, BGE-Reranker, cross-encoder).

- ✅ **Sin costo extra** — todo cabe en una sola query a la DB.
- ⚠️ **Calidad subóptima** — embeddings tienen recall alto pero
  precision media. Un re-ranker dedicado (ej: Cohere Rerank) elevaría la
  calidad notablemente.

### Estado

✅ **Aceptable porque** el sistema ya tiene buenos filtros (framework, país)
y `apply_per_document_cap` que da diversidad. Para ISO 42001 / NIST AI RMF,
los embeddings de OpenAI son suficientemente buenos.

### Plan

Cuando agreguemos más frameworks (>10) o más países, evaluar Cohere
Rerank v3 (multi-lingual) — probablemente +10-15% en MRR.

---

## 5. ~~Chunking simple (fixed-size)~~ (RESUELTO PARCIALMENTE)

### Trade-off original

`app/domain/documents/chunking.py:chunk_text` dividía por número de
palabras con overlap, **sin detectar secciones, encabezados, tablas
ni listas**.

### Estado

✅ **Resuelto en 2026-04-27** con el chunker section-aware. Ver
[ADR-013](design_choices.md#adr-013--chunking-section-aware--hnsw).

`chunk_text_with_sections()` detecta marcadores legales con regex
(`Artículo`, `Sección`, `Cláusula`, `Capítulo`, `Anexo`, `Apéndice`,
y sus equivalentes en inglés) y usa esos marcadores como bordes
preferidos del sliding window. Cada chunk se etiqueta con el marcador
más reciente en `document_chunks.section_ref`, que ahora viaja por el
RPC y aparece en el header del chunk en el prompt.

### Pendiente

- ⚠️ **Tablas y listas** todavía se chunkean como texto plano. Para
  documentos como GDPR Annex tables, perdemos contexto. Plan futuro:
  detectar markdown / patrones tabulares en el parser y reservarlos
  como un solo chunk.
- ⚠️ **Documentos sin marcadores formales** (NIST AI RMF Playbook,
  whitepapers) degradan a sliding window con `section_ref = NULL`.
  Aceptable.

---

## 6. ~~Sin streaming de respuestas~~ (RESUELTO)

### Trade-off original

El endpoint `/ai/query` solo retornaba la respuesta completa cuando
terminaba la generación, dando UX pobre en use cases largos
(`risk_analysis` 8-20 s, `policy_generation` 20-40 s).

### Estado

✅ **Resuelto en 2026-04-27.** Ver
[ADR-014](design_choices.md#adr-014--streaming-sse-via-stream-param).

`POST /ai/query` acepta `"stream": true` en el body y devuelve
`text/event-stream` con eventos `event: token` por cada delta del LLM,
cerrando con `event: done` con la metadata. Implementación
provider-agnostic vía `LLMRouter.stream_route()` y `StreamEvent`
normalizado.

### Pendiente

- `stream + response_mode="structured"` rechazado con 422 (decisión
  consciente, ver ADR-014).
- Errores mid-stream se reportan como `event: error` y persisten texto
  parcial en `traces`.

---

## 7. Evaluadores son heurísticos

### Trade-off

`grounding`, `length`, `citations` son heurísticas simples (regex, contar
chunks citados, ratio de longitud). **No usan un LLM-as-judge.**

- ✅ **Rápido, sin costo** — corre síncrono después de cada respuesta.
- ⚠️ **Limitado** — no detecta sutilezas como "¿la respuesta realmente
  responde la pregunta?" o "¿hay contradicción interna?".

### Estado

✅ Suficiente para detectar regresiones obvias (respuestas vacías, sin
citas, demasiado cortas/largas).

### Plan

Agregar evaluador async tipo `llm_judge`:

- Llama a GPT-4o-mini con la query, los chunks y la respuesta.
- Pide score 1-5 en: `relevance`, `faithfulness`, `completeness`.
- Persiste como `evaluation_type='llm_judge'`.

Costo estimado: ~$0.001 por respuesta. Aceptable.

---

## 8. `uploads/` en disco local

### Trade-off

Los archivos subidos se guardan en `uploads/<tenant>/<uuid>_<filename>`
en el disco del contenedor.

- ✅ **Trivial de operar** en desarrollo.
- ❌ **No sobrevive a redeploy** si no se monta volumen.
- ❌ **No funciona en horizontal scaling** — replicas no comparten disco.

### Estado

⚠️ **Ya es deuda técnica.** El primer momento que escalemos a >1 réplica,
hay que mover.

### Plan

Migrar a S3 / GCS / Supabase Storage. Cambios:

1. `app/domain/documents/service.py:upload_document` — usar SDK de S3.
2. Una env var nueva `STORAGE_BACKEND={local,s3,supabase}`.
3. Backfill de archivos existentes (script `scripts/migrate_uploads.py`).

---

## 9. Sin caché de embeddings de query

### Trade-off

Cada llamada a `/ai/query` genera embedding del query (~100ms, ~$0.00002).

- ✅ Simple.
- ⚠️ Si dos usuarios hacen la misma pregunta ("¿qué es ISO 42001?"),
  pagamos dos veces.

### Estado

✅ Volumen actual no justifica caché. Pero a >10K queries/día, sí.

### Plan

Caché Redis con key `hash(query + lang)`. TTL 24h. ROI claro.

---

## 10. Sin rate limiting

### Trade-off

No hay protección contra abuso del endpoint `/ai/query`. Un cliente
malicioso podría disparar miles de queries y consumir tokens.

### Estado

❌ **Deuda crítica antes de exponer públicamente.**

### Plan

Middleware `slowapi` (FastAPI) con límites por `tenant_id`:

- Free tier: 100 queries/día.
- Paid tier: 10K queries/día.
- Por IP también, contra abuso pre-auth.

---

## 11. Migraciones SQL ejecutadas manualmente

### Trade-off

`app/db/migrations/*.sql` se aplican copiando-pegando en el SQL editor
de Supabase. **No hay tracker** (`schema_migrations` table).

- ✅ Simple y explícito.
- ⚠️ Es fácil olvidar correr una migración en producción.
- ⚠️ Imposible saber qué versión de schema corre cada ambiente.

### Plan

Adoptar **Alembic** (estándar Python) o **Atlas**, con CI que verifique
que prod está al día.

---

## 12bis. ~~Chunks sin atribución de fuente en el prompt~~ (RESUELTO)

### Trade-off original

Los chunks recuperados llegaban al prompt como `[Chunk N | similarity:
X]` **sin nombre de documento ni framework**. Cuando el usuario
preguntaba algo específico de un framework, el modelo no podía atribuir
contenido y caía en refusals falsos.

### Estado

✅ **Resuelto en 2026-04-25.** Ver
[ADR-011](design_choices.md#adr-011--atribución-de-fuente-en-chunks).
`AIService` ahora hidrata `{filename, framework}` por `document_id` antes
del ensamblaje del prompt. Costo: una query SQL extra por request, ~5-15 ms.

---

## 12ter. ~~Prompt v3 demasiado literal~~ (RESUELTO)

### Trade-off original

El prompt v3 del chatbot tenía una lógica de decisión binaria que
provocaba refusals cuando el contexto usaba terminología distinta a la
del usuario (ej: usuario pregunta por "comité" y el doc dice
"organización interna").

### Estado

✅ **Resuelto en 2026-04-25** con el prompt v4. Ver
[ADR-012](design_choices.md#adr-012--prompt-v4-del-chatbot-con-inferencia-semántica).
v4 introduce un PASO 0 obligatorio de interpretación semántica con lista
explícita de equivalencias normativas comunes. v3 se conserva en la
tabla `prompts` con `is_active=false` para auditoría.

---

## 12. No hay multi-language para los prompts

### Trade-off

Los system prompts están en **español**. La query del usuario puede ser
en cualquier idioma — el LLM se adapta.

- ✅ Simple.
- ⚠️ Si un usuario pregunta en inglés, la respuesta a veces mezcla los
  idiomas (sobre todo en headings).

### Plan

Detectar idioma del query (heurística simple por caracteres / langdetect)
y elegir prompt según `(use_case, lang)`. La tabla `prompts` ya soporta
esto vía `version` — solo es agregar `lang` como columna.

---

## Resumen ejecutivo

| Trade-off | Severidad | Bloquea producción |
|---|---|---|
| 1. pgvector vs dedicated (HNSW activado) | Bajo | No |
| 2. L3 con regex | Bajo | No |
| 3. **Stub de auth** | **Alto** | **Sí** |
| 4. Re-ranker stub | Medio | No |
| 5. ~~Chunking simple~~ → section-aware | Bajo (residual: tablas) | No |
| 6. ~~Sin streaming~~ → SSE vía `stream=true` | — | No |
| 7. Evals heurísticos | Bajo | No |
| 8. `uploads/` local | Medio | Solo si >1 réplica |
| 9. Sin caché | Bajo | No |
| 10. **Sin rate limiting** | **Alto** | **Sí** |
| 11. Migraciones manuales | Bajo | No |
| 12. Prompts solo ES | Bajo | No |

**Bloqueadores reales para producción pública:** auth (3) y rate
limiting (10). Todo lo demás es optimización progresiva.

---

## Ver también

- [Decisiones de diseño](design_choices.md)
- [Estructura del proyecto](../project_structure.md)
- [Docker y deployment](../docker_deployment.md)
