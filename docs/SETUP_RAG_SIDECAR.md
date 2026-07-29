# Setup del RAG — Solo el sidecar (`ai-core`)

Guía **paso a paso para un dev nuevo** que quiere correr y verificar el motor
RAG de `llm_governance_ai` sin levantar el frontend.

El sidecar usa el cliente de Supabase (REST/RPC/Storage), por lo que hay dos
configuraciones compatibles:

- **Supabase Cloud + sidecar local**: la forma más rápida de comprobar el LLM.
- **Supabase local + sidecar local**: todo corre en la máquina mediante
  Supabase CLI.

El `docker-compose.yml` de esta carpeta levanta un Postgres crudo con pgvector,
pero **no incluye la API REST de Supabase**. Sirve para inspeccionar el esquema
y ejecutar SQL; por sí solo no permite probar el pipeline end-to-end.

> ¿Querés probar el RAG **desde la UI de GovAI** (frontend + Supabase Cloud +
> ngrok)? Esa es la otra guía: [`SETUP_RAG_FULL_STACK.md`](SETUP_RAG_FULL_STACK.md).

Al terminar vas a poder:

- Levantar la API FastAPI del sidecar en `http://localhost:8000`.
- Tener la base de datos con todas las tablas y funciones del RAG.
- Cargar la biblioteca normativa (`docs/admin/`) y los system prompts.
- Hacer una consulta RAG real con `curl` y ver la respuesta con citas.

---

## Tabla de contenidos

1. [¿Qué es esto? (contexto rápido)](#1-qué-es-esto-contexto-rápido)
2. [Requisitos previos](#2-requisitos-previos)
3. [Obtener el código](#3-obtener-el-código)
4. [Crear el entorno Python](#4-crear-el-entorno-python)
5. [Configurar las variables de entorno (`.env`)](#5-configurar-las-variables-de-entorno-env)
6. [Dónde obtener cada API key](#6-dónde-obtener-cada-api-key)
7. [Preparar la base de datos](#7-preparar-la-base-de-datos)
8. [Crear las tablas: correr las migraciones](#8-crear-las-tablas-correr-las-migraciones)
9. [Cargar los datos: seeds](#9-cargar-los-datos-seeds)
10. [Prender el sidecar](#10-prender-el-sidecar)
11. [Probar el RAG](#11-probar-el-rag)
12. [Troubleshooting](#12-troubleshooting)
13. [Quick reference (1 página)](#13-quick-reference-1-página)

---

## 1. ¿Qué es esto? (contexto rápido)

`ai-core` es un motor **RAG** (Retrieval Augmented Generation). En vez de que el
LLM responda "de memoria", el sistema:

1. Convierte tu pregunta en un **embedding** (vector numérico).
2. Busca en `document_chunks` los fragmentos más parecidos (búsqueda vectorial +
   por palabras clave = **híbrida**).
3. Inyecta esos fragmentos como **contexto** en el prompt del LLM.
4. El LLM responde **citando** esas fuentes.

Piezas que vas a montar:

| Pieza | Qué es | Cómo la levantás |
|---|---|---|
| **Supabase** | Postgres + pgvector + REST/RPC/Storage | Cloud o Supabase CLI |
| **Postgres crudo (opcional)** | Inspección local del esquema y SQL | Docker (`docker-compose.yml`) |
| **Migraciones** | Los `.sql` que crean tablas, índices y funciones | `psql` |
| **Seeds** | Cargan prompts + documentos normativos | Scripts Python |
| **Sidecar FastAPI** | La API en `:8000` que orquesta el RAG | Docker o `uvicorn` |

---

## 2. Requisitos previos

Instalá esto **una sola vez**. Los comandos son para macOS con
[Homebrew](https://brew.sh); abajo hay notas para Linux/Windows.

| Herramienta | Versión | Para qué | Verificar |
|---|---|---|---|
| **Python** | 3.11 | Correr el sidecar y los scripts | `python3.11 --version` |
| **Docker Desktop** | cualquiera | Supabase local o ejecución en contenedor (opcional con Cloud + uvicorn) | `docker --version` |
| **Git** | cualquiera | Clonar el repo | `git --version` |
| **psql** (cliente) | 14+ | Aplicar migraciones locales (opcional con SQL Editor) | `psql --version` |
| **Supabase CLI** | actual | Stack end-to-end 100% local (opcional con Cloud) | `supabase --version` |

```bash
# macOS (Homebrew): ejecutá cada comando por separado.
brew install python@3.11
brew install --cask docker
brew install libpq
brew link --force libpq
```

Después de instalar Docker Desktop, abrilo una vez y esperá hasta ver
**Engine running**. Tener el comando `docker` instalado no significa que el
daemon esté activo.

```bash
docker ps
psql --version
```

- **Linux (Ubuntu/Debian):** `sudo apt install python3.11 python3.11-venv postgresql-client` + [Docker Engine](https://docs.docker.com/engine/install/ubuntu/).
- **Windows:** instalá [Python 3.11](https://www.python.org/downloads/), [Docker Desktop](https://www.docker.com/products/docker-desktop/) y [psql](https://www.postgresql.org/download/windows/). Ejecutá los comandos en **WSL2** o PowerShell adaptando las rutas.

> **Nota:** el sidecar corre sobre **Python 3.11** (la imagen Docker es
> `python:3.11-slim`). Versiones más nuevas pueden romper dependencias nativas.

---

## 3. Obtener el código

Este repo (`llm_governance_ai`) normalmente vive como **submódulo** dentro de
`prism-ai-path`. Para trabajar solo el sidecar podés clonarlo directo:

```bash
git clone https://github.com/sebasfg99/llm_governance_ai.git
cd llm_governance_ai
```

> Si ya tenés `prism-ai-path` clonado, el sidecar está en
> `prism-ai-path/llm_governance_ai/`. Entrá ahí y seguí igual:
> `cd prism-ai-path/llm_governance_ai`.

Todos los comandos de esta guía se corren **desde la raíz de `llm_governance_ai/`**
salvo que se indique lo contrario.

---

## 4. Crear el entorno Python

Necesitás un virtualenv para correr las migraciones/seeds y (opcionalmente) el
sidecar sin Docker.

```bash
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

> A partir de acá, cada vez que abras una terminal nueva para este proyecto,
> corré `source .venv/bin/activate` primero.

---

## 5. Configurar las variables de entorno (`.env`)

El sidecar lee su configuración de un archivo `.env` en la raíz. Partí del
template:

```bash
cp .env.example .env
```

Abrí `.env` y completá los valores. Estos son los **obligatorios para arrancar el
RAG**:

```bash
# Generá el valor de AI_CORE_SERVICE_KEY y pegalo en .env.
openssl rand -hex 32
```

| Variable | Obligatoria | Qué es |
|---|---|---|
| `OPENAI_API_KEY` | **Sí** | Genera los embeddings (`text-embedding-3-small`) y, como fallback, respuestas de chat. **Sin esto el RAG no puede indexar ni buscar.** |
| `GEMINI_API_KEY` | Según provider | Obligatoria solo si `LLM_DEFAULT_PROVIDER=gemini` o querés usar Gemini como fallback. Debe tener facturación activa. |
| `SUPABASE_URL` | Según DB | URL del Postgres/Supabase que usa el sidecar. Para DB **100% local** ver la nota de abajo. |
| `SUPABASE_SERVICE_KEY` | Según DB | Service key con la que el sidecar lee/escribe. |
| `AI_CORE_SERVICE_KEY` | **Sí** | Secreto compartido para autenticar llamadas entrantes (32+ chars). Generá uno propio; no uses texto de ejemplo como `TU_AI_CORE_SERVICE_KEY_AQUI`. |

Variables de tuning del RAG (podés dejar los defaults):

```bash
LLM_DEFAULT_PROVIDER=openai
LLM_DEFAULT_MODEL=gpt-4o-mini
GEMINI_THINKING_BUDGET=0          # 0 = no gastar créditos en "thinking"
EMBEDDING_MODEL=text-embedding-3-small
RETRIEVAL_TOP_K=5
SIMILARITY_THRESHOLD=0.45
CHUNK_SIZE=300
CHUNK_OVERLAP=30
```

> El provider predeterminado real es el valor de `LLM_DEFAULT_PROVIDER` en tu
> `.env` (`openai` en el template actual). El router puede hacer fallback al
> otro provider si su API key también está configurada.

> **Seguridad:** `.env` está en `.gitignore` — **nunca lo commitees**. Usá claves
> propias, no compartas su valor en capturas o logs y rotá cualquier clave que
> haya sido expuesta.

### DB local vs Supabase

El cliente de FastAPI (`app/db/supabase.py`) espera un endpoint tipo Supabase.
Elegí **un solo camino** y asegurate de ejecutar migraciones y seeds en esa
misma base:

- **A) Supabase local** (recomendado, más fiel): instalá la
  [Supabase CLI](https://supabase.com/docs/guides/cli), corré `supabase start`,
  y poné en `.env` la `SUPABASE_URL` (`http://localhost:54321`) y el
  `service_role key` que imprime el comando. Las migraciones se aplican con
  `psql` contra el Postgres que levanta la CLI (puerto `54322`).
- **B) Supabase Cloud** (más rápido si ya tenés proyecto): configurá su URL y
  `service_role` key en `.env`; ejecutá las migraciones en el SQL Editor del
  mismo proyecto y luego los seeds.

> **No mezcles destinos:** si `.env` apunta a Cloud pero ejecutás las
> migraciones en `localhost:5432`, el sidecar no verá esas tablas.

> Para la prueba desde la UI con datos compartidos, usá directamente la guía
> [`SETUP_RAG_FULL_STACK.md`](SETUP_RAG_FULL_STACK.md) (Supabase Cloud).

---

## 6. Dónde obtener cada API key

| Key | Dónde sacarla | Notas |
|---|---|---|
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Necesita saldo/billing. Empieza con `sk-...`. |
| `GEMINI_API_KEY` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | La key debe pertenecer a un proyecto de Google Cloud **con facturación activa**. |
| `SUPABASE_URL` | Supabase local: lo imprime `supabase start`. Cloud: Dashboard → **Project Settings → API** | — |
| `SUPABASE_SERVICE_KEY` | Igual que arriba → **service_role** key | **Secreto.** No la expongas en el frontend. |
| `SUPABASE_JWT_SECRET` | Dashboard → **Project Settings → API → JWT Secret** | Opcional para este flujo (solo si un caller manda JWT directo). |
| `AI_CORE_SERVICE_KEY` | Lo generás vos: `openssl rand -hex 32` | Para local podés reusar el del `.env.example`. |

---

## 7. Preparar la base de datos

### Opción A — Supabase local (end-to-end local)

Desde la raíz del repo que contiene `supabase/config.toml`:

```bash
supabase start
supabase status
```

Copiá a `.env` la `API URL` (normalmente `http://localhost:54321`) y la
`service_role key` que imprime `supabase status`. El Postgres queda normalmente
en `localhost:54322`.

### Opción B — Supabase Cloud

No necesitás levantar una base local. Confirmá que `.env` contiene la URL y la
`service_role` key del mismo proyecto donde aplicarás las migraciones.

### Opción C — Postgres crudo (solo SQL/diagnóstico)

El `docker-compose.yml` del repo levanta un Postgres con la extensión `pgvector`
ya disponible. El servicio se llama **`postgres`**, no `db`:

```bash
docker compose up -d postgres
```

Verificá que esté arriba y aceptando conexiones:

```bash
docker compose ps
# El servicio de Postgres debe aparecer como "healthy"/"running"
```

Si aparece `Cannot connect to the Docker daemon`, abrí Docker Desktop y esperá
a que indique **Engine running**.

Los datos de conexión por defecto (definidos en `.env`) son:

```
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=aicore
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
```

> Este contenedor no expone PostgREST, Auth ni Storage. No apuntes
> `SUPABASE_URL` a `localhost:5432`; usalo solo para validar SQL.

---

## 8. Crear las tablas: correr las migraciones

Las migraciones viven en `app/db/migrations/` y **deben aplicarse en orden**
(`001` → `008`). Cada una es idempotente (usa `IF NOT EXISTS` / `OR REPLACE`),
así que es seguro re-ejecutarlas.

### Qué crea cada migración

| Archivo | Qué hace |
|---|---|
| `001_initial.sql` | Esquema base: `tenants`, `documents`, `document_chunks` (con `embedding vector(1536)` + índice ivfflat), `prompts`, `traces`, `feedback`, `evaluations` y las funciones `match_chunks` / `match_global_chunks`. |
| `002_add_country.sql` | Columna `country` para filtrar retrieval por país (`NULL` = global). |
| `003_fix_ivfflat_probes.sql` | Ajusta `ivfflat.probes` para que la búsqueda no descarte silenciosamente chunks de documentos chicos. |
| `004_projects_memory_framework.sql` | Memoria L2 (`projects`) + memoria L1 (`threads`/`messages`) + columna `framework`. |
| `005_section_ref_hnsw.sql` | Columna `section_ref` (citas a nivel de cláusula) + índice **HNSW** en reemplazo de ivfflat. |
| `006_rag_governance.sql` | Gobernanza: costo/tokens/riesgo en `traces`, 8 métricas en `evaluations`, tabla `response_risks`, **búsqueda híbrida** (`content_tsv` + RPCs `match_chunks_hybrid` / `match_global_chunks_hybrid`) y la vista `rag_metrics_v`. |
| `007_fix_hybrid_rpc_types.sql` | Hotfix de tipos (`real` vs `double precision`) en los RPCs híbridos de la 006. |
| `008_industry_codes.sql` | Columna `industry_codes` para filtrar documentos por sector (Capa 3). |
| `009_agent_tags.sql` | Columna `agent_tags` (+ índice GIN) para filtrar por agente de negocio, y parámetro `p_agent_tags` en los cuatro RPCs de retrieval. Requerida para usar `agent_tags` en `POST /ai/query` — ver [`rag/agent_tags.md`](rag/agent_tags.md). Después de aplicarla hay que correr `python scripts/seed_documents.py --force` para etiquetar el corpus. |
| `010_industry_filter_fix.sql` | **Corrige un bug silencioso**: el filtro por `industry_codes` nunca filtró nada. Agrega `p_industry_codes` a los cuatro RPCs, devuelve la columna `industry_codes` en el resultado, y crea el índice GIN que faltaba. Sin esta migración una cooperativa puede recibir contenido exclusivo de bancos, sin error ni log — ver [`rag/industry_codes.md`](rag/industry_codes.md). |

### Aplicarlas en Supabase local o Postgres de diagnóstico

Usá puerto `54322` para Supabase local o `5432` para el Postgres crudo del
`docker-compose.yml`:

```bash
export PGPASSWORD=postgres
DB_PORT=54322
for f in app/db/migrations/0*.sql; do
  echo "── aplicando $f"
  psql -h localhost -p "$DB_PORT" -U postgres -d postgres \
    -v ON_ERROR_STOP=1 -f "$f"
done
```

Para el Postgres crudo, cambiá `DB_PORT=5432` y `-d postgres` por `-d aicore`.

> En versiones antiguas de la migración 004 puede aparecer
> `function name "match_chunks" is not unique` en un `COMMENT ON FUNCTION`.
> Las tablas ya creadas permanecen y la migración 005 recrea las funciones.
> Continuá hasta 008 y verificá el esquema final. No ignores otros errores.

### Aplicarlas en Supabase Cloud

Abrí **SQL Editor**, pegá el contenido de cada archivo `001` → `008` en orden y
hacé **Run** uno por uno. No ejecutes estas migraciones únicamente en Docker si
el `SUPABASE_URL` de `.env` apunta a Cloud.

Verificá que las tablas existan:

```bash
psql -h localhost -p "$DB_PORT" -U postgres -d postgres -c "\dt"
# Deberías ver: tenants, documents, document_chunks, prompts, traces,
# feedback, evaluations, projects, threads, messages, response_risks
```

En Cloud, verificá las mismas tablas desde **Table Editor**.

---

## 9. Cargar los datos: seeds

Con las tablas creadas, cargá los **prompts** y los **documentos normativos**.
Corré los scripts desde la raíz del repo con el venv activo.

### 9.1 Prompts (obligatorio)

Inserta los system prompts versionados (uno por `use_case`: `chatbot`,
`executive_summary`, `policy_generation`, `risk_intake`, `risk_analysis`,
`controls_generation`).

```bash
PYTHONPATH=. python scripts/seed_prompts.py
```

Es idempotente: si ya existe un prompt activo para un `use_case`, lo respeta.

### 9.2 Documentos (biblioteca RAG)

El corpus normativo vive en `docs/admin/`, organizado por marco y país:

```
docs/admin/{Framework}/file.pdf              → global (todos los países)
docs/admin/{Framework}/{País}/file.pdf       → específico de ese país
```

Primero comprobá que el corpus exista:

```bash
ls -R docs/admin
```

Si la carpeta está vacía o no existe, pedile el corpus al owner o agregá
documentos propios respetando la estructura. Sin documentos, el LLM puede
responder, pero no habrá retrieval ni citas que validen el RAG.

Cargalos:

```bash
PYTHONPATH=. python scripts/seed_documents.py
```

Esto, por cada archivo: extrae el texto → lo parte en chunks (section-aware) →
genera embeddings con OpenAI → los inserta en `document_chunks` bajo
`tenant_id='global'`. Es idempotente (salta los ya procesados).

Opciones útiles:

```bash
# Re-chunkear TODO con el chunker section-aware (necesario tras migración 005/008)
PYTHONPATH=. python scripts/seed_documents.py --force

# Cargar desde otra carpeta
PYTHONPATH=. python scripts/seed_documents.py --folder /ruta/a/mis/docs
```

> El seed de documentos **consume OpenAI** (embeddings). Con el corpus completo
> puede tardar varios minutos y tener un costo pequeño pero real.

Verificá que se cargaron. Para Supabase local:

```bash
psql -h localhost -p 54322 -U postgres -d postgres \
  -c "SELECT status, count(*) FROM documents GROUP BY status;"
psql -h localhost -p 54322 -U postgres -d postgres \
  -c "SELECT count(*) AS chunks_con_embedding FROM document_chunks WHERE embedding IS NOT NULL;"
```

En Cloud, ejecutá esas dos consultas en el SQL Editor.

---

## 10. Prender el sidecar

Dos opciones. Para iterar rápido usá **uvicorn**; para replicar producción usá
**Docker**.

### Opción A — uvicorn (recomendado para dev)

```bash
source .venv/bin/activate
uvicorn app.main:app --port 8000 --reload
```

### Opción B — Docker (clone standalone)

```bash
docker build -t ai-core:local .
docker run --rm --name ai-core -p 8000:8000 --env-file .env ai-core:local
```

Si trabajás dentro de `prism-ai-path`, también podés ejecutar desde la raíz:

```bash
docker compose -f docker-compose.ai-core.yml up --build ai-core
```

Ese compose tiene `restart: unless-stopped`. Para volver a `uvicorn` manual y
liberar el puerto 8000, detenelo con:

```bash
docker compose -f docker-compose.ai-core.yml down
```

### Verificá que arrancó

En otra terminal:

```bash
curl http://localhost:8000/health
# → {"status":"ok","version":"0.1.0","service":"ai-core"}

curl http://localhost:8000/ready
# → {"status":"ready","service":"ai-core"}
```

Si `/health` responde pero `/ready` falla, casi siempre es un problema de
conexión a la DB o de keys faltantes (ver [Troubleshooting](#12-troubleshooting)).

Si `uvicorn` muestra `[Errno 48] Address already in use`, identificá primero al
dueño del puerto:

```bash
lsof -i :8000
docker ps --format 'table {{.Names}}\t{{.Ports}}'
```

No mates PIDs repetidamente si Docker los vuelve a crear; bajá el compose que
publica `8000:8000`.

---

## 11. Probar el RAG

Hacé una consulta real al endpoint principal `POST /ai/query`. La autenticación
actual **no acepta** `Bearer dev-token`. Para una prueba local usá Path A:
`AI_CORE_SERVICE_KEY` + `X-Tenant-Id` + `X-User-Id`.

```bash
KEY=$(grep '^AI_CORE_SERVICE_KEY=' .env | cut -d= -f2-)

curl -X POST http://localhost:8000/ai/query \
  -H "Authorization: Bearer $KEY" \
  -H 'X-Tenant-Id: dev-tenant' \
  -H 'X-User-Id: dev-user' \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "¿Qué exige ISO 42001 sobre el análisis de riesgos de IA?",
    "use_case": "chatbot",
    "top_k": 5,
    "include_global": true,
    "framework_filter": ["iso_42001"]
  }'
```

No escribas literalmente `TU_AI_CORE_SERVICE_KEY_AQUI`: es un placeholder. Si
la respuesta es `401 Unauthorized`, comprobá que `$KEY` no esté vacío y
reiniciá el sidecar después de cambiar `.env`.

Respuesta esperada (resumida):

```jsonc
{
  "response_text": "Según ISO 42001... [Fuente: ISO42001.pdf]",
  "trace_id": "…",
  "model_provider": "gemini",
  "model_name": "gemini-2.5-flash",
  "latency_ms": 1840
}
```

Probar **streaming** (SSE, token a token):

```bash
curl -N -X POST http://localhost:8000/ai/query \
  -H "Authorization: Bearer $KEY" \
  -H 'X-Tenant-Id: dev-tenant' \
  -H 'X-User-Id: dev-user' \
  -H 'Content-Type: application/json' \
  -d '{"query":"Resumen de OWASP LLM Top 10","use_case":"chatbot","stream":true}'
```

Si ves respuestas **con citas** (`[Fuente: ...]`), el pipeline RAG completo
—embeddings, retrieval híbrido, generación y trazas— está funcionando.

> Documentación completa del endpoint en [`docs/api/ai.md`](api/ai.md) y del
> pipeline en [`docs/rag/full_pipeline.md`](rag/full_pipeline.md).

---

## 12. Troubleshooting

| Síntoma | Causa probable | Solución |
|---|---|---|
| `no such service: db` | El nombre de servicio de la guía antigua era incorrecto | Usá `docker compose up -d postgres`. |
| `Cannot connect to the Docker daemon` | Docker Desktop está instalado pero el engine no está activo | Abrí Docker Desktop, esperá **Engine running** y verificá con `docker ps`. |
| `/ready` devuelve 5xx | El sidecar no puede conectar a Supabase | Verificá `SUPABASE_URL`/keys en `.env`; un Postgres crudo en `:5432` no ofrece la API de Supabase. |
| `/ai/query` devuelve `401 Unauthorized` | Se usó `dev-token`, un placeholder o faltan headers | Usá el valor real de `AI_CORE_SERVICE_KEY` y mandá también `X-Tenant-Id`/`X-User-Id`; reiniciá el sidecar tras cambiar `.env`. |
| `relation "document_chunks" does not exist` | Faltan migraciones | Corré el paso 8 en orden `001`→`008`. |
| `function name "match_chunks" is not unique` en migración 004 | `COMMENT ON FUNCTION` ambiguo en una versión antigua | Continuá con 005→008, que recrean las firmas, y verificá el esquema final. |
| Respuestas sin citas o "no tengo contexto" | No hay chunks con embedding | Corré `seed_documents.py`; verificá `SELECT count(*) FROM document_chunks WHERE embedding IS NOT NULL`. |
| `structure of query does not match function result type` | Falta el hotfix de tipos | Aplicá `007_fix_hybrid_rpc_types.sql`. |
| Todas las respuestas caen a OpenAI / errores 429 de Gemini | `GEMINI_API_KEY` sin billing | Usá una key de un proyecto con facturación activa; mantené `GEMINI_THINKING_BUDGET=0`. |
| `openai.AuthenticationError` | `OPENAI_API_KEY` inválida o sin saldo | Regenerá la key en el dashboard de OpenAI y verificá el billing. |
| El seed de documentos "no hace nada" | Ya están procesados (idempotente) | Usá `--force` para re-procesar. |
| `psql: command not found` | Falta el cliente | macOS: `brew install libpq && brew link --force libpq`. |
| Puerto `8000` ocupado y el PID reaparece | Un compose con `restart: unless-stopped` reinicia `ai-core` | Desde `prism-ai-path`: `docker compose -f docker-compose.ai-core.yml down`. |
| `zsh: parse error` al copiar comandos | Se pegó un placeholder como `kill <PID>` o una `\` con espacios después | Sustituí placeholders por valores reales y dejá la barra `\` como último carácter de la línea. |

---

## 13. Quick reference (1 página)

```bash
# 0. Estar en llm_governance_ai/

# 1. Entorno Python
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configurar un destino Supabase (Cloud o CLI local)
cp .env.example .env      # completar OPENAI_API_KEY, GEMINI_API_KEY,
                          # SUPABASE_URL, SUPABASE_SERVICE_KEY, AI_CORE_SERVICE_KEY
openssl rand -hex 32      # pegar el resultado como AI_CORE_SERVICE_KEY

# 3A. Supabase Cloud: ejecutar 001→008 en SQL Editor.
# 3B. Supabase local:
supabase start
export PGPASSWORD=postgres
DB_PORT=54322
for f in app/db/migrations/0*.sql; do
  psql -h localhost -p "$DB_PORT" -U postgres -d postgres \
    -v ON_ERROR_STOP=1 -f "$f"
done

# 4. Seeds (se escriben en el SUPABASE_URL de .env)
PYTHONPATH=. python scripts/seed_prompts.py
PYTHONPATH=. python scripts/seed_documents.py

# 5. Sidecar
uvicorn app.main:app --port 8000 --reload

# 6. Verificar (en otra terminal)
curl http://localhost:8000/ready

# 7. Probar RAG con la autenticación actual
KEY=$(grep '^AI_CORE_SERVICE_KEY=' .env | cut -d= -f2-)
curl -X POST http://localhost:8000/ai/query \
  -H "Authorization: Bearer $KEY" \
  -H 'X-Tenant-Id: dev-tenant' \
  -H 'X-User-Id: dev-user' \
  -H 'Content-Type: application/json' \
  -d '{"query":"¿Qué exige ISO 42001 sobre riesgos de IA?","use_case":"chatbot"}'
```
