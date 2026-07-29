# Setup del RAG — Stack completo (UI + sidecar + Supabase Cloud)

Guía **paso a paso para un dev nuevo** que quiere probar el RAG **desde la UI de
GovAI**: frontend React + sidecar `ai-core` local + **Supabase Cloud**
compartido + túnel **ngrok** (o Railway).

> ¿Solo querés el motor RAG aislado, con `curl` y sin UI? Esa es la otra guía:
> [`SETUP_RAG_SIDECAR.md`](SETUP_RAG_SIDECAR.md).

## Cómo encajan las piezas

```
Navegador → Frontend (localhost:8080)
                │
                ▼
        Supabase Cloud (oqahrrauhbleyxaianli)
        · Auth · Postgres · Edge Function "agent-chat"
                │  (necesita una URL pública para llegar a tu máquina)
                ▼
         ngrok / Railway  →  Sidecar ai-core (localhost:8000)
                                     │
                                     ▼
                              OpenAI + Gemini
```

**Por qué ngrok:** Supabase Cloud corre en internet y **no puede llamar a tu
`localhost`** directamente. ngrok crea una URL pública que hace de puente hacia
tu sidecar local. La alternativa permanente es desplegar el sidecar en Railway.

---

## Tabla de contenidos

1. [Requisitos previos](#1-requisitos-previos)
2. [Clonar el repo con el submódulo](#2-clonar-el-repo-con-el-submódulo)
3. [Archivos de entorno (`.env`)](#3-archivos-de-entorno-env)
4. [Dónde obtener cada API key](#4-dónde-obtener-cada-api-key)
5. [Preparar Supabase Cloud (tablas + datos)](#5-preparar-supabase-cloud-tablas--datos)
6. [Deploy del Edge Function `agent-chat`](#6-deploy-del-edge-function-agent-chat)
7. [Levantar el stack (3 terminales)](#7-levantar-el-stack-3-terminales)
8. [Instalar y configurar ngrok](#8-instalar-y-configurar-ngrok)
9. [Conectar ngrok con Supabase](#9-conectar-ngrok-con-supabase)
10. [Alternativa sin ngrok: Railway](#10-alternativa-sin-ngrok-railway)
11. [Probar el RAG desde la UI](#11-probar-el-rag-desde-la-ui)
12. [Troubleshooting](#12-troubleshooting)
13. [Quick reference (1 página)](#13-quick-reference-1-página)

---

## 1. Requisitos previos

Instalá una sola vez (comandos para macOS/Homebrew; abajo notas para otros SO):

| Herramienta | Versión | Para qué | Verificar |
|---|---|---|---|
| **Node.js** | 18+ | Frontend React (Vite) | `node --version` |
| **npm** | 9+ | Dependencias del frontend | `npm --version` |
| **Docker Desktop** | cualquiera | Correr el sidecar `ai-core` | `docker --version` |
| **ngrok** | cualquiera | Túnel público → sidecar local | `ngrok --version` |
| **Git** | cualquiera | Clonar el repo + submódulo | `git --version` |

```bash
# macOS (Homebrew)
brew install node
brew install --cask docker      # abrí Docker Desktop una vez
brew install ngrok
```

- **Windows:** [Node.js](https://nodejs.org) (LTS), [Docker Desktop](https://www.docker.com/products/docker-desktop/), [ngrok](https://ngrok.com/download).
- **Linux:** Node vía [nodesource](https://github.com/nodesource/distributions), [Docker Engine](https://docs.docker.com/engine/install/ubuntu/), ngrok desde [ngrok.com/download](https://ngrok.com/download).

Además necesitás:

- Acceso al proyecto **Supabase Cloud** `oqahrrauhbleyxaianli` (pedíselo al owner).
- Los archivos `.env` (frontend + sidecar) — el owner te los pasa.

---

## 2. Clonar el repo con el submódulo

El frontend, las Edge Functions y los scripts SQL viven en `prism-ai-path`. El
sidecar `ai-core` está incluido como **submódulo** en
`prism-ai-path/llm_governance_ai/`.

```bash
git clone --recurse-submodules https://github.com/dougvillav/prism-ai-path.git
cd prism-ai-path
```

Si ya lo clonaste sin `--recurse-submodules`, inicializá el submódulo:

```bash
git submodule update --init --recursive
```

Verificá que el sidecar esté presente:

```bash
ls llm_governance_ai/app/main.py     # debe existir
```

Todos los comandos de esta guía se corren **desde la raíz de `prism-ai-path/`**
salvo que se indique lo contrario.

---

## 3. Archivos de entorno (`.env`)

Necesitás **dos** archivos `.env`. Pedile los valores al owner del proyecto.

### 3.1 Frontend — `prism-ai-path/.env`

```bash
VITE_SUPABASE_URL=https://oqahrrauhbleyxaianli.supabase.co
VITE_SUPABASE_PUBLISHABLE_KEY=<pedir al owner>
VITE_SUPABASE_PROJECT_ID=oqahrrauhbleyxaianli
```

### 3.2 Sidecar — `prism-ai-path/llm_governance_ai/.env`

Partí del template y completá:

```bash
cp llm_governance_ai/.env.example llm_governance_ai/.env
```

```bash
SUPABASE_URL=https://oqahrrauhbleyxaianli.supabase.co
SUPABASE_SERVICE_KEY=<pedir al owner>
AI_CORE_SERVICE_KEY=<pedir al owner — debe coincidir con el secret del Edge Function>

# --- Proveedores LLM (ambos obligatorios) ---
OPENAI_API_KEY=<propia o del owner>
GEMINI_API_KEY=<de un proyecto Google Cloud CON billing activo>
LLM_DEFAULT_PROVIDER=gemini
LLM_DEFAULT_MODEL=gemini-2.5-flash
GEMINI_THINKING_BUDGET=0
```

> **Proveedores LLM:** el router usa **Gemini por defecto** con **fallback
> automático a OpenAI**. Necesitás **ambas** claves. La de Gemini debe ser de un
> proyecto **con facturación activa** o todo caerá a OpenAI. Mantené
> `GEMINI_THINKING_BUDGET=0` para no quemar créditos.

> **Clave:** el `AI_CORE_SERVICE_KEY` del sidecar debe ser **idéntico** al secret
> `AI_CORE_SERVICE_KEY` configurado en Supabase (ver [paso 6](#6-deploy-del-edge-function-agent-chat)),
> o el Edge Function no podrá autenticarse contra el sidecar.

> `.env` está en `.gitignore` — nunca lo commitees.

---

## 4. Dónde obtener cada API key

| Key | Dónde | Notas |
|---|---|---|
| `VITE_SUPABASE_PUBLISHABLE_KEY` | Supabase → **Project Settings → API** → anon/publishable | Pública, va en el frontend. |
| `SUPABASE_SERVICE_KEY` | Supabase → **Project Settings → API** → service_role | **Secreta.** Solo backend/sidecar. |
| `AI_CORE_SERVICE_KEY` | El owner la definió; o generá una nueva con `openssl rand -hex 32` | Debe coincidir en `.env` **y** en el secret de Supabase. |
| `OPENAI_API_KEY` | [platform.openai.com/api-keys](https://platform.openai.com/api-keys) | Necesita saldo. |
| `GEMINI_API_KEY` | [aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) | Proyecto con billing activo. |
| `SUPABASE_ACCESS_TOKEN` | [supabase.com](https://supabase.com) → avatar → **Account → Access Tokens** | Solo para deployar Edge Functions (paso 6). Empieza con `sbp_...`. |

---

## 5. Preparar Supabase Cloud (tablas + datos)

El proyecto Cloud `oqahrrauhbleyxaianli` normalmente **ya tiene** el esquema del
sidecar y del frontend aplicado. Si estás partiendo de un proyecto limpio o
faltan tablas del RAG, aplicá las migraciones del sidecar y el seed demo.

### 5.1 Migraciones del sidecar (RAG)

En **Supabase Dashboard → SQL Editor**, pegá y ejecutá **en orden** el contenido
de cada archivo de `llm_governance_ai/app/db/migrations/`:

```
001_initial.sql  →  002_add_country.sql  →  003_fix_ivfflat_probes.sql
→ 004_projects_memory_framework.sql  →  005_section_ref_hnsw.sql
→ 006_rag_governance.sql  →  007_fix_hybrid_rpc_types.sql  →  008_industry_codes.sql
→ 009_agent_tags.sql  →  010_industry_filter_fix.sql
```

Estas crean `documents`, `document_chunks` (pgvector), `prompts`, `traces`, la
búsqueda híbrida, etc. Detalle de cada una en
[`SETUP_RAG_SIDECAR.md` §8](SETUP_RAG_SIDECAR.md#8-crear-las-tablas-correr-las-migraciones).

### 5.2 Cargar prompts y documentos (biblioteca RAG)

Desde `llm_governance_ai/`, con el `.env` apuntando a **Cloud** y el venv
activo (ver [SETUP_RAG_SIDECAR §4](SETUP_RAG_SIDECAR.md#4-crear-el-entorno-python)):

```bash
cd llm_governance_ai
source .venv/bin/activate
PYTHONPATH=. python scripts/seed_prompts.py
PYTHONPATH=. python scripts/seed_documents.py
cd ..
```

Los documentos de `docs/admin/` están versionados en el repo del sidecar, así
que ya los tenés tras clonar.

### 5.3 Datos demo del frontend (usuarios, institución, proyecto)

En **SQL Editor** pegá y ejecutá `scripts/seed_demo_users.sql` (raíz de
`prism-ai-path`). Crea la institución **Banco Ficticio de Pruebas S.A.** con 4
usuarios demo (password `Test1234!`):

| Rol | Email |
|---|---|
| AI Governance Officer | `co-demo@test.com` |
| Project Owner | `pl-demo@test.com` |
| AI Engineer | `tl-demo@test.com` |
| Executive Sponsor | `exec-demo@test.com` |

---

## 6. Deploy del Edge Function `agent-chat`

El Edge Function `agent-chat` es el que la UI llama; él a su vez reenvía al
sidecar. Necesita el secret `AI_CORE_SERVICE_KEY`.

```bash
# 1. Token de Supabase para deployar (Account → Access Tokens)
export SUPABASE_ACCESS_TOKEN='sbp_TU_TOKEN'

# 2. Setear el secret (mismo valor que en llm_governance_ai/.env)
npx supabase secrets set AI_CORE_SERVICE_KEY=<tu-valor> \
  --project-ref oqahrrauhbleyxaianli

# 3. Deploy del/los Edge Functions
for fn in agent-chat ai-core-proxy ai-upload-document ai-process-document \
          ai-search-documents ai-delete-document; do
  npx supabase functions deploy $fn --project-ref oqahrrauhbleyxaianli --use-api
done
```

Verificá los secrets:

```bash
npx supabase secrets list --project-ref oqahrrauhbleyxaianli
```

> Si el proyecto Cloud ya tiene estos Edge Functions desplegados y el secret
> configurado, podés saltar este paso. Detalles en
> [`../../DEPLOY_AI_CORE.md`](../../DEPLOY_AI_CORE.md).

---

## 7. Levantar el stack (3 terminales)

Necesitás **tres terminales** abiertas en simultáneo.

### Terminal 1 — Frontend

```bash
cd prism-ai-path
npm install            # solo la primera vez
npm run dev
```

App disponible en **http://localhost:8080**.

### Terminal 2 — Sidecar `ai-core`

```bash
cd prism-ai-path
docker compose -f docker-compose.ai-core.yml up --build
```

Este compose apunta el sidecar al **mismo Supabase Cloud** que usa la app.
Verificá en otra terminal:

```bash
curl http://localhost:8000/ready
# → {"status":"ready","service":"ai-core"}
```

### Terminal 3 — ngrok

Ver el [paso 8](#8-instalar-y-configurar-ngrok).

---

## 8. Instalar y configurar ngrok

### 8.1 Crear cuenta y authtoken

1. Registrate gratis en [dashboard.ngrok.com](https://dashboard.ngrok.com/signup).
2. Copiá tu authtoken desde **Your Authtoken** en el dashboard.
3. Configurá ngrok (una sola vez):

```bash
ngrok config add-authtoken TU_AUTHTOKEN
```

### 8.2 Levantar el túnel

```bash
ngrok http 8000
```

ngrok muestra algo como:

```
Forwarding  https://abc123.ngrok-free.app -> http://localhost:8000
```

**Copiá esa URL HTTPS** (`https://abc123.ngrok-free.app`).

> **Ojo (plan free):** la URL **cambia cada vez que reiniciás ngrok**. Cada vez
> que lo relances tenés que actualizar `ai_core_url` en Supabase (paso 9). Un
> dominio fijo requiere plan pago de ngrok o usar Railway (paso 10).

---

## 9. Conectar ngrok con Supabase

Decile a la app cuál es la URL pública del sidecar. Dos opciones:

### Opción A — SQL Editor

En **Supabase Dashboard → SQL Editor** (reemplazá la URL por la tuya):

```sql
UPDATE platform_settings SET value = 'https://abc123.ngrok-free.app' WHERE id = 'ai_core_url';
UPDATE platform_settings SET value = 'true' WHERE id = 'ai_core_enabled';
UPDATE platform_settings SET value = 'true' WHERE id = 'agent_proactive_messages';
UPDATE platform_settings SET value = 'true' WHERE id = 'rag_enabled';
```

### Opción B — desde la app

1. Entrá a la app como admin → **Admin → Platform Settings**.
2. Pegá la URL de ngrok en `ai_core_url`.
3. Activá `ai_core_enabled = true`.

> Repetí este paso **cada vez que reinicies ngrok** (la URL cambia).

---

## 10. Alternativa sin ngrok: Railway

Para que el sidecar esté siempre disponible sin levantar ngrok cada vez,
desplegalo en [Railway](https://railway.app) (el repo trae `railway.json` y
`Dockerfile`).

1. Creá un servicio en Railway apuntando al repo `llm_governance_ai`.
2. Configurá estas variables en Railway:
   - `SUPABASE_URL`, `SUPABASE_SERVICE_KEY`, `AI_CORE_SERVICE_KEY`
   - `OPENAI_API_KEY`, `GEMINI_API_KEY` (con billing)
   - `LLM_DEFAULT_PROVIDER=gemini`, `LLM_DEFAULT_MODEL=gemini-2.5-flash`
   - `GEMINI_THINKING_BUDGET=0`
3. Tomá la URL permanente (ej. `https://tu-servicio.railway.app`).
4. Seteala **una sola vez** en `platform_settings.ai_core_url` (paso 9).

Con Railway te saltás la Terminal 3 (ngrok) y la URL no cambia. Detalles en
[`../../DEPLOY_AI_CORE.md`](../../DEPLOY_AI_CORE.md).

---

## 11. Probar el RAG desde la UI

1. Abrí **http://localhost:8080**.
2. Iniciá sesión con un usuario demo (ej. `co-demo@test.com` / `Test1234!`).
3. Abrí el **asistente / chat lateral** y preguntá algo normativo, por ejemplo:
   *"¿Qué exige ISO 42001 sobre el análisis de riesgos de IA?"*
4. Deberías ver una respuesta **en streaming** y **con citas**
   (`[Fuente: ...]`) a los documentos de la biblioteca.

### Checklist si algo no responde

- Terminal 2: `curl http://localhost:8000/ready` responde `ready`.
- Terminal 3: ngrok sigue vivo y la URL coincide con `ai_core_url`.
- Supabase: `ai_core_enabled = true` y `rag_enabled = true`.
- Docker Desktop está corriendo.

---

## 12. Troubleshooting

| Síntoma | Causa probable | Solución |
|---|---|---|
| El chat no responde | ngrok reiniciado → URL vieja | Actualizá `ai_core_url` con la nueva URL (paso 9). |
| `502`/timeout desde el Edge Function | Sidecar caído o URL incorrecta | Verificá Terminal 2 (`/ready`) y que `ai_core_url` apunte al ngrok actual. |
| `401` del sidecar | `AI_CORE_SERVICE_KEY` no coincide | El valor en `llm_governance_ai/.env` debe ser igual al secret de Supabase (paso 6). |
| Respuestas sin citas | Falta la biblioteca RAG | Corré `seed_documents.py` contra Cloud (paso 5.2). |
| Login demo falla | Falta el seed demo | Ejecutá `scripts/seed_demo_users.sql` (paso 5.3). |
| Errores 429 de Gemini | Key sin billing | Usá una `GEMINI_API_KEY` con facturación activa. |
| `ai_core_enabled` no existe | Falta la fila en `platform_settings` | Insertá/actualizá las filas de settings (paso 9). |
| Frontend no levanta | Dependencias | `npm install` de nuevo; Node 18+. |
| Latencia alta (15–30s) | Cadena edge→ngrok→sidecar | Considerá Railway (paso 10) o bajá `top_k`. |

---

## 13. Quick reference (1 página)

```bash
# 0. Requisitos: Node 18+, Docker, ngrok, Git. Acceso a Supabase Cloud + .env

# 1. Clonar con submódulo
git clone --recurse-submodules https://github.com/dougvillav/prism-ai-path.git
cd prism-ai-path

# 2. Envs
#   prism-ai-path/.env                     (frontend)
#   prism-ai-path/llm_governance_ai/.env   (sidecar; AI_CORE_SERVICE_KEY == secret Supabase)

# 3. (si el Cloud está limpio) migraciones 001→008 + seeds + seed_demo_users.sql

# 4. Edge Function (si no está desplegado)
export SUPABASE_ACCESS_TOKEN='sbp_...'
npx supabase secrets set AI_CORE_SERVICE_KEY=<valor> --project-ref oqahrrauhbleyxaianli
npx supabase functions deploy agent-chat --project-ref oqahrrauhbleyxaianli --use-api

# 5. Tres terminales
npm install && npm run dev                                   # T1 frontend  :8080
docker compose -f docker-compose.ai-core.yml up --build      # T2 sidecar   :8000
ngrok http 8000                                              # T3 túnel

# 6. Conectar ngrok (SQL Editor) — repetir cada reinicio de ngrok
#   UPDATE platform_settings SET value='https://XXX.ngrok-free.app' WHERE id='ai_core_url';
#   UPDATE platform_settings SET value='true' WHERE id='ai_core_enabled';

# 7. Probar: login demo en http://localhost:8080 y preguntar al asistente
```
