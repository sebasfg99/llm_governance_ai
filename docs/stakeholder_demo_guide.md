# Guia de Demo para Stakeholders — AI Governance Platform

## 1. Como levantar el programa

### Pre-requisitos

- Python 3.11
- Cuenta en Supabase (supabase.com) con acceso al SQL Editor
- OpenAI API Key (platform.openai.com)
- Git

### Pasos de instalacion

**1. Clonar el repositorio y crear entorno virtual**

```bash
git clone <repo-url> llm_ai_governance_project
cd llm_ai_governance_project

python3.11 -m venv .venv
source .venv/bin/activate        # macOS / Linux
# .venv\Scripts\activate         # Windows

pip install --upgrade pip
pip install -r requirements.txt
```

**2. Configurar variables de entorno**

```bash
cp .env.example .env
```

Abrir `.env` y completar los siguientes valores:

```env
SUPABASE_URL=https://<tu-proyecto>.supabase.co
SUPABASE_SERVICE_KEY=<service-role-key>   # No la anon key, la service_role key
OPENAI_API_KEY=sk-...
LLM_DEFAULT_PROVIDER=openai
LLM_DEFAULT_MODEL=gpt-4o-mini
EMBEDDING_MODEL=text-embedding-3-small
```

**3. Correr las migraciones SQL en Supabase**

Ir a Supabase > SQL Editor y ejecutar los siguientes archivos en orden:

```
app/db/migrations/001_initial.sql
app/db/migrations/002_add_country.sql
app/db/migrations/003_fix_ivfflat_probes.sql
app/db/migrations/004_projects_memory_framework.sql
```

**4. Cargar datos semilla**

```bash
PYTHONPATH=. python scripts/seed_prompts.py
PYTHONPATH=. python scripts/seed_documents.py
```

El segundo script carga los documentos regulatorios desde `docs/admin/`. Si la carpeta esta vacia, el RAG no tendra corpus y respondera sin contexto.

**5. Levantar el servidor**

```bash
uvicorn app.main:app --reload
```

El servidor queda disponible en:
- API base: `http://localhost:8000`
- Swagger UI (interfaz interactiva): `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 2. Guia de testing paso a paso para stakeholders

Todos los flujos a continuacion se pueden ejecutar desde Swagger UI en `http://localhost:8000/docs`.

En cada llamada que requiera autenticacion, usar los headers:
```
Authorization: Bearer <AI_CORE_SERVICE_KEY real>
X-Tenant-Id: dev-tenant
X-User-Id: dev-user
```

`Bearer dev-token` ya no es valido. Como alternativa, puede usarse un JWT real
de Supabase en `Authorization`, sin los headers `X-*`.

---

### Flujo A — Consulta RAG regulatoria (el flujo principal)

Muestra la capacidad central: responder preguntas sobre marcos regulatorios usando el corpus cargado.

**Paso 1 — Verificar que el servidor esta activo**

```
GET /health
```

Respuesta esperada: `{ "status": "ok" }`

**Paso 2 — Hacer una consulta regulatoria**

```
POST /ai/query
```

Body de ejemplo:
```json


```

Que verificar en la respuesta:
- `response`: la respuesta del modelo con texto y citaciones
- `trace_id`: identificador unico del registro de auditoria
- `thread_id`: ID de la conversacion creada automaticamente
- `retrieved_chunks`: fragmentos del corpus usados como contexto

---

### Flujo B — Gestion de documentos propios

Muestra que el sistema puede ingerir documentos del cliente y hacerlos consultables.

**Paso 1 — Subir un documento**

```
POST /documents/upload
```

Parametros (form-data):
- `file`: archivo PDF, DOCX o TXT
- `tenant_id`: `dev-tenant`
- `country`: `costa-rica` (o el que corresponda)

Respuesta: `document_id`

**Paso 2 — Procesar el documento (parsear, chunkear, embeber)**

```
POST /documents/{document_id}/process
```

Este paso puede tardar algunos segundos segun el tamano del archivo. El documento pasa por los estados: `pending → processing → ready`.

**Paso 3 — Verificar que el documento esta listo**

```
GET /documents/{document_id}
```

Confirmar que `status` es `ready`.

**Paso 4 — Consultar usando el documento recien procesado**

```
POST /ai/query
```

Body:
```json
{
  "query": "Que dice el documento sobre politicas de privacidad?",
  "use_case": "chatbot",
  "tenant_id": "dev-tenant"
}
```

El RAG deberia recuperar chunks del documento recien subido.

---

### Flujo C — Expediente de proyecto y evaluacion de riesgo

Muestra el flujo estructurado de onboarding de un proyecto de IA.

**Paso 1 — Crear el proyecto (expediente)**

```
POST /projects
```

Body de ejemplo:
```json
{
  "tenant_id": "dev-tenant",
  "name": "Asistente de creditos IA",
  "profile": {
    "industry": "financial_services",
    "ai_type": "llm",
    "use_case": "credit_scoring",
    "country": "costa-rica",
    "framework": ["iso_42001", "nist_ai_rmf"],
    "maturity_level": "pilot"
  }
}
```

Respuesta: `project_id`

**Paso 2 — Iniciar entrevista de intake de riesgo (B2)**

```
POST /risk/{project_id}/intake
```

Body:
```json
{
  "tenant_id": "dev-tenant",
  "user_id": "demo-user",
  "message": "Hola, queremos evaluar los riesgos de nuestro sistema."
}
```

Llamar este endpoint varias veces con respuestas al asistente hasta que la respuesta incluya "perfil completo". El asistente hara preguntas sobre el sistema de IA para construir el perfil de riesgo.

**Paso 3 — Ejecutar analisis de riesgo (B3)**

```
POST /risk/{project_id}/analyze
```

Body:
```json
{
  "tenant_id": "dev-tenant"
}
```

La respuesta incluye un `risk_register` estructurado con riesgos identificados por framework (ISO 42001, NIST AI RMF, GDPR, EU AI Act).

---

### Flujo D — Generacion de politicas y controles

Muestra la capacidad de generar documentacion de cumplimiento automaticamente.

**Paso 1 — Generar politica para el proyecto**

Requiere haber completado el Flujo C (el proyecto necesita tener `risk_register`).

```
POST /policies/{project_id}/generate
```

Body:
```json
{
  "tenant_id": "dev-tenant"
}
```

Respuesta: documento de politica en JSON con secciones, objetivos y controles recomendados.

**Paso 2 — Derivar controles del registro de riesgo**

```
POST /policies/{project_id}/controls
```

Body:
```json
{
  "tenant_id": "dev-tenant"
}
```

Respuesta: lista de controles especificos derivados de los riesgos identificados.

---

### Flujo E — Memoria conversacional (threads)

Muestra que el asistente mantiene contexto a lo largo de una conversacion.

**Paso 1 — Crear un thread**

```
POST /threads
```

Body:
```json
{
  "tenant_id": "dev-tenant",
  "user_id": "demo-user",
  "title": "Consulta sobre GDPR"
}
```

Respuesta: `thread_id`

**Paso 2 — Primera pregunta en el thread**

```
POST /ai/query
```

Body:
```json
{
  "query": "Que es el GDPR y a quien aplica?",
  "use_case": "chatbot",
  "tenant_id": "dev-tenant",
  "thread_id": "<thread_id>"
}
```

**Paso 3 — Pregunta de seguimiento (mismo thread)**

```
POST /ai/query
```

Body:
```json
{
  "query": "Y en el caso especifico de Costa Rica, aplica igualmente?",
  "use_case": "chatbot",
  "tenant_id": "dev-tenant",
  "thread_id": "<thread_id>"
}
```

El asistente deberia recordar que se estaba hablando de GDPR sin necesidad de repetirlo.

**Paso 4 — Ver el historial completo**

```
GET /threads/{thread_id}/messages
```

---

### Flujo F — Trazabilidad y calidad

Muestra la capa de auditoria y retroalimentacion del sistema.

**Paso 1 — Ver la evaluacion automatica de una respuesta**

Tomar cualquier `trace_id` de una respuesta anterior de `/ai/query`.

```
GET /evals/{trace_id}
```

Respuesta incluye:
- `grounding_score`: que tan anclada esta la respuesta en los chunks recuperados (0-1)
- `length_ok`: si la respuesta tiene longitud razonable
- `has_citations`: si la respuesta incluye citaciones
- `passed`: veredicto final

**Paso 2 — Enviar feedback sobre una respuesta**

```
POST /feedback
```

Body:
```json
{
  "tenant_id": "dev-tenant",
  "user_id": "demo-user",
  "trace_id": "<trace_id>",
  "rating": 5,
  "comment": "Respuesta precisa y bien citada."
}
```

---

## 3. Limitaciones actuales del codigo

### Criticas — bloquean uso en produccion

**1. Autenticacion no implementada**
El header `Authorization` es aceptado pero no validado. Cualquier valor pasa. No hay verificacion de JWT ni politicas RLS en la base de datos. Cualquier persona con acceso a la URL puede leer o escribir datos de cualquier tenant.

**2. Sin streaming de respuestas**
El sistema espera a que el LLM termine de generar toda la respuesta antes de enviarla al cliente. En consultas complejas o documentos grandes, esto puede tardar entre 15 y 30 segundos con la pantalla en blanco. No hay indicador de progreso ni respuesta parcial.

---

### Medias — afectan calidad de los resultados

**3. Re-ranking heuristico solamente**
La seleccion de chunks relevantes usa unicamente similitud coseno con pgvector. No hay re-ranking con Cohere Rerank ni cross-encoder. En consultas ambiguas, los chunks seleccionados pueden no ser los mas relevantes semanticamente.

**4. Sin reintentos ni failover de proveedor LLM**
Si OpenAI devuelve un error (rate limit, timeout, outage), la llamada falla directamente. No hay logica de reintento con backoff ni cambio automatico a Gemini como proveedor de respaldo.

**5. Chunking por palabras fijas**
Los documentos se dividen en chunks de tamano fijo (300 tokens aprox.) sin respetar limites de oraciones ni estructura semantica. Un chunk puede cortar una oracion a la mitad, afectando la calidad del contexto recuperado.

**6. OCR no implementado**
Los PDFs escaneados (imagenes sin capa de texto) no se procesan correctamente. El sistema extrae texto vacio y el documento queda sin contenido util. Solo funciona con PDFs que tienen texto seleccionable.

---

### Menores — deuda tecnica o mejoras de UX

**7. Token counting aproximado**
El conteo de tokens para el prompt se hace por palabras, no con tiktoken (la libreria oficial de OpenAI). En prompts muy largos, el sistema podria superar el limite de contexto del modelo sin advertencia y recibir un error.

**8. Score de grounding heuristico**
El `grounding_score` mide solapamiento de palabras entre la respuesta y los chunks, no similitud semantica. Puede marcar como "no fundamentada" una respuesta correcta que usa sinonimos, y pasar una respuesta incorrecta que copie frases del contexto sin coherencia.

**9. Cobertura de tests limitada**
Hay solo 3 archivos de test. No existen tests de integracion contra Supabase real, ni tests unitarios para los servicios de riesgo, politicas, evaluacion o feedback. Los cambios al pipeline RAG no tienen cobertura automatizada.

**10. CORS hardcodeado**
Los origenes permitidos para CORS estan escritos directamente en `app/main.py`, no en el archivo de configuracion. Cambiar los origenes requiere modificar codigo y reiniciar el servidor.

**11. Almacenamiento de archivos en disco local**
Los documentos subidos se guardan en la carpeta `./uploads/` del servidor. En un entorno de contenedores o multi-instancia, los archivos no se comparten entre instancias. No esta integrado con S3 ni ningun almacenamiento externo.
