# Prompt maestro — Integrar `ai-core` con cualquier UI

Copiá este documento completo, reemplazá los valores entre `<...>` y entregalo
a un agente de programación con acceso a ambos repositorios.

---

## Datos de entrada

- **Repositorio de la UI:** `<RUTA_O_URL_DEL_REPOSITORIO_UI>`
- **Repositorio de ai-core:** `<RUTA_O_URL_DE_LLM_GOVERNANCE_AI>`
- **Entornos requeridos:** desarrollo local con Supabase CLI y producción con
  Supabase Cloud.
- **URL pública de ai-core en producción:** `<URL_O_PENDIENTE_DE_DEFINIR>`
- **Referencia del proyecto Supabase:** `<PROJECT_REF_O_PENDIENTE>`
- **Proveedor LLM preferido:** `<openai|gemini|preguntar>`
- **Proveedor de despliegue de ai-core:** `<Railway|Cloud Run|Docker|preguntar>`

No incluyas API keys, service-role keys ni otros secretos dentro de este
prompt. Deben configurarse mediante el gestor de secretos del entorno.

---

## Instrucción para el agente

Actuá como arquitecto y desarrollador full-stack senior. Tu objetivo es
configurar `llm_governance_ai` (`ai-core`) e integrar **todas sus capacidades**
en la UI indicada, adaptándote al framework, backend, autenticación, diseño y
convenciones que ya existan en el proyecto.

La integración debe quedar operativa en:

1. Desarrollo local con Supabase CLI.
2. Producción con Supabase Cloud.

Debés completar la implementación, las pruebas y la documentación. No inventes
endpoints, esquemas ni contratos: verificá siempre el código fuente, OpenAPI y
la documentación actual de `ai-core`.

## Protocolo obligatorio de interacción

1. Empezá en modo de inspección **solo lectura**. No modifiques archivos, no
   instales dependencias, no cambies configuración y no ejecutes comandos con
   efectos externos.
2. Revisá ambos repositorios y detectá qué información falta.
3. Hacé todas las preguntas necesarias **una por una**:
   - formulá una sola pregunta por mensaje;
   - esperá la respuesta antes de hacer la siguiente;
   - no repitas preguntas contestadas por los datos de entrada o por el código;
   - cuando sea útil, ofrecé opciones y marcá una recomendación;
   - explicá brevemente por qué la decisión afecta la implementación.
4. Como mínimo, confirmá cuando no pueda deducirse del código:
   - comandos para instalar, ejecutar, probar y compilar la UI;
   - sistema de autenticación y forma de obtener la identidad del usuario;
   - fuente autoritativa de `tenant_id` o institución;
   - ubicación y runtime del backend o Edge Function que actuará como proxy;
   - URLs y estrategia de despliegue por entorno;
   - proveedor LLM, billing y disponibilidad del corpus documental;
   - alcance visual de los cambios y flujos que deben exponerse inicialmente.
5. Cuando ya no falte información, presentá:
   - resumen de requisitos;
   - arquitectura propuesta;
   - plan por fases;
   - archivos que prevés crear o modificar;
   - dependencias nuevas;
   - migraciones, comandos y servicios externos involucrados;
   - riesgos, supuestos y estrategia de reversión;
   - pruebas y criterios de aceptación.
6. Después preguntá, en un mensaje separado:
   **“¿Autorizás que realice estos cambios?”**
7. No hagas ningún cambio hasta recibir autorización explícita.
8. Tras la autorización, implementá el plan de forma autónoma. Si aparece una
   decisión nueva que cambie el alcance, implique una operación destructiva o
   requiera permisos adicionales, detenete y preguntá **una sola cosa a la
   vez**.
9. Nunca hagas commits, push, despliegues, migraciones remotas ni rotaciones de
   secretos sin permiso explícito específico.

## Arquitectura no negociable

Usá siempre esta separación:

```text
UI → backend seguro / Edge Function → ai-core → Supabase + proveedor LLM
```

- La UI **nunca** debe conocer `AI_CORE_SERVICE_KEY`,
  `SUPABASE_SERVICE_KEY`, claves de OpenAI/Gemini ni secretos equivalentes.
- Conservá el proveedor de autenticación existente de la UI. Creá un adaptador
  hacia `ai-core`; no migres la autenticación sin aprobación.
- El backend proxy debe validar la sesión y permisos del usuario antes de
  llamar a `ai-core`.
- Obtené `tenant_id` y `user_id` de fuentes verificadas server-side. Nunca
  confíes en valores enviados por el navegador, `localStorage` o
  `sessionStorage` para autorización.
- El proxy llama a `ai-core` mediante Path A:

```text
Authorization: Bearer <AI_CORE_SERVICE_KEY>
X-Tenant-Id: <tenant verificado server-side>
X-User-Id: <usuario verificado server-side>
```

- El aislamiento multi-tenant es obligatorio para documentos, proyectos,
  threads, mensajes, trazas, riesgos, políticas, controles y feedback.
- No crees un bypass de desarrollo que use `Bearer dev-token`.
- No ejecutes queries de Supabase directamente desde componentes visuales.
  Encapsulá acceso remoto en la capa de servicios existente de la aplicación.
- Para consultas RAG, usá `POST /ai/query`, fachada del orquestador
  `AIService`. No llames directamente a embeddings, tablas de chunks, prompts
  internos ni proveedores LLM.

## Inspección inicial obligatoria

### En la UI

Identificá y respetá:

- framework, lenguaje, package manager y versiones;
- estructura de rutas, layouts y componentes;
- librería visual, tokens, temas y patrones de accesibilidad;
- capa de servicios/API y manejo de errores;
- estado local/global y cache de datos;
- autenticación, autorización y roles;
- modelo de institución/organización/tenant;
- backend, BFF, server actions o Edge Functions disponibles;
- estrategia de variables de entorno y secretos;
- pruebas, lint, formatter, tipos y pipeline CI;
- funcionalidades de IA ya existentes para evitar duplicaciones.

### En `ai-core`

Verificá como mínimo:

- `README.md`;
- `docs/SETUP_RAG_SIDECAR.md`;
- `docs/api/README.md`;
- `docs/api/ai.md`;
- `docs/api/integration_guide.md`;
- `/docs` u `/openapi.json` del servicio en ejecución;
- `app/main.py`, rutas, schemas y autenticación;
- migraciones disponibles y scripts de seeds;
- `.env.example`, Dockerfile y archivos compose relevantes.

Si la documentación contradice el código, tratá el código y OpenAPI como fuente
de verdad, documentá la diferencia y corregí la documentación dentro del
alcance autorizado.

## Configuración de `ai-core`

### Desarrollo local

- Usá Supabase CLI para disponer de Postgres, pgvector, REST/RPC, Auth y
  Storage.
- No confundas el contenedor de Postgres crudo del `docker-compose.yml` con un
  stack Supabase completo.
- Configurá `.env` sin commitearlo.
- Verificá que las migraciones `001` a `008` se apliquen en orden sobre la
  misma base indicada por `SUPABASE_URL`.
- Ejecutá los seeds de prompts y documentos.
- Confirmá que el corpus exista antes de prometer respuestas con citas.
- Levantá `ai-core` con `uvicorn` o Docker, pero no ambos sobre el puerto 8000.
- Validá `GET /health`, `GET /ready` y una consulta autenticada real.

### Producción

- Usá Supabase Cloud y un despliegue persistente de `ai-core`.
- Configurá secretos exclusivamente en el gestor del proveedor.
- Restringí CORS a orígenes autorizados cuando corresponda.
- Configurá timeouts, límites de payload, rate limiting, logs estructurados,
  health checks y observabilidad sin registrar secretos ni contenido sensible.
- Asegurá que el proxy y `ai-core` compartan el mismo
  `AI_CORE_SERVICE_KEY`.
- Documentá despliegue, rollback y rotación de claves, pero no ejecutes cambios
  remotos sin permiso.

## Capacidades que debe integrar

Verificá los contratos reales y construí una capa adaptadora tipada para todas
las capacidades disponibles:

1. **Chat RAG**
   - respuesta normal y streaming SSE;
   - cancelación de requests;
   - estados de carga y reconexión;
   - Markdown sanitizado;
   - citas y fuentes;
   - metadatos como `trace_id`, `thread_id`, modelo y latencia cuando proceda.
2. **Memoria conversacional**
   - crear/reutilizar threads;
   - recuperar historial;
   - asociar conversaciones a proyectos;
   - no usar almacenamiento del navegador como fuente de autorización.
3. **Proyectos o expedientes**
   - listar, crear, consultar y actualizar según permisos.
4. **Documentos**
   - upload, progreso, procesamiento y estado;
   - validación de tipo/tamaño;
   - acceso restringido por tenant;
   - biblioteca global solo donde el contrato la permita.
5. **Riesgos**
   - intake y análisis;
   - presentación clara de resultados estructurados.
6. **Políticas y controles**
   - generación, revisión y controles ligados al proyecto.
7. **Feedback, trazas y evaluaciones**
   - permitir feedback sobre una respuesta;
   - mostrar trazabilidad y evaluación solo a roles autorizados.
8. **Prompts y administración**
   - exponer gestión únicamente si el API y el rol lo permiten;
   - nunca permitir que un usuario común modifique prompts globales.

No asumas que todos los endpoints listados aquí conservan el mismo nombre.
Inspeccioná las rutas reales y generá tipos a partir de OpenAPI cuando sea
compatible con el stack.

## Backend proxy

Creá o extendé la capa server-side existente con operaciones equivalentes a
las capacidades anteriores.

El proxy debe:

- validar autenticación, rol, tenant y propiedad del recurso;
- construir internamente los headers hacia `ai-core`;
- impedir que el cliente sobrescriba `X-Tenant-Id` o `X-User-Id`;
- filtrar campos permitidos del request y de la respuesta;
- soportar streaming sin acumular toda la respuesta en memoria;
- propagar cancelación y códigos HTTP útiles;
- normalizar errores de validación, auth, límites, proveedor y disponibilidad;
- implementar límites de tamaño y timeout apropiados;
- evitar logs con tokens, prompts sensibles o documentos completos;
- aplicar protección CSRF cuando use cookies;
- preservar auditoría y trazabilidad.

## Adaptación de la UI

Podés refactorizar y mejorar la UX, pero antes debés preguntar una por una las
decisiones visuales que no puedan inferirse y obtener permiso para los cambios.

La solución debe:

- reutilizar el design system y patrones del proyecto cuando existan;
- mantener componentes pequeños y servicios desacoplados del framework visual;
- ofrecer estados vacío, carga, streaming, éxito, cancelación y error;
- ser accesible por teclado y lector de pantalla;
- adaptarse a las plataformas que el proyecto ya soporte;
- evitar HTML no sanitizado;
- mostrar claramente cuándo una respuesta no tiene fuentes;
- diferenciar errores técnicos de falta de evidencia RAG;
- evitar exponer información de otros tenants en caches o estados compartidos.

No reemplaces toda la UI ni agregues una librería visual nueva sin justificarlo
y pedir autorización.

## Pruebas obligatorias

Antes de declarar la tarea completa, ejecutá y reportá:

1. formatter y lint;
2. chequeo de tipos;
3. tests unitarios;
4. tests de componentes;
5. tests de integración del proxy;
6. smoke test de `health` y `ready`;
7. consulta RAG no-streaming autenticada;
8. consulta RAG streaming con recepción progresiva;
9. respuesta con citas usando corpus real;
10. persistencia y recuperación de thread;
11. upload y procesamiento de un documento permitido;
12. al menos un flujo de riesgos;
13. al menos un flujo de política/controles;
14. feedback y consulta de evaluación con rol autorizado;
15. pruebas negativas de `401`, `403`, tenant cruzado, payload inválido,
    timeout y proveedor no disponible;
16. build de producción;
17. checklist manual de los flujos principales en la UI.

No ocultes fallos ni afirmes que una prueba pasó si no pudo ejecutarse. Separá
claramente resultados comprobados, pendientes y bloqueos.

## Documentación que debe quedar

Actualizá o creá documentación que incluya:

- arquitectura y diagrama del flujo UI → proxy → ai-core;
- variables requeridas por entorno, sin valores secretos;
- setup local reproducible desde cero;
- despliegue de producción;
- comandos para ejecutar y probar;
- matriz de roles y permisos;
- contratos del proxy;
- troubleshooting;
- checklist de aceptación;
- decisiones y limitaciones conocidas.

## Criterios de aceptación

La integración solo está terminada cuando:

- ninguna clave privilegiada llega al navegador o al repositorio;
- identidad y tenant se derivan y validan server-side;
- dos tenants de prueba no pueden acceder a recursos cruzados;
- la UI puede usar todas las capacidades autorizadas de `ai-core`;
- el chat transmite progresivamente, permite cancelar y muestra citas;
- memoria, documentos, riesgos, políticas, controles y feedback funcionan;
- local y producción tienen configuraciones documentadas y separadas;
- las pruebas obligatorias pasan o los bloqueos quedan demostrados;
- la documentación permite repetir el setup sin conocimiento previo.

## Formato del informe final

Al terminar, respondé en español con:

1. resultado general;
2. arquitectura implementada;
3. archivos modificados;
4. variables y secretos que el usuario debe configurar;
5. migraciones o acciones manuales pendientes;
6. pruebas ejecutadas y resultados;
7. riesgos o limitaciones;
8. pasos exactos para correr localmente;
9. pasos exactos para desplegar, sin ejecutar el despliegue salvo autorización.

Empezá ahora únicamente con la inspección de solo lectura. Después hacé la
primera pregunta necesaria y esperá mi respuesta.
