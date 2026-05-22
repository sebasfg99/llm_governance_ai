# 06 · Seguridad multi-tenant

> Invariante crítico:
>
> **Ningún cliente puede leer ni escribir datos de otro cliente, NUNCA, en ninguna ruta del código.**

## Modelo de datos

El sistema modela cada cliente con un `tenant_id` (texto). El spec usa el término `client_id`; conceptualmente son lo mismo. La migración 006 añadió a `traces` un alias `client_id` mantenido por trigger:

```sql
client_id := COALESCE(client_id, tenant_id)
```

Toda tabla con datos de cliente lleva `tenant_id`:

- `documents`, `document_chunks` — documentos cargados.
- `traces` — cada llamada al LLM (también `client_id`).
- `evaluations` — heredado de `traces.trace_id`.
- `response_risks` — `tenant_id` y `client_id` propios, `trace_id` como FK.
- `projects`, `threads`, `messages`, `feedback` — heredan vía referencias.

Tenant reservado: `'global'`. Solo se usa para la librería compartida de regulaciones (ISO, OWASP, NIST). Nadie puede operar como `client_id='global'`.

## Punto único de validación

`app/core/tenancy.py`:

```python
from app.core.tenancy import validate_tenant_access

validate_tenant_access(
    client_id=...,
    user_id=...,
    document_client_id=None,   # opcional
)
```

Reglas:

1. `client_id` debe ser no vacío → `AuthenticationError(401)` si falla.
2. `user_id` debe ser no vacío → `AuthenticationError(401)` si falla.
3. `client_id != "global"` (slug reservado) → `TenantIsolationError(403)`.
4. Si se pasa `document_client_id`, debe coincidir con `client_id` → `TenantIsolationError(403)` si no.

Cada incumplimiento se logea con nivel ERROR con la palabra `TENANT_ISOLATION_VIOLATION` para que el SIEM lo recoja.

## FastAPI dependency

Re-exportado desde `app/api/deps.py`:

```python
from fastapi import Depends
from app.api.deps import get_validated_tenant

@router.get("/widgets")
def list_widgets(identity: dict = Depends(get_validated_tenant)):
    client_id = identity["client_id"]
    user_id   = identity["user_id"]
    ...
```

`get_validated_tenant` lee:

- `Authorization: Bearer <token>` (en producción → JWT de Supabase).
- `X-Client-Id: <client>` (override service-to-service, opcional).

Y produce un dict `{client_id, tenant_id, user_id}` ya validado.

## Defensas en capas

| Capa | Defensa |
| ---- | ------- |
| API | `Depends(get_validated_tenant)` en routes con datos de cliente. |
| Servicio | Cada query SQL incluye `tenant_id = $tenant`. |
| Retrieval | `match_chunks` y `match_chunks_hybrid` filtran por `match_tenant_id` dentro del PL/pgSQL. |
| Documentos | Cada `documents.tenant_id` se compara con el caller antes de leer chunks. |
| Logs | Todo cross-tenant attempt → log ERROR con ambos IDs. |
| Tabla `response_risks` | Tiene `tenant_id` y `client_id` propios para auditoría futura aún si la trace se borra. |

## Threat model (mini)

| Amenaza | Mitigación |
| ------- | ---------- |
| Caller pasa `client_id` ajeno | `Depends(get_validated_tenant)` rechaza si el token no autoriza. |
| Documento pertenece a otro cliente | `validate_tenant_access(... document_client_id)` |
| Chunk pertenece a otro cliente | RPCs filtran SQL-side; el servicio valida que `chunk.tenant_id == caller.tenant_id`. |
| Manipulación del `X-Client-Id` | El header solo se acepta cuando el token también lo permite (en stub: misma identidad). |
| Trace de un cliente expuesto en `/evals` | `EvalService.get_evaluation` debe filtrar por `tenant_id` (TODO: agregar test E2E con dos tenants). |
| Métricas cruzadas en `/metrics/rag` | El endpoint loggea cuando `client_id` difiere del token; en producción debe rechazar. |

## Roadmap (post-MVP)

1. **Row Level Security** en Supabase: hoy el aislamiento es app-layer. Activar RLS con políticas `current_setting('app.tenant_id')` desde Postgres pega un cinturón extra.
2. **JWT real**: reemplazar `verify_token` con validación de Supabase JWT (TODO ya marcado en `app/core/security.py`).
3. **Auditoría inmutable**: enviar las violaciones tenant a una tabla append-only o a un sink externo.
4. **Tests E2E con dos tenants**: ya tenemos `test_tenant_isolation.py`; ampliar con tests de `/ai/query` cruzando IDs.

## Tests

`app/tests/test_tenant_isolation.py`:

- `validate_tenant_access` rechaza cross-tenant, slug reservado, IDs faltantes/blanco.
- `TraceService.list_traces` siempre filtra por tenant.
- `RetrievalService.retrieve` pasa `match_tenant_id` al RPC.

## Reglas para nuevos servicios

Antes de mergear un servicio nuevo, verificar:

- [ ] Las tablas que toca llevan `tenant_id`.
- [ ] Las queries incluyen `WHERE tenant_id = ...`.
- [ ] El endpoint usa `Depends(get_validated_tenant)`.
- [ ] Hay un test que cruza tenants y verifica el rechazo.
- [ ] Los logs incluyen `client_id` para correlación.
