# Endpoints de Políticas y Controles

Tres endpoints, todos delegados a `PoliciesService`:

| Endpoint | Para qué |
|---|---|
| `POST /policies/generate` | Generar una política sin proyecto (modo "discover") |
| `POST /policies/{project_id}/generate` | Generar la política para un proyecto existente |
| `POST /policies/{project_id}/controls` | Derivar controles desde el `risk_register` del proyecto |

> Los flujos de C1 (política) y C2 (controles) son **largos**: el LLM puede
> tardar 20–40 s en producir un documento extenso. La UI debería mostrar
> spinner y, opcionalmente, hacer polling de `GET /projects/{id}` para
> detectar la actualización.

---

## `POST /policies/generate` — Modo discover (sin proyecto)

Útil para previsualizar una política sin haber creado un proyecto. No persiste
nada al expediente.

### Request

```jsonc
{
  "country": "costa-rica",         // opcional, restringe RAG por país
  "include_project_context": true   // ignorado en este endpoint
}
```

### Response

```jsonc
{
  "project_id": null,
  "trace_id": "uuid",
  "policy": {
    "title": "Política de Uso Responsable de IA",
    "version": "1.0",
    "review_cycle_months": 12,
    "iso_42001_coverage": ["6.1", "6.2", "8.1"],
    "nist_functions_covered": ["GOVERN", "MAP"],
    "sections": [
      {
        "id": "S-1",
        "title": "Alcance",
        "content_markdown": "Esta política aplica a..."
      }
    ],
    "citations": [
      { "chunk_id": "...", "document_name": "ISO 42001:2023", "section": "6.2" }
    ]
  }
}
```

### Ejemplo

```bash
curl -X POST http://localhost:8000/policies/generate \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{ "country": "costa-rica" }'
```

---

## `POST /policies/{project_id}/generate`

Igual que el anterior, pero ligado a un proyecto. Si `include_project_context`
es `true` (default) el orquestador inyecta un resumen del expediente para que
la política mencione los riesgos específicos identificados.

El resultado se persiste en `projects.policy` y `status` pasa a
`policy_generated`.

### Request

```jsonc
{
  "country": "costa-rica",
  "include_project_context": true
}
```

### Response

Igual al modo discover, con `project_id` poblado.

### Ejemplo (Python)

```python
import requests

resp = requests.post(
    "http://localhost:8000/policies/3f8a-.../generate",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={"country": "costa-rica"},
    timeout=120,  # ¡importante!
).json()

policy = resp["policy"]
print(policy["title"])
for section in policy["sections"]:
    print(f"\n## {section['title']}\n{section['content_markdown']}")
```

### Errores

| HTTP | Causa |
|---|---|
| 404 | Proyecto no existe. |
| 422 | LLM devolvió JSON inválido. |
| 502 | Timeout del LLM (subir el `timeout` del cliente). |

---

## `POST /policies/{project_id}/controls`

Genera controles concretos a partir del `risk_register` del proyecto. Cada
control queda ligado a uno o varios riesgos del registro.

### Request

```jsonc
{
  "risk_register_override": null  // opcional, pasar un registro distinto sin persistirlo
}
```

### Response

```jsonc
{
  "project_id": "uuid",
  "trace_id": "uuid",
  "controls": {
    "controls": [
      {
        "id": "C-001",
        "title": "Auditoría de equidad por subgrupo",
        "description": "Ejecutar evaluación de paridad demográfica antes de cada release...",
        "addresses_risks": ["R-001", "R-004"],
        "iso_42001_clause": "8.4",
        "nist_function": "MEASURE",
        "implementation_guidance": "Usar fairlearn o equivalente...",
        "owner_role": "Data Science Lead",
        "evidence_required": ["Reporte de paridad firmado", "Tickets en backlog"],
        "priority": "alta"        // alta | media | baja
      }
    ],
    "coverage_summary": {
      "risks_addressed": ["R-001", "R-002", "R-004"],
      "risks_not_addressed": ["R-003"],
      "rationale_gap": "R-003 requiere control físico no aplicable a este sistema"
    }
  }
}
```

El resultado se persiste en `projects.controls`.

### Errores

| HTTP | Causa |
|---|---|
| 404 | Proyecto no existe o no tiene `risk_register`. |
| 422 | JSON inválido del LLM. |

---

## Workflow recomendado

```mermaid
flowchart LR
    A[POST /projects] --> B[POST /risk/{id}/intake]
    B --> C[POST /risk/{id}/analyze]
    C --> D[POST /policies/{id}/generate]
    C --> E[POST /policies/{id}/controls]
    D --> F[GET /projects/{id} → expediente completo]
    E --> F
```

---

## Tips de UX

- Como las llamadas son largas, **deshabilitar el botón** mientras está en
  curso y mostrar feedback ("Analizando riesgos…", "Generando política…").
- Si la respuesta tarda > 30 s, considerar un patrón **fire-and-poll**: el
  cliente llama al endpoint, recibe `trace_id`, y hace polling de
  `GET /projects/{id}` esperando ver `policy` poblado.
- Mostrar siempre las **citas** (`citations`) en la UI: dan trazabilidad y
  reducen percepción de hallucination.
