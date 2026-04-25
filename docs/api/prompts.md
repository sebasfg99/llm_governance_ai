# Endpoints de Prompts

La biblioteca de prompts almacena los **system prompts versionados** que el
orquestador usa según el `use_case` solicitado. Cada `use_case` tiene un único
prompt activo (`is_active=true`) en cualquier momento; los anteriores se
preservan para auditoría y rollback.

---

## `GET /prompts`

Lista todos los prompts (todos los use cases, todas las versiones).

```bash
curl http://localhost:8000/prompts \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Response

```json
{
  "prompts": [
    {
      "id": "uuid",
      "name": "chatbot",
      "version": 3,
      "use_case": "chatbot",
      "content": "Eres un asistente especializado en gobernanza de IA...",
      "is_active": true,
      "created_at": "2026-04-25T12:00:00Z"
    },
    {
      "id": "uuid",
      "name": "chatbot",
      "version": 2,
      "use_case": "chatbot",
      "content": "...",
      "is_active": false,
      "created_at": "2026-04-20T10:00:00Z"
    }
  ],
  "total": 9
}
```

---

## `POST /prompts`

Crea una nueva versión de un prompt. Si `is_active=true`, el prompt activo
anterior para ese mismo `use_case` se desactiva automáticamente.

### Request

```jsonc
{
  "name": "chatbot",
  "use_case": "chatbot",
  "content": "Nuevo system prompt completo...",
  "is_active": true
}
```

```bash
curl -X POST http://localhost:8000/prompts \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "chatbot",
    "use_case": "chatbot",
    "content": "Eres un asistente experto en ISO 42001...",
    "is_active": true
  }'
```

### Response

```json
{
  "id": "uuid",
  "name": "chatbot",
  "version": 4,
  "use_case": "chatbot",
  "is_active": true,
  "message": "Prompt created successfully."
}
```

---

## Use cases disponibles

| `use_case` | Para qué |
|---|---|
| `chatbot` | Q&A general con citas |
| `executive_summary` | Resumen ejecutivo de un proyecto |
| `risk_intake` | Entrevista guiada para llenar `project_profile` |
| `risk_analysis` | Generación del `RiskRegister` estructurado |
| `policy_generation` | Borrador de política organizacional |
| `controls_generation` | Derivación de controles desde riesgos |

> Si necesitas un nuevo `use_case`, primero agrégalo al `Literal` de
> `app/schemas/ai.py` (campo `UseCaseLiteral`) y luego carga el prompt aquí.

---

## Sembrar prompts iniciales

El script `scripts/seed_prompts.py` carga los prompts base en orden. Es
**idempotente**: si ya existe la versión N para un use case, la salta.

```bash
PYTHONPATH=. python scripts/seed_prompts.py
```

---

## Tips de versionado

- **Nunca edites un prompt activo en producción sin crear una nueva versión.**
  Cada trace guarda el `prompt_text` completo, pero solo el ID del prompt es
  trazable hacia atrás.
- **Hardening footer:** independientemente del contenido, el sistema **siempre**
  añade reglas de seguridad inmutables al final del system prompt
  (`SYSTEM_PROMPT_HARDENING_FOOTER` en `app/domain/llm/guardrails.py`).
- **Rollback rápido:** si una nueva versión empeora la calidad, llama
  `POST /prompts` re-enviando el contenido de la versión anterior con
  `is_active=true`.

## Versión activa del `chatbot`

Hoy el prompt activo es **"Chatbot Documental v4 (semantic inference)"**
(version=4). Diferencia clave vs. v3: introduce un PASO 0 obligatorio de
**interpretación semántica con equivalencias normativas** ("comité" ≈
"estructura de gobernanza", "auditoría" ≈ "evaluación", etc.) antes de
clasificar la consulta. Razón histórica:
[ADR-012](../decisions/design_choices.md#adr-012--prompt-v4-del-chatbot-con-inferencia-semántica).

`scripts/seed_prompts.py` está actualizado para sembrar v4 en bases de
datos nuevas. v3 se mantiene como histórico (`is_active=false`).
