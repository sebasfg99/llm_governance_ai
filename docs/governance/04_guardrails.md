# 04 · Guardrails de respuesta

Los **input guardrails** ya existían (`scan_for_injection`, hardening del system prompt). Este documento describe los **response guardrails** que se ejecutan después del LLM y antes de devolver la respuesta al cliente.

Punto de entrada:

```python
from app.domain.llm.guardrails import validate_response_guardrails

result = validate_response_guardrails(
    query=...,
    response=...,
    chunks=[...],
    client_id="acme",
    user_id="alice",
    evaluation=...,    # opcional, salida del evaluator
    use_case="chatbot",
    strict_context=None,
)

# result keys:
# - allowed (bool)
# - violations (list[dict])
# - safe_response (str)
# - requires_disclaimer (bool)
# - pii_findings (list[dict])
```

## Reglas

| Regla | Severidad | Acción | Bloquea |
| ----- | --------- | ------ | ------- |
| `client_id` ausente o vacío | high | Rechaza la entrega | ✅ |
| PII detectado (email, teléfono, RFC, CURP, cédula CR, IBAN, tarjeta) | high | Reemplaza por `[REDACTED:<TIPO>]` | ❌ (entrega versión enmascarada) |
| `evaluation.hallucination=True` | high | Reescribe con mensaje de seguridad | ✅ |
| `evaluation.should_have_said_idk=True` | medium | Reescribe como abstención controlada | ❌ |
| Caso legal con `context_adherence < 0.7` (modo estricto) | high | Bloquea la entrega | ✅ |
| Caso legal/compliance | medium | Anexa disclaimer automático | ❌ |

## Detección de PII

`app/domain/llm/pii_detector.py`. Patrones cubiertos:

- **EMAIL** — RFC 5322 simplificado.
- **PHONE** — Internacional, CR (8 dígitos), MX/US (10 dígitos), formatos con guiones y paréntesis.
- **CR_CEDULA** — `1-1234-5678` y formato compacto `101234567`.
- **CURP** — México (18 caracteres).
- **RFC** — México (12-13 caracteres).
- **IBAN** — Cualquier país.
- **CREDIT_CARD** — 13-19 dígitos validados con Luhn (evita falsos positivos en IDs largos).

Cuando se detecta, la respuesta no se bloquea: se entrega con la PII reemplazada por `[REDACTED:EMAIL]`, `[REDACTED:PHONE]`, etc. Los hallazgos van al campo `pii_findings` del resultado y al risk scoring (siempre fuerzan HIGH).

## Disclaimers automáticos

Casos de uso que requieren disclaimer:

```
{policy_generation, controls_generation, risk_analysis, risk_intake, compliance, legal}
```

El texto añadido (con sentinel para idempotencia):

> ⚠️ Disclaimer legal/compliance: Este contenido es una orientación basada en los documentos recuperados. NO constituye asesoría legal vinculante. Valide con su equipo legal/compliance antes de aplicar.

Si la guardrail corre dos veces sobre la misma respuesta, el sentinel `⚠️` evita duplicar el bloque.

## Adherencia estricta al contexto

Cuando `use_case` es legal/compliance (lista anterior), se exige:

```
context_adherence_score >= 0.7  (configurable: governance_strict_context_threshold)
```

Si está por debajo, la respuesta se bloquea con código `low_context_adherence`. La motivación: en consultoría regulatoria una respuesta parcialmente correcta puede ser más peligrosa que una abstención.

Cuando se llama desde un chatbot conversacional con `strict_context=False`, este check no aplica.

## Reescritura segura

Cuando una regla bloqueante se dispara, `safe_response` se reemplaza por uno de los textos canónicos:

- **Hallucination** → "No puedo entregar esta respuesta porque contiene afirmaciones que no se pueden verificar contra los documentos disponibles. Reformula la consulta o consulta a un experto humano."
- **should_have_said_idk** → "No tengo información suficiente en los documentos disponibles para responder con certeza. Te recomiendo aportar más contexto o consultar la fuente original."
- **missing_client_id** → cadena vacía + 4xx en el endpoint.

## Tests

`app/tests/test_response_guardrails.py` cubre:

- `client_id` ausente/blanco bloquea.
- Email y teléfono se enmascaran.
- PII sola no bloquea, solo enmascara.
- Hallucination dispara reescritura.
- Adherencia baja bloquea solo en casos legales.
- Disclaimer no se duplica al ejecutar dos veces.
- IDK guard reescribe como abstención.

## Cómo extender

Agregar una nueva regla:

1. Añadir el patrón/condición en `validate_response_guardrails`.
2. Insertar `violations.append({code, severity, message, ...})`.
3. Si bloquea, agregar el `code` a `blocking_codes`.
4. Documentar el código en este archivo y en `03_risk_scoring.md`.
5. Agregar caso en `test_response_guardrails.py`.
