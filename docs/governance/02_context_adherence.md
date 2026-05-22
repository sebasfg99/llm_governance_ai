# 02 · Adherencia al contexto y detección de alucinaciones

> Métrica más importante para gobernanza enterprise.
>
> Responde a: **"¿Qué porción de la respuesta está realmente soportada por los documentos recuperados?"**

## Definición formal

```
context_adherence_score = #claims_supported / #claims_total
```

donde:

- **claim** = afirmación factual atómica e independientemente verificable extraída de la respuesta.
- **supported** = al menos un chunk recuperado contiene la información que sustenta el claim.
- **claims_total** = número total de claims encontrados en la respuesta.

Cuando la respuesta no contiene claims (saludos, IDK), el score se fija en `1.0` si la respuesta tiene texto, `0.0` si está vacía.

## Por qué un LLM judge y no Jaccard

| Enfoque | Limitación | Ejemplo |
| ------- | ---------- | ------- |
| Token overlap (Jaccard) | Falla con paráfrasis o sinónimos | "El comité revisa riesgos" ≠ "La junta evalúa peligros" |
| BLEU / ROUGE | Mide similitud léxica, no semántica | Falsos positivos con textos largos genéricos |
| Embeddings | Confunde "similar" con "verdadero" | Frases falsas pero temáticas pasan |
| **LLM-as-judge** | Costo de 1 llamada extra | Detecta paráfrasis correctas y afirmaciones inventadas |

Para gobernanza y auditoría enterprise, la precisión semántica del judge justifica los ~$0.0002 y ~300 ms adicionales por respuesta.

## Cómo se extraen los claims

`app/domain/evals/llm_judge.py` envía un prompt al modelo evaluador (default `gpt-4o-mini`) que pide:

> "Descompón la respuesta en afirmaciones factuales atómicas e independientemente verificables. Excluye saludos, hedges ('creo que…') y meta-comentarios."

Para cada claim el judge marca:

- `text`: la afirmación tal como aparece (o normalizada).
- `supported`: TRUE/FALSE basado SOLO en los chunks (no en conocimiento general).
- `supporting_chunk_indexes`: lista de índices de chunks que sustentan el claim.
- `is_hallucination`: TRUE cuando el claim se afirma como hecho pero no está soportado.

## Ejemplos

### Ejemplo 1 — adherencia perfecta (1.0)

**Query**: "¿Qué dice ISO 42001 sobre riesgos de equidad?"

**Chunks** (simplificados):

```
[1] ISO 42001 Articulo 9 establece que el sistema de gestión de IA debe
    identificar riesgos de equidad y discriminación.
[2] El comité de gobernanza revisa periódicamente esos riesgos.
```

**Respuesta**: "Según ISO 42001, el artículo 9 obliga a identificar riesgos de equidad y discriminación, y el comité de gobernanza los revisa periódicamente."

**Claims detectados**:

| Claim | Soportado | Hallucination |
| ----- | --------- | ------------- |
| ISO 42001 artículo 9 obliga a identificar riesgos de equidad y discriminación | ✅ chunk 1 | ❌ |
| El comité de gobernanza revisa periódicamente esos riesgos | ✅ chunk 2 | ❌ |

`context_adherence_score = 2 / 2 = 1.0`

### Ejemplo 2 — alucinación parcial (0.5)

**Respuesta**: "ISO 42001 obliga a identificar riesgos de equidad y fue ratificada por la ONU en 1953."

**Claims detectados**:

| Claim | Soportado | Hallucination |
| ----- | --------- | ------------- |
| ISO 42001 obliga a identificar riesgos de equidad | ✅ chunk 1 | ❌ |
| ISO 42001 fue ratificada por la ONU en 1953 | ❌ | ✅ |

`context_adherence_score = 1 / 2 = 0.5`, `hallucination = True`.

### Ejemplo 3 — abstención correcta

**Query**: "¿Cuál es el horario del banco?"

**Chunks** (todos sobre ISO 42001).

**Respuesta**: "No tengo información suficiente en los documentos para responder."

**Claims detectados**: ninguno (la respuesta es una abstención, no afirma nada factual).

`context_adherence_score = 1.0`, `should_have_said_idk = False`.

### Ejemplo 4 — debería haberse abstenido

**Misma situación que (3)**, pero la respuesta dice "El banco abre de 9 a 5pm".

**Claims**: 1 (no soportado, hallucination).

`should_have_said_idk = True`, `hallucination = True`, `context_adherence = 0.0`.

## Umbrales operativos

| Uso | Umbral mínimo | Acción cuando se incumple |
| --- | ------------- | ------------------------- |
| Chatbot general | 0.7 | Marca risk MEDIUM, sigue sirviendo |
| Legal / compliance | 0.7 (estricto) | Guardrail bloquea la entrega |
| Fine-tune export | 0.9 | Trace excluido del dataset |

Configurable en `app/core/config.py` (`governance_strict_context_threshold`).

## Persistencia

Columna: `evaluations.context_adherence_score numeric(5,4)`.
Histórico de claims en `evaluations.judge_raw_json` (JSONB) para auditoría legal.
