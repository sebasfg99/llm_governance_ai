# Metodología de evaluación RAGAs

Evaluación estandarizada del pipeline RAG de GovAI con
[RAGAs](https://docs.ragas.io), complementaria a las 8 métricas heurísticas
propias (`app/domain/evals/`). Permite comparar objetivamente el "antes y
después" de cambios en prompts, retrieval o modelos.

## 1. Golden dataset

`evals/golden_dataset_cr.jsonl` — 25 casos para el contexto demo
(Costa Rica / banca):

| Grupo | Casos | Fuente |
|---|---|---|
| ISO 42001 | 8 | `ISO42001_Replica.pdf` (global) |
| Normativa CR (CONASSIF 5-24, SUGEF 24-22, CONASSIF 4-16, Ley 8968, Reglamento 45061, NCS) | 8 | documentos `country=costa-rica` |
| Cross-framework (EU AI Act, NIST AI RMF, OWASP, GDPR, PCI DSS, MITRE/SAFE AI) | 6 | documentos globales |
| Fuera de corpus (IDK) | 3 | — el agente debe declinar |

Campos por línea: `id`, `question`, `ground_truth` (respuesta canónica
verificada contra los chunks reales ingestados), `source_doc`,
`expected_frameworks`, `country`, `industry`, `case_type`
(`answerable` | `idk`).

**Regla de mantenimiento:** las preguntas deben ser respondibles con los
documentos *ingestados* (verificar contra la tabla `documents`), no con el
conocimiento general del modelo. Al agregar corpus nuevo, agregar casos.

## 2. Harness

`scripts/ragas_eval.py`:

1. Lee el dataset y llama `POST /ai/query` del sidecar local por caso
   (`use_case=chatbot`, `stream=false`, `top_k=5`, `country`/`industry` del
   caso) — el mismo camino que usa el edge function en producción.
2. Captura `response_text` (answer) y los `retrieved_chunks`; el contenido
   completo de cada chunk se lee de `document_chunks` vía Supabase REST
   (el API devuelve solo un preview truncado).
3. Métricas RAGAs sobre los casos `answerable`:
   - **faithfulness** — ¿cada afirmación de la respuesta se sostiene en los contextos recuperados? (anti-alucinación)
   - **answer_relevancy** — ¿la respuesta atiende directamente la pregunta?
   - **context_precision** — ¿los chunks recuperados son relevantes y están bien ordenados?
   - **context_recall** — ¿los contextos cubren las afirmaciones del ground truth? (calidad de retrieval)
4. Los casos `idk` se excluyen del agregado RAGAs (declinar correctamente
   puntúa mal en answer_relevancy) y se miden aparte como **idk_accuracy**:
   % de casos donde la respuesta declina explícitamente (regex de frases
   tipo "no dispongo de información").
5. Judge: `gpt-4o-mini` (temperature 0) vía `OPENAI_API_KEY` del `.env`.

## 3. Ejecución reproducible

```bash
cd llm_governance_ai
.venv/bin/pip install -r requirements-eval.txt   # una vez
# sidecar corriendo en :8000 con el corpus ingestado
.venv/bin/python scripts/ragas_eval.py --label baseline
# … aplicar cambios …
.venv/bin/python scripts/ragas_eval.py --label post
```

Salidas en `evals/results/ragas_<label>_<fecha>.json`: agregado, score por
caso, respuestas completas, latencias y costos estimados. La comparación
baseline↔post se documenta en `docs/AI_OPTIMIZATION_REPORT.md` (repo padre).

## 4. Interpretación y umbrales orientativos

| Métrica | Preocupante | Aceptable | Bueno |
|---|---|---|---|
| faithfulness | < 0.7 | 0.7–0.85 | > 0.85 |
| answer_relevancy | < 0.7 | 0.7–0.85 | > 0.85 |
| context_precision | < 0.5 | 0.5–0.75 | > 0.75 |
| context_recall | < 0.5 | 0.5–0.75 | > 0.75 |
| idk_accuracy | < 2/3 | 2/3 | 3/3 |

Notas:
- `context_recall` bajo con `faithfulness` alto ⇒ problema de *retrieval*
  (los documentos correctos no llegan), no de generación.
- `faithfulness` bajo con `context_recall` alto ⇒ problema de *generación*
  (el modelo ignora o contradice el contexto) — revisar prompt.
- El judge introduce varianza (±0.03 típico); comparar tendencias, no
  centésimas.

## 5. Costos

Un run completo (25 casos) consume ~25 llamadas al sidecar (Gemini 2.5
Flash, ~$0.002/caso) + las llamadas del judge (gpt-4o-mini, ~$0.01-0.03 el
run). Total estimado < $0.10 por evaluación.
