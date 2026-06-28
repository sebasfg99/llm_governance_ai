# Costa Rica — Normativa de Ciberseguridad

Coloca aquí los PDFs oficiales de SUGEF/CONASSIF para que sean indexados
por `seed_documents.py`. Los siguientes documentos están pendientes:

| Archivo esperado | Fuente oficial | Industrias |
|-----------------|----------------|------------|
| `SUGEF 10-07.pdf` | https://www.sugef.fi.cr → Normativa → SUGEF 10-07 (Gestión de Tecnologías de Información) | bank, cooperative |
| `SUGEF 18-16.pdf` | https://www.sugef.fi.cr → Normativa → SUGEF 18-16 (Gestión de la Seguridad de la Información) | bank, cooperative |
| `SUGEF 24-22 (v2 1° de enero de 2023).pdf` | Ya presente en este directorio | bank, cooperative |

Una vez colocados, ejecuta desde `llm_governance_ai/`:

```bash
PYTHONPATH=. python scripts/seed_documents.py
```

Los documentos del directorio `Ciberseguridad/` se etiquetan con el framework
`cybersecurity`. Los que están en `Costa Rica/` reciben `country=costa-rica`.
Los documentos SUGEF/CONASSIF se etiquetan adicionalmente con
`industry_codes=['bank','cooperative']` (Layer 3 — filtrado por industria).
