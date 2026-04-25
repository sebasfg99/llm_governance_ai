# Endpoint de Salud

Endpoint público (sin autenticación) usado por load balancers, Docker y
Kubernetes para verificar que el proceso está vivo.

---

## `GET /health`

```
GET /health
```

### Response 200

```json
{
  "status": "ok",
  "version": "0.1.0",
  "service": "ai-core"
}
```

### Notas

- **Es un check superficial**: solo confirma que el proceso responde, no que
  Supabase u OpenAI estén accesibles.
- Para un health "deep" (con dependencias) hace falta extender este endpoint —
  ver TODO en `app/api/routes/health.py`.

---

## Ejemplo

```bash
curl http://localhost:8000/health
```

```bash
# Como liveness probe en Kubernetes:
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10
```
