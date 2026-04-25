# Endpoints de Documentos

Estos endpoints permiten subir archivos, dispararlos a través del pipeline de
ingesta (parse → chunk → embed) y consultarlos.

> **Multi-tenancy:** todos los documentos se asocian al `tenant_id` del usuario
> autenticado. La biblioteca normativa global se siembra con
> `tenant_id="global"` desde `scripts/seed_documents.py`.

---

## `POST /documents/upload`

Sube un archivo y crea su registro en la tabla `documents` con
`status="pending"`. **No** ejecuta el procesamiento (eso es un paso separado).

### Request

```
POST /documents/upload
Content-Type: multipart/form-data
Authorization: Bearer <token>
```

| Campo (form-data) | Tipo | Requerido | Descripción |
|---|---|---|---|
| `file` | file | sí | Archivo PDF, DOCX o TXT (≤ 50 MB). |
| `country` | string | no | Slug del país al que aplica el documento (ej. `costa-rica`, `panama`). Si se omite, el documento es global. |

### Response 200

```json
{
  "id": "8f2c1234-...",
  "tenant_id": "dev-tenant",
  "filename": "politica_ia.pdf",
  "file_type": "pdf",
  "status": "pending",
  "country": "costa-rica",
  "created_at": "2026-04-25T15:30:00Z",
  "message": "Document uploaded. Call /documents/{id}/process to start processing."
}
```

### Errores

| HTTP | Causa |
|---|---|
| 400 | Extensión no soportada (solo `.pdf`, `.docx`, `.txt`). |
| 413 | Archivo > 50 MB. |
| 422 | `file` faltante en el form-data. |

### Ejemplos

**curl**

```bash
curl -X POST http://localhost:8000/documents/upload \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "file=@./politica_ia.pdf" \
  -F "country=costa-rica"
```

**Python (requests)**

```python
import requests

with open("./politica_ia.pdf", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/documents/upload",
        headers={"Authorization": "Bearer YOUR_TOKEN"},
        files={"file": ("politica_ia.pdf", f, "application/pdf")},
        data={"country": "costa-rica"},
    )
print(resp.json())
```

**JavaScript (browser)**

```javascript
const form = new FormData();
form.append("file", fileInput.files[0]);
form.append("country", "costa-rica");

const resp = await fetch("/documents/upload", {
  method: "POST",
  headers: { Authorization: "Bearer YOUR_TOKEN" },
  body: form,
});
const data = await resp.json();
console.log("doc_id:", data.id);
```

---

## `POST /documents/{document_id}/process`

Dispara el pipeline completo de ingesta. **Bloqueante**: el request retorna
solo cuando termina (o falla) el parsing + embedding.

### Pipeline

1. Cambia status a `processing`.
2. Parsea el archivo (PDF / DOCX / TXT) → texto plano.
3. Lo trocea en chunks (sliding window, ~300 palabras con overlap de 30).
4. Llama a OpenAI Embeddings → vector de 1536 dimensiones por chunk.
5. Borra chunks previos del mismo documento (idempotencia).
6. Inserta los chunks en `document_chunks`.
7. Cambia status a `ready` (o `failed` con `error_message`).

### Request

```
POST /documents/{document_id}/process
Authorization: Bearer <token>
```

(no body)

### Response 200

```json
{
  "id": "8f2c1234-...",
  "status": "ready",
  "chunk_count": 47,
  "page_count": 12,
  "message": "Document processed successfully: 47 chunks created."
}
```

### Errores

| HTTP | Causa |
|---|---|
| 404 | Documento no existe. |
| 502 | Falla del API de embeddings de OpenAI. |
| 500 | Error de parsing (PDF corrupto, DOCX inválido). |

> Si el procesamiento falla, el documento queda con `status="failed"` y
> `error_message` lleno. Puedes reintentar llamando al mismo endpoint.

### Ejemplo

```bash
curl -X POST http://localhost:8000/documents/8f2c1234-.../process \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## `GET /documents`

Lista todos los documentos del tenant del usuario, ordenados por
`created_at DESC`.

### Request

```
GET /documents
Authorization: Bearer <token>
```

### Response 200

```json
{
  "documents": [
    {
      "id": "8f2c1234-...",
      "tenant_id": "dev-tenant",
      "filename": "politica_ia.pdf",
      "file_type": "pdf",
      "status": "ready",
      "country": "costa-rica",
      "page_count": 12,
      "chunk_count": 47,
      "created_at": "2026-04-25T15:30:00Z",
      "processed_at": "2026-04-25T15:31:30Z",
      "error_message": null
    }
  ],
  "total": 1
}
```

### Ejemplo

```bash
curl http://localhost:8000/documents \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## `GET /documents/{document_id}`

Devuelve el detalle de un documento. Incluye `error_message` si quedó en
`failed`.

### Errores

| HTTP | Causa |
|---|---|
| 404 | El documento no existe. |
| 403 | El documento pertenece a otro tenant. |

### Ejemplo

```bash
curl http://localhost:8000/documents/8f2c1234-... \
  -H "Authorization: Bearer YOUR_TOKEN"
```

---

## Carga masiva de documentos (admin)

Para sembrar la biblioteca global se usa el script:

```bash
PYTHONPATH=. python scripts/seed_documents.py
```

Lee desde `docs/admin/<Framework>/<País>/archivo.pdf` y mapea automáticamente
el folder de framework al slug correcto (`ISO 42001 → iso_42001`,
`NIST → nist_airmf`, etc.). Es idempotente: documentos `ready` se omiten salvo
que pases `--force`.

---

## Ver también

- [Pipeline de ingesta detallado](../rag/full_pipeline.md#1-ingesta-de-datos)
- [Project structure](../project_structure.md)
