# RNSR FastAPI Wrapper

A REST API around the [`RNSRClient`](../rnsr/client.py) so you can drive the
**Recursive Neural-Symbolic Retriever** over HTTP — upload PDFs, ask questions,
extract entities, browse outlines, and run SQL-like queries on detected tables
from any HTTP client.

> **TL;DR**
> ```bash
> pip install -r api/requirements.txt 'rnsr[gemini]'
> export RNSR_API_KEY=...your-gemini-key...
> uvicorn api.main:app --reload --port 8000
> open http://localhost:8000/docs
> ```

---

## Table of contents

- [Features](#features)
- [Architecture](#architecture)
- [Install](#install)
- [Configure](#configure)
- [Run](#run)
- [API reference](#api-reference)
- [Examples](#examples)
  - [`curl`](#curl)
  - [Python](#python)
  - [JavaScript / `fetch`](#javascript--fetch)
- [Background jobs](#background-jobs)
- [Deployment](#deployment)
- [Adding authentication](#adding-authentication)
- [Troubleshooting](#troubleshooting)
- [Project layout](#project-layout)

---

## Features

- **One endpoint per `RNSRClient` capability.** Simple Q&A, RLM-Navigator
  Q&A with verification, vision Q&A, raw-text Q&A, cross-document Q&A,
  outline, structure stats, and table query/aggregate.
- **Two ways to ingest a PDF.** Multipart `file` upload, or reference an
  existing path on the server's filesystem.
- **Stable document IDs.** Each registered PDF gets a `doc_id` you reuse
  across endpoints. The mapping is persisted to a JSON registry so the
  service survives restarts.
- **Hybrid execution.** Q&A endpoints run synchronously (so callers get
  answers in a single round-trip), while explicit indexing — which can
  take minutes when building a knowledge graph — runs as a background
  job with `/jobs/{id}` polling.
- **Non-blocking event loop.** All blocking work goes through
  `asyncio.to_thread`. A per-`doc_id` lock prevents concurrent
  re-indexing of the same document.
- **Auto-generated docs.** Visit `/docs` for Swagger UI or `/redoc` for
  ReDoc — generated from Pydantic schemas, so request/response shapes
  stay accurate as the library evolves.
- **CORS-ready** for browser clients.

## Architecture

```
                          ┌─────────────────────────────┐
                          │         HTTP client          │
                          │  (curl / Python / browser)   │
                          └──────────────┬──────────────┘
                                         │
                                         ▼
       ┌─────────────────────────────────────────────────────────────┐
       │                       FastAPI app                            │
       │                                                              │
       │   routers/                                                   │
       │     health · documents · indexing · jobs · qa                │
       │     cross_doc · structure · tables                           │
       │                                                              │
       │   ──── async endpoint ─── asyncio.to_thread ────►            │
       │                                                              │
       │   AppState (singleton, built in lifespan)                    │
       │     ├── RNSRClient (with cache_dir)                          │
       │     ├── DocumentRegistry  (doc_id → path, persisted)         │
       │     ├── JobRegistry       (in-memory, per-process)           │
       │     └── per-doc_id asyncio.Locks                             │
       └─────────────────────────────────┬───────────────────────────┘
                                         │
                                         ▼
       ┌─────────────────────────────────────────────────────────────┐
       │                    RNSR library (rnsr.*)                     │
       │   ingestion → indexing → knowledge graph → RLM navigator     │
       └─────────────────────────────────────────────────────────────┘
                                         │
                                         ▼
       ┌─────────────────────────────────────────────────────────────┐
       │             $RNSR_API_STORAGE_DIR (on disk)                  │
       │   ├── uploads/<doc_id>/<filename>.pdf                        │
       │   ├── registry.json                                          │
       │   └── rnsr_cache/<cache_key>/{skeleton, kv, kg, ...}         │
       └─────────────────────────────────────────────────────────────┘
```

## Install

```bash
# From the repo root, with your existing RNSR venv activated:
pip install -r api/requirements.txt

# Plus at least one LLM provider:
pip install 'rnsr[gemini]'        # or [openai] / [anthropic]

# Optional: vision support for /ask_vision
pip install 'rnsr[vision]'
```

The wrapper itself only adds three dependencies on top of RNSR:

```text
fastapi>=0.110
uvicorn[standard]>=0.29
python-multipart>=0.0.9
```

## Configure

The server reads everything it needs from environment variables. None are
required — sensible defaults are used — but you'll typically want to set the
LLM provider and key.

| Variable                | Default                   | Purpose |
| ----------------------- | ------------------------- | ------- |
| `RNSR_API_STORAGE_DIR`  | `./.rnsr_api_storage`     | Where uploads, the registry JSON, and the RNSR cache live. |
| `RNSR_LLM_PROVIDER`     | _auto-detected_           | `openai`, `anthropic`, or `gemini`. |
| `RNSR_LLM_MODEL`        | provider default          | Override the model name. |
| `RNSR_API_KEY`          | _none_                    | API key for the provider. Falls back to `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY` from the environment or a `.env` file. |
| `RNSR_API_CORS_ORIGINS` | `*`                       | Comma-separated list of allowed CORS origins. |

> **Model names change frequently.** Confirm the current catalogue from
> your provider's docs before pinning a value.

A typical `.env` for local development:

```dotenv
RNSR_API_STORAGE_DIR=./.rnsr_api_storage
RNSR_LLM_PROVIDER=gemini
RNSR_LLM_MODEL=gemini-2.5-flash
RNSR_API_KEY=AIza...
RNSR_API_CORS_ORIGINS=http://localhost:5173,http://localhost:3000
```

## Run

**Development** (auto-reload on file changes):

```bash
uvicorn api.main:app --reload --port 8000
```

**Production-style** single process:

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
```

> Use a **single worker**. The service holds in-memory caches (skeleton
> index, knowledge graph, navigator instances) per `RNSRClient`. Multiple
> workers would each maintain their own cache and step on each other's
> work. If you need horizontal scaling, put a load balancer in front of
> several single-worker instances and pin clients via sticky sessions, or
> share storage and accept cache duplication.

Once running:

- Swagger UI: <http://localhost:8000/docs>
- ReDoc: <http://localhost:8000/redoc>
- OpenAPI JSON: <http://localhost:8000/openapi.json>

## API reference

| Method     | Path                                                      | Description |
| ---------- | --------------------------------------------------------- | ----------- |
| `GET`      | `/health`                                                 | Liveness probe + version info. |
| `POST`     | `/documents`                                              | Register a PDF. Provide **either** multipart `file` **or** form field `path`. Returns `doc_id`. |
| `GET`      | `/documents`                                              | List registered documents. |
| `GET`      | `/documents/{doc_id}`                                     | Document metadata + cache status. |
| `DELETE`   | `/documents/{doc_id}?delete_files=false`                  | Remove from registry. With `delete_files=true`, also deletes uploaded bytes. |
| `POST`     | `/documents/{doc_id}/index`                               | Build skeleton (and optionally knowledge graph) in the background. Returns `job_id`. |
| `GET`      | `/jobs`                                                   | List background jobs. |
| `GET`      | `/jobs/{job_id}`                                          | Get job status / result / error. |
| `POST`     | `/documents/{doc_id}/ask`                                 | Simple Q&A → just the answer string. |
| `POST`     | `/documents/{doc_id}/ask_advanced`                        | Full RLM Navigator → answer, confidence, full result dict. |
| `POST`     | `/documents/{doc_id}/ask_vision`                          | Vision/hybrid Q&A on page images. Requires `rnsr[vision]`. |
| `POST`     | `/ask/text`                                               | Q&A over raw text — no PDF required. |
| `POST`     | `/ask/cross-document`                                     | Question spanning multiple registered documents. |
| `GET`      | `/documents/{doc_id}/structure`                           | Hierarchy stats (sections, depth, character counts). |
| `GET`      | `/documents/{doc_id}/outline?max_depth=2`                 | Table of contents. |
| `GET`      | `/documents/{doc_id}/tables`                              | List auto-detected tables. |
| `POST`     | `/documents/{doc_id}/tables/{table_id}/query`             | SQL-like SELECT/WHERE/ORDER BY/LIMIT. |
| `POST`     | `/documents/{doc_id}/tables/{table_id}/aggregate`         | sum / avg / count / min / max on a numeric column. |

Full request/response schemas are documented at `/docs`.

## Examples

### `curl`

**Register a PDF (upload):**
```bash
curl -F "file=@samples/contract.pdf" http://localhost:8000/documents
# → { "doc_id": "71466555079d4545", ... }
```

**Register a PDF (server-side path):**
```bash
curl -F "path=/srv/docs/contract.pdf" http://localhost:8000/documents
```

**Simple ask** (synchronous; will index on first call):
```bash
curl -X POST http://localhost:8000/documents/71466555079d4545/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "What are the payment terms?"}'
```

**Advanced ask with verification:**
```bash
curl -X POST http://localhost:8000/documents/71466555079d4545/ask_advanced \
  -H 'Content-Type: application/json' \
  -d '{
        "question": "What is the liability cap?",
        "enable_verification": true,
        "max_recursion_depth": 4
      }'
```

**Pre-warm the index in the background** and poll the job:
```bash
JOB=$(curl -s -X POST http://localhost:8000/documents/71466555079d4545/index \
        -H 'Content-Type: application/json' \
        -d '{"build_knowledge_graph": true}' | jq -r .job_id)

watch -n 2 "curl -s http://localhost:8000/jobs/$JOB | jq ."
```

**Browse the outline:**
```bash
curl 'http://localhost:8000/documents/71466555079d4545/outline?max_depth=3' | jq
```

**Query a detected table:**
```bash
curl -X POST http://localhost:8000/documents/$DOC_ID/tables/table_001/query \
  -H 'Content-Type: application/json' \
  -d '{
        "columns": ["Quarter", "Revenue"],
        "where":   {"Revenue": {"op": ">=", "value": 1000}},
        "order_by": "-Revenue",
        "limit":   10
      }'
```

**Cross-document question:**
```bash
curl -X POST http://localhost:8000/ask/cross-document \
  -H 'Content-Type: application/json' \
  -d '{
        "doc_ids": ["doc_a_id", "doc_b_id"],
        "question": "What contradicts between these two reports?"
      }'
```

### Python

```python
import requests

BASE = "http://localhost:8000"

# 1. Upload
with open("contract.pdf", "rb") as f:
    doc = requests.post(f"{BASE}/documents", files={"file": f}).json()
doc_id = doc["doc_id"]

# 2. (Optional) Pre-warm the index — returns immediately
job = requests.post(
    f"{BASE}/documents/{doc_id}/index",
    json={"build_knowledge_graph": True},
).json()

# 3. Poll until ready
import time
while True:
    status = requests.get(f"{BASE}/jobs/{job['job_id']}").json()
    if status["status"] in ("completed", "failed"):
        break
    time.sleep(2)
print("Index ready:", status["result"])

# 4. Ask
r = requests.post(
    f"{BASE}/documents/{doc_id}/ask_advanced",
    json={
        "question": "Who are the parties and what are the payment terms?",
        "enable_verification": True,
    },
).json()

print(r["answer"])
print(f"confidence: {r['confidence']:.2f}")
```

### JavaScript / `fetch`

```js
const BASE = "http://localhost:8000";

// 1. Upload (e.g. from an <input type="file">)
async function upload(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${BASE}/documents`, { method: "POST", body: fd });
  return (await r.json()).doc_id;
}

// 2. Ask
async function ask(docId, question) {
  const r = await fetch(`${BASE}/documents/${docId}/ask_advanced`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, enable_verification: true }),
  });
  return r.json();
}

const docId = await upload(document.querySelector("input[type=file]").files[0]);
const result = await ask(docId, "What are the payment terms?");
console.log(result.answer, "confidence:", result.confidence);
```

## Background jobs

Indexing endpoints return a `job_id` immediately:

```json
{
  "job_id": "c85e861380fb48db",
  "kind":   "index",
  "doc_id": "71466555079d4545",
  "status": "queued",
  "created_at": "2026-05-06T03:20:21.982194+00:00"
}
```

Poll `GET /jobs/{job_id}` until `status` is `"completed"` or `"failed"`:

```json
{
  "job_id": "c85e861380fb48db",
  "kind":   "index",
  "doc_id": "71466555079d4545",
  "status": "completed",
  "started_at":  "...",
  "finished_at": "...",
  "result": {
    "doc_id": "71466555079d4545",
    "nodes": 12,
    "knowledge_graph_built": true,
    "kg_entities": 47,
    "kg_relationships": 19
  }
}
```

Notes:

- Jobs live in memory only — restarting the server clears the list.
  Indexing is idempotent, so this is safe; just re-issue the request.
- Concurrent `POST .../index` calls for the **same** `doc_id` are
  serialised with a per-document `asyncio.Lock`. Different documents
  index concurrently as expected.
- The Q&A endpoints transparently index on demand, so calling
  `/documents/{id}/index` first is purely a UX/latency optimisation.

## Deployment

### Docker

```dockerfile
FROM python:3.12-slim
WORKDIR /app

COPY pyproject.toml requirements.txt ./
COPY rnsr ./rnsr
COPY api ./api
RUN pip install --no-cache-dir -r requirements.txt -r api/requirements.txt 'rnsr[gemini]'

ENV RNSR_API_STORAGE_DIR=/data
VOLUME ["/data"]
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
```

```bash
docker build -t rnsr-api .
docker run --rm -p 8000:8000 \
  -e RNSR_API_KEY=$GOOGLE_API_KEY \
  -e RNSR_LLM_PROVIDER=gemini \
  -v rnsr-data:/data \
  rnsr-api
```

### `systemd` unit

```ini
[Unit]
Description=RNSR API
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/rnsr
EnvironmentFile=/etc/rnsr-api.env
ExecStart=/opt/rnsr/.venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 1
Restart=on-failure
User=rnsr

[Install]
WantedBy=multi-user.target
```

## Adding authentication

The wrapper ships **without** authentication so it stays out of your way for
local use. A minimal API-key check is just a few lines — drop it in front of
the routers:

```python
import os
from fastapi import Depends, Header, HTTPException

def require_api_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("RNSR_API_AUTH_KEY")
    if not expected or x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

# In api/main.py:
app.include_router(documents.router, dependencies=[Depends(require_api_key)])
# ...repeat for the other routers
```

For anything richer (OAuth, JWTs, per-user rate limiting), wire in
[`fastapi-users`](https://fastapi-users.github.io/fastapi-users/) or terminate
auth at your reverse proxy / API gateway.

## Troubleshooting

| Symptom                                              | Likely cause / fix |
| ---------------------------------------------------- | ------------------ |
| `503 Application state is not initialised`           | The lifespan handler crashed during startup. Check the uvicorn logs — usually a missing dependency or a bad env var. |
| `400 Provide exactly one of file or path`            | You sent both, or neither. Pick one. |
| `404 File not found: ...`                            | The `path` you registered no longer exists, or relative paths weren't resolved as you expected. Use absolute paths. |
| `410 Document file no longer exists at ...`          | The PDF was deleted after registration. Re-upload or remove the registry entry with `DELETE /documents/{id}`. |
| `501 Vision dependencies are not installed`          | Install with `pip install 'rnsr[vision]'`. |
| Indexing job stays in `running` for a long time      | Knowledge-graph extraction calls the LLM once per (or per batch of) document section. Large PDFs with `build_knowledge_graph=true` legitimately take minutes. Watch your LLM provider quota / errors in the server logs. |
| 500s with `No LLM provider available` (or similar)   | `RNSR_API_KEY` (or `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / `GOOGLE_API_KEY`) isn't set, or `RNSR_LLM_PROVIDER` doesn't match the key you provided. |
| Different answers across requests                    | LLMs are stochastic. Set a deterministic temperature in your provider config if you need reproducibility. |

## Project layout

```
api/
├── __init__.py
├── main.py            # FastAPI app + lifespan + CORS
├── dependencies.py    # AppState, RNSRClient singleton, dep helpers
├── registry.py        # Persisted doc_id → path registry
├── jobs.py            # In-memory background-job tracker
├── schemas.py         # Pydantic request/response models
├── routers/
│   ├── __init__.py
│   ├── health.py      # GET /health
│   ├── documents.py   # POST/GET/DELETE /documents
│   ├── indexing.py    # POST /documents/{id}/index (background)
│   ├── jobs.py        # GET /jobs[, /{id}]
│   ├── qa.py          # ask, ask_advanced, ask_vision, /ask/text
│   ├── cross_doc.py   # /ask/cross-document
│   ├── structure.py   # /structure, /outline
│   └── tables.py      # /tables, /tables/{id}/query, /aggregate
├── requirements.txt   # fastapi, uvicorn[standard], python-multipart
└── README.md          # ← you are here
```

---

Built on top of the [`rnsr`](../rnsr) library. PRs and issues welcome.
