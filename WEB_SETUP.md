# Web UI Setup

## Prerequisites

1. **Activate the virtual environment**
   ```
   .venv\Scripts\activate
   ```

2. **Start Ollama** (if not already running)
   ```
   ollama serve
   ```
   Model must be pulled: `ollama pull qwen2.5:7b-instruct-q4_K_M`

3. **Ingest the knowledge base** (first time only)
   ```
   python run.py --ingest data/papers/
   ```

4. **Install SSE support** (for live progress streaming)
   ```
   pip install sse-starlette>=1.8.2
   ```
   Without this, the server returns HTTP 503 on the `/stream` endpoint but
   everything else works normally.

## Start the server

```
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

Add `--reload` during development to auto-restart on file changes.

## Verify it works

```
curl http://127.0.0.1:8000/health
```

Expected: `{"status":"ok","llm":true,"rag_chunks":<n>}`

## How the UI works

When you submit a task, the server queues it as a background job and returns
`{status: "queued", task_id: "..."}` immediately. The browser then opens a
Server-Sent Events stream to `/api/tasks/{id}/stream` and shows an animated
progress bar. When the background worker completes, the stream emits a
`completed` event carrying the full result, which the UI renders as a card.

## Key endpoints

| Endpoint | Method | Description |
|---|---|---|
| `GET /` | — | Web dashboard |
| `GET /health` | — | LLM + RAG health check |
| `POST /run-task` | — | Queue a standard agent task (async) |
| `POST /research-task` | — | Queue full staged research pipeline (async) |
| `POST /ingest` | — | Ingest a file or directory into the KB |
| `GET /kan-demo` | — | Run the built-in KAN demo |
| `GET /api/tasks` | — | List all saved tasks (paginated) |
| `GET /api/tasks/{id}` | — | Get full task: messages, artifacts |
| `GET /api/tasks/search?q=...` | — | Keyword search across tasks |
| `GET /api/tasks/{id}/stream` | — | SSE stream for background task progress |
| `POST /api/tasks/{id}/messages` | — | Send follow-up message to a task |
| `POST /api/tasks/{id}/branch` | — | Branch a task from a given iteration |
| `POST /api/tasks/{id}/artifacts/{aid}/re-run` | — | Re-execute artifact with edited code |
| `POST /api/tasks/{id}/template` | — | Export task as reusable template |
| `GET /api/tasks/{id}/research-report` | — | Serve auto-generated markdown research report |
| `POST /api/research/autonomous` | — | Start autonomous iterative research loop (async) |
| `GET /api/reports/{id}/pdf` | — | Download compiled PDF report |
| `POST /api/kb/approve` | — | Ingest user-selected discovered sources into KB |
| `GET /api/kg/summary` | — | Knowledge graph summary |
| `GET /api/literature/registry` | — | Literature acquisition registry stats |
| `GET /runs` | — | List legacy CLI run files |

## Troubleshooting

| Problem | Fix |
|---|---|
| `LLM not healthy` | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull qwen2.5:7b-instruct-q4_K_M` |
| Port 8000 in use | Use `--port 8001` and update your browser URL |
| Empty RAG results | Run `python run.py --ingest data/papers/` first |
| Progress bar never completes | Install `sse-starlette`: `pip install sse-starlette>=1.8.2` |
| `/stream` returns 503 | Same as above — `sse-starlette` not installed |
