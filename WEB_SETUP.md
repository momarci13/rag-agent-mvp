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

## Key endpoints

| Endpoint | Description |
|---|---|
| `GET /` | Web dashboard |
| `GET /health` | LLM + RAG health check |
| `POST /run-task` | Run a standard agent task |
| `POST /research-task` | Full staged research pipeline |
| `POST /ingest` | Ingest a file or directory into the KB |
| `GET /kan-demo` | Run the built-in KAN demo |
| `GET /api/tasks` | List all saved tasks (paginated) |
| `GET /api/tasks/{id}` | Get full task: messages, artifacts |
| `GET /api/tasks/search?q=...` | Keyword search across tasks |
| `POST /api/tasks/{id}/messages` | Send follow-up message to a task |
| `POST /api/tasks/{id}/branch` | Branch a task from a given iteration |
| `POST /api/tasks/{id}/artifacts/{aid}/re-run` | Re-execute artifact with edited code |
| `POST /api/tasks/{id}/template` | Export task as reusable template |
| `GET /api/reports/{id}/pdf` | Download compiled PDF report |
| `GET /api/kg/summary` | Knowledge graph summary |
| `GET /api/literature/registry` | Literature acquisition registry stats |
| `GET /runs` | List legacy CLI run files |

## Troubleshooting

| Problem | Fix |
|---|---|
| `LLM not healthy` | Run `ollama serve` in a separate terminal |
| `Model not found` | Run `ollama pull qwen2.5:7b-instruct-q4_K_M` |
| Port 8000 in use | Use `--port 8001` and update your browser URL |
| Empty RAG results | Run `python run.py --ingest data/papers/` first |
