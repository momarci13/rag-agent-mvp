from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

try:
    from sse_starlette.sse import EventSourceResponse
    _SSE_AVAILABLE = True
except ImportError:
    _SSE_AVAILABLE = False

logger = logging.getLogger(__name__)

# In-memory registry for background task progress
_bg_tasks: dict[str, dict] = {}

from run import load_config, make_llm_config, make_tools, run_kan_demo
from agents.llm import LLMConfig, OllamaLLM
from tools.backtest import BacktestConfig, compile_signal, run_portfolio_backtest
from tools.data import fetch_yahoo
from tools import task_storage, task_conversation

ROOT = Path(__file__).resolve().parent
CONFIG_PATH = ROOT / "configs" / "config.yaml"
OUTPUT_RUNS = ROOT / "output" / "runs"
OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="Finance Assistant.ai")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


# Startup: configure logging + run migration from legacy format
@app.on_event("startup")
async def startup_migration():
    (ROOT / "output").mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(str(ROOT / "output" / "agent.log")),
        ],
    )
    try:
        migration_result = task_storage.migrate_legacy_runs()
        if migration_result["migrated"] > 0:
            logger.info("[STARTUP] Migrated %d legacy tasks", migration_result["migrated"])
    except Exception as e:
        logger.warning("[STARTUP] Migration warning: %s", e)


def _config() -> dict[str, Any]:
    return load_config(str(CONFIG_PATH))


def _llm(cfg: dict[str, Any]) -> OllamaLLM:
    llm_cfg = make_llm_config(cfg)
    return OllamaLLM(llm_cfg)


def _rag(cfg: dict[str, Any]):
    try:
        from rag.hybrid import LiteHybridRAG
    except Exception as exc:
        raise HTTPException(500, f"RAG backend unavailable: {exc}")
    return LiteHybridRAG(
        db_path=cfg["rag"]["db_path"],
        embedding_model=cfg["rag"]["embedding_model"],
        alpha_dense=cfg["rag"]["alpha_dense"],
        query_expansion_enabled=cfg["rag"]["query_expansion"]["enabled"],
        query_expansion_method=cfg["rag"]["query_expansion"]["method"],
        max_expansions=cfg["rag"]["query_expansion"]["max_expansions"],
        reranking_enabled=cfg["rag"]["reranking"]["enabled"],
        reranking_model=cfg["rag"]["reranking"]["model"],
        top_k_before_rerank=cfg["rag"]["reranking"]["top_k_before_rerank"],
        top_k_after_rerank=cfg["rag"]["reranking"]["top_k_after_rerank"],
    )


def _build_task_response(state: Any, task_id: str) -> dict[str, Any]:
    """Extract narrative, code, and PDF URL from the last non-literature artifact."""
    last_artifact = next(
        (a for a in reversed(state.artifacts) if a.get("type") != "literature"), None
    )
    narrative: dict = {}
    code: str = ""
    pdf_url: str | None = None

    if last_artifact:
        report = last_artifact.get("report", {})
        narrative = report.get("narrative", {})

        art_type = last_artifact.get("type", "")
        payload = last_artifact.get("payload", {})
        if art_type == "ds":
            code = payload.get("code", "")
        elif art_type == "quant":
            spec = payload.get("spec", {})
            code = spec.get("signal_code", "")
        elif art_type == "writing":
            code = payload.get("tex", "")[:3000]

        pdf_info = report.get("pdf")
        if isinstance(pdf_info, dict) and pdf_info.get("pdf"):
            pdf_url = f"/api/reports/{task_id}/pdf"

    return {
        "status": "ok",
        "task_id": task_id,
        "narrative": narrative,
        "code": code,
        "pdf_url": pdf_url,
        "run": task_storage._serialize_for_json(state),
    }


def _save_run(state: Any) -> Path:
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    idx = len(list(OUTPUT_RUNS.glob("run_*.json")))
    run_path = OUTPUT_RUNS / f"run_{idx:04d}.json"
    run_path.write_text(json.dumps(asdict(state), indent=2, default=str), encoding="utf-8")
    return run_path


# ── Background task worker functions ─────────────────────────────────────────

def _run_task_background_sync(task_id: str, task_text: str, cfg: dict) -> None:
    _bg_tasks[task_id] = {"status": "running", "progress": "Starting LLM pipeline..."}
    try:
        llm = _llm(cfg)
        _bg_tasks[task_id]["progress"] = "Retrieving context..."
        rag = _rag(cfg)
        tools = make_tools(cfg)
        from agents.graph import run
        _bg_tasks[task_id]["progress"] = "Running analysis..."
        state = run(task_text, llm, rag, max_iter=1, tools=tools)
        _bg_tasks[task_id]["progress"] = "Saving results..."
        saved_id = task_storage.save_task(state)
        result = _build_task_response(state, saved_id)
        _bg_tasks[task_id] = {"status": "completed", "result": result}
        logger.info("[SERVER] run_task completed: bg_id=%s task_id=%s", task_id, saved_id)
    except Exception as exc:
        logger.exception("[SERVER] run_task background failed: %s", task_id)
        _bg_tasks[task_id] = {"status": "failed", "error": str(exc)}


def _run_research_background_sync(
    task_id: str, task_text: str, n_papers: int, kg_enabled: bool, cfg: dict
) -> None:
    _bg_tasks[task_id] = {"status": "running", "progress": "Starting research pipeline..."}
    try:
        llm = _llm(cfg)
        _bg_tasks[task_id]["progress"] = "Acquiring literature..."
        rag = _rag(cfg)
        tools = make_tools(cfg)
        from agents.graph import research_run
        research_cfg = cfg.get("research", {})
        _bg_tasks[task_id]["progress"] = "Running research pipeline..."
        state = research_run(
            task_text, llm, rag,
            max_iter=cfg["agent"]["max_iterations"],
            tools=tools,
            n_papers=n_papers or research_cfg.get("n_papers", 8),
            kg_enabled=kg_enabled and research_cfg.get("kg_enabled", True),
        )
        _bg_tasks[task_id]["progress"] = "Saving results..."
        saved_id = task_storage.save_task(state)
        result = _build_task_response(state, saved_id)
        _bg_tasks[task_id] = {"status": "completed", "result": result}
        logger.info("[SERVER] research_task completed: bg_id=%s task_id=%s", task_id, saved_id)
    except Exception as exc:
        logger.exception("[SERVER] research_task background failed: %s", task_id)
        _bg_tasks[task_id] = {"status": "failed", "error": str(exc)}


def _run_autonomous_research_background_sync(
    task_id: str, topic: str, n_iterations: int, n_papers_per_iter: int, cfg: dict
) -> None:
    _bg_tasks[task_id] = {"status": "running", "progress": "Starting autonomous research loop..."}
    try:
        from tools.research_loop import autonomous_research_loop
        from tools.citation_dag import CitationDAG
        llm = _llm(cfg)
        rag = _rag(cfg)
        dag = CitationDAG()
        loop_cfg = cfg.get("autonomous_loop", {})
        _bg_tasks[task_id]["progress"] = f"Running {n_iterations} research iterations..."
        result = autonomous_research_loop(
            topic=topic,
            llm=llm,
            rag=rag,
            citation_dag=dag,
            task_id=task_id,
            n_iterations=n_iterations,
            n_papers_per_iter=n_papers_per_iter,
            quality_threshold=loop_cfg.get("quality_threshold", 0.4),
        )
        _bg_tasks[task_id] = {
            "status": "completed",
            "result": {
                "task_id": task_id,
                "topic": topic,
                "total_papers": result.total_papers_ingested,
                "iterations": len(result.iterations),
                "dag_nodes": result.citation_dag_nodes,
                "report_url": f"/api/tasks/{task_id}/research-report",
            },
        }
        logger.info("[SERVER] Autonomous research completed: %s", task_id)
    except Exception as exc:
        logger.exception("[SERVER] Autonomous research failed: %s", task_id)
        _bg_tasks[task_id] = {"status": "failed", "error": str(exc)}


# ── Pydantic request models ───────────────────────────────────────────────────

class TaskRequest(BaseModel):
    task: str


class ResearchTaskRequest(BaseModel):
    task: str
    n_papers: int = 8
    kg_enabled: bool = True


class IngestRequest(BaseModel):
    path: str


class MessageRequest(BaseModel):
    content: str
    iteration: int = 0


class BranchRequest(BaseModel):
    branch_name: str | None = None
    from_iteration: int = 0


class ReRunRequest(BaseModel):
    code: str


class ApproveSourcesRequest(BaseModel):
    task_id: str
    source_ids: list[str]


class AutonomousResearchRequest(BaseModel):
    topic: str
    n_iterations: int = 3
    n_papers_per_iter: int = 6


@app.get("/", response_class=FileResponse)
async def index():
    return ROOT / "web" / "index.html"


@app.get("/health")
async def health() -> dict[str, Any]:
    cfg = _config()
    llm = _llm(cfg)
    llm_ok = False
    try:
        llm_ok = llm.health()
    except Exception as exc:
        return {"status": "error", "detail": f"LLM health check failed: {exc}"}

    try:
        rag = _rag(cfg)
        rag_count = len(rag)
    except HTTPException as exc:
        rag_count = None
        return {"status": "partial", "llm": llm_ok, "rag": str(exc.detail)}

    return {"status": "ok", "llm": llm_ok, "rag_chunks": rag_count}


@app.post("/run-task")
async def run_task(payload: TaskRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    cfg = _config()
    # Quick LLM health-check before queuing (fast, catches offline Ollama early)
    if not _llm(cfg).health():
        raise HTTPException(500, "LLM is not healthy or unreachable.")
    task_id = str(uuid4())
    _bg_tasks[task_id] = {"status": "queued", "progress": "Task queued"}
    background_tasks.add_task(_run_task_background_sync, task_id, payload.task, cfg)
    return {"status": "queued", "task_id": task_id}


@app.post("/research-task")
async def research_task(payload: ResearchTaskRequest, background_tasks: BackgroundTasks) -> dict[str, Any]:
    """Full staged research pipeline: literature + hypothesis + experiment + KG."""
    cfg = _config()
    if not _llm(cfg).health():
        raise HTTPException(500, "LLM is not healthy or unreachable.")
    task_id = str(uuid4())
    _bg_tasks[task_id] = {"status": "queued", "progress": "Research task queued"}
    background_tasks.add_task(
        _run_research_background_sync,
        task_id, payload.task, payload.n_papers, payload.kg_enabled, cfg,
    )
    return {"status": "queued", "task_id": task_id}


@app.get("/api/reports/{task_id}/pdf", response_class=FileResponse)
async def get_report_pdf(task_id: str):
    """Serve the compiled PDF report for a task."""
    task = task_storage.load_task(task_id)
    if not task:
        raise HTTPException(404, "Task not found")

    for art in reversed(task.artifacts):
        report = art.get("report", {})
        pdf_info = report.get("pdf")
        if isinstance(pdf_info, dict):
            pdf_path = pdf_info.get("pdf")
            if pdf_path and Path(pdf_path).exists():
                return FileResponse(
                    pdf_path,
                    media_type="application/pdf",
                    filename=f"report_{task_id[:8]}.pdf",
                )

    raise HTTPException(404, "PDF not found for this task (compilation may have failed)")


@app.get("/api/tasks/{task_id}/stream")
async def stream_task_progress(task_id: str):
    """SSE endpoint streaming progress for a background task."""
    if not _SSE_AVAILABLE:
        raise HTTPException(
            503,
            "sse-starlette is not installed. "
            "Run: pip install sse-starlette>=1.8.2",
        )

    async def event_generator():
        while True:
            meta = _bg_tasks.get(task_id)
            if meta is None:
                yield {
                    "event": "progress",
                    "data": json.dumps({"status": "unknown", "progress": "Waiting for task..."}),
                }
            elif meta["status"] == "completed":
                yield {"event": "completed", "data": json.dumps(meta.get("result", {}))}
                break
            elif meta["status"] == "failed":
                yield {
                    "event": "failed",
                    "data": json.dumps({"error": meta.get("error", "Unknown error")}),
                }
                break
            else:
                yield {
                    "event": "progress",
                    "data": json.dumps({
                        "status": meta["status"],
                        "progress": meta.get("progress", ""),
                    }),
                }
            await asyncio.sleep(1.5)

    return EventSourceResponse(event_generator())


@app.get("/api/tasks/{task_id}/research-report")
async def get_research_report(task_id: str):
    """Serve the auto-generated markdown research report for a task."""
    from tools.auto_report import load_report
    content = load_report(task_id)
    if content is None:
        raise HTTPException(404, "Research report not found (not yet generated for this task)")
    return PlainTextResponse(content, media_type="text/markdown")


@app.post("/api/research/autonomous")
async def autonomous_research(
    payload: AutonomousResearchRequest,
    background_tasks: BackgroundTasks,
) -> dict[str, Any]:
    """Start an autonomous iterative research loop (background)."""
    cfg = _config()
    if not _llm(cfg).health():
        raise HTTPException(500, "LLM is not healthy or unreachable.")
    task_id = str(uuid4())
    _bg_tasks[task_id] = {"status": "queued", "progress": "Autonomous research queued"}
    background_tasks.add_task(
        _run_autonomous_research_background_sync,
        task_id, payload.topic, payload.n_iterations, payload.n_papers_per_iter, cfg,
    )
    return {"status": "queued", "task_id": task_id}


@app.get("/api/kg/summary")
async def kg_summary() -> dict[str, Any]:
    """Return a summary of the knowledge graph."""
    try:
        from kg.graph import ResearchKnowledgeGraph
        kg = ResearchKnowledgeGraph()
        papers = kg.find_by_type("paper")
        findings = kg.find_by_type("finding")
        tasks = kg.find_by_type("task")
        return {
            "summary": kg.summarize(),
            "nodes": kg.G.number_of_nodes(),
            "edges": kg.G.number_of_edges(),
            "papers": [{"arxiv_id": p.get("arxiv_id"), "title": p.get("title"), "year": p.get("year")}
                       for p in papers[:20]],
            "recent_findings": [{"text": f.get("text", "")[:120], "task_id": f.get("source_task_id")}
                                 for f in findings[-5:]],
            "tasks": len(tasks),
        }
    except Exception as e:
        raise HTTPException(500, f"Knowledge graph unavailable: {e}")


@app.get("/api/literature/registry")
async def literature_registry() -> dict[str, Any]:
    """Return stats from the literature acquisition registry."""
    try:
        from tools.literature import registry_stats
        return registry_stats()
    except Exception as e:
        raise HTTPException(500, f"Literature registry unavailable: {e}")


@app.post("/ingest")
async def ingest(payload: IngestRequest) -> dict[str, Any]:
    cfg = _config()
    try:
        from rag.hybrid import LiteHybridRAG
        from rag.ingest import ingest_path
    except Exception as exc:
        raise HTTPException(500, f"RAG ingestion unavailable: {exc}")

    rag = LiteHybridRAG(
        db_path=cfg["rag"]["db_path"],
        embedding_model=cfg["rag"]["embedding_model"],
        alpha_dense=cfg["rag"]["alpha_dense"],
        query_expansion_enabled=cfg["rag"]["query_expansion"]["enabled"],
        query_expansion_method=cfg["rag"]["query_expansion"]["method"],
        max_expansions=cfg["rag"]["query_expansion"]["max_expansions"],
        reranking_enabled=cfg["rag"]["reranking"]["enabled"],
        reranking_model=cfg["rag"]["reranking"]["model"],
        top_k_before_rerank=cfg["rag"]["reranking"]["top_k_before_rerank"],
        top_k_after_rerank=cfg["rag"]["reranking"]["top_k_after_rerank"],
    )
    n = ingest_path(payload.path, rag, chunk_tokens=cfg["rag"]["chunk_tokens"])
    return {"status": "ok", "chunks": len(rag), "added": n}


@app.get("/kan-demo")
async def kan_demo() -> dict[str, Any]:
    result = run_kan_demo()
    return {"status": "ok", "demo": result}


@app.get("/runs")
async def list_runs() -> dict[str, Any]:
    OUTPUT_RUNS.mkdir(parents=True, exist_ok=True)
    files = sorted(OUTPUT_RUNS.glob("run_*.json"))
    return {"runs": [f.name for f in files]}


@app.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    run_path = OUTPUT_RUNS / run_id
    if not run_path.exists() or not run_path.is_file():
        raise HTTPException(404, "Run file not found")
    content = run_path.read_text(encoding="utf-8")
    return json.loads(content)


# New Task Management API Endpoints
@app.get("/api/tasks")
async def list_tasks_api(limit: int = 50, offset: int = 0, sort_by: str = "-updated_at") -> dict[str, Any]:
    """List all tasks with pagination and sorting."""
    try:
        tasks, total = task_storage.list_tasks(limit=limit, offset=offset, sort_by=sort_by)
        return {"tasks": tasks, "total": total, "limit": limit, "offset": offset}
    except Exception as e:
        raise HTTPException(500, f"Failed to list tasks: {str(e)}")


@app.get("/api/tasks/{task_id}")
async def get_task_api(task_id: str) -> dict[str, Any]:
    """Get full task with all messages and artifacts."""
    try:
        task = task_storage.load_task(task_id)
        if not task:
            raise HTTPException(404, "Task not found")
        
        # Serialize task for JSON response
        task_data = task_storage._serialize_for_json(task)
        return {"task": task_data}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Failed to load task: {str(e)}")


@app.get("/api/tasks/search")
async def search_tasks_api(q: str, limit: int = 20) -> dict[str, Any]:
    """Search tasks by keyword."""
    try:
        results = task_storage.search_tasks(q, limit=limit)
        return {"results": results, "query": q, "count": len(results)}
    except Exception as e:
        raise HTTPException(500, f"Search failed: {str(e)}")


@app.post("/api/tasks/{task_id}/messages")
async def add_task_message(task_id: str, payload: MessageRequest) -> dict[str, Any]:
    """Add a user message and get assistant response."""
    try:
        cfg = _config()
        llm = _llm(cfg)
        rag = _rag(cfg)
        
        response, artifacts, discovered_sources = task_conversation.process_user_message(
            task_id=task_id,
            message_content=payload.content,
            llm=llm,
            rag=rag,
            iteration=payload.iteration,
        )

        return {
            "status": "ok",
            "assistant_response": response,
            "new_artifacts": artifacts,
            "message_id": f"{task_id}_{payload.iteration}",
            "task_id": task_id,
            "discovered_sources": discovered_sources,
        }
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to process message: {str(e)}")


@app.post("/api/tasks/{task_id}/branch")
async def branch_task_api(task_id: str, payload: BranchRequest) -> dict[str, Any]:
    """Create a branched copy of a task."""
    try:
        new_task_id = task_conversation.branch_task(
            task_id=task_id,
            branch_name=payload.branch_name,
            from_iteration=payload.from_iteration,
        )
        return {
            "status": "ok",
            "new_task_id": new_task_id,
            "branch_name": payload.branch_name,
            "parent_id": task_id,
        }
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to branch task: {str(e)}")


@app.post("/api/tasks/{task_id}/artifacts/{artifact_id}/re-run")
async def re_run_artifact_api(task_id: str, artifact_id: str, payload: ReRunRequest) -> dict[str, Any]:
    """Re-execute an artifact with edited code."""
    try:
        result = task_conversation.re_execute_artifact(
            task_id=task_id,
            artifact_id=artifact_id,
            edited_code=payload.code,
        )
        return {
            "status": "ok" if result["returncode"] == 0 else "error",
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "returncode": result.get("returncode", -1),
            "execution_time": result.get("execution_time", 0),
        }
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to re-run artifact: {str(e)}")


@app.post("/api/kb/approve")
async def approve_sources(payload: ApproveSourcesRequest) -> dict[str, Any]:
    """Ingest user-selected pending sources into the knowledge base."""
    from tools import source_search
    pending = source_search.load_pending_sources(payload.task_id)
    approved_ids = set(payload.source_ids)
    to_ingest = [s for s in pending if s.id in approved_ids]
    if to_ingest:
        cfg = _config()
        rag = _rag(cfg)
        rag.ingest_papers(to_ingest)
    source_search.clear_pending_sources(payload.task_id)
    return {"status": "ok", "ingested": len(to_ingest)}


@app.post("/api/tasks/{task_id}/template")
async def export_template_api(task_id: str) -> dict[str, Any]:
    """Export task as a reusable template."""
    try:
        template_data = task_storage.export_template(task_id)
        # Save template
        templates_dir = Path("output") / "templates"
        templates_dir.mkdir(parents=True, exist_ok=True)
        template_id = str(uuid4())
        template_file = templates_dir / f"{template_id}.json"
        template_file.write_text(json.dumps(template_data, indent=2, default=str), encoding="utf-8")
        
        return {
            "status": "ok",
            "template_id": template_id,
            "task_id": task_id,
        }
    except ValueError as e:
        raise HTTPException(404, str(e))
    except Exception as e:
        raise HTTPException(500, f"Failed to export template: {str(e)}")
