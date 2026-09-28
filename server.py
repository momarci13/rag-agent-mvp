"""HTTP API and web UI for the Modeler / Validator studio.

Run:  uvicorn server:app --reload
Mutating endpoints require the header X-Studio-Token matching the
STUDIO_API_TOKEN environment variable; they are disabled when it is unset.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import threading
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from studio.config import ROOT, StudioSettings, load_config, make_llm_config
from studio.frameworks import FRAMEWORK_LABELS
from studio.orchestrator import InputError, Studio
from studio.schemas import Project, ProjectInput, Signoff
from studio.storage import resolve_inside

logger = logging.getLogger(__name__)
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
DOWNLOADABLE = {".docx", ".pdf", ".py", ".json", ".png", ".csv", ".xml", ".txt", ".md", ".ini"}
_running: set[str] = set()
_running_lock = threading.Lock()


def _config() -> dict[str, Any]:
    return load_config()


def _llm(cfg: dict[str, Any]):
    from agents.llm import HostedLLM

    return HostedLLM(make_llm_config(cfg))


@lru_cache(maxsize=1)
def _studio() -> Studio:
    from studio.library import Library, make_rag
    from studio.storage import ProjectStore

    cfg = _config()
    settings = StudioSettings.from_config(cfg)
    rag = make_rag(cfg, settings.library_collection)
    library = Library(settings.output_dir.parent / "library", rag)
    try:
        library.ensure_builtin(settings.builtin_corpus_dir)
    except Exception as exc:  # noqa: BLE001 - library still usable for uploads later
        logger.warning("Built-in reference corpus not ingested: %s", exc)
    return Studio(cfg, _llm(cfg), library, ProjectStore(settings.output_dir), settings)


def _recover_interrupted() -> None:
    try:
        studio = _studio()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Studio not initialised at startup: %s", exc)
        return
    for project in studio.store.list():
        if project.status in ("queued", "running") and project.project_id not in _running:
            project.status = "failed"
            project.error = "The server stopped while this project was running. Start a new project to retry."
            studio.store.save(project)


@asynccontextmanager
async def lifespan(_: FastAPI):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s %(message)s")
    _recover_interrupted()
    yield


app = FastAPI(title="Model Development & Validation Studio", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "web"), name="static")


def _require_token(provided: str | None) -> None:
    expected = os.getenv("STUDIO_API_TOKEN", "")
    if not expected:
        raise HTTPException(503, "Disabled until STUDIO_API_TOKEN is set on the server")
    if provided is None or not secrets.compare_digest(provided, expected):
        raise HTTPException(401, "Invalid X-Studio-Token")


def _load(project_id: str) -> Project:
    try:
        return _studio().store.load(project_id)
    except (FileNotFoundError, ValueError):
        raise HTTPException(404, "Project not found") from None


async def _read_uploads(files: list[UploadFile] | None) -> list[tuple[str, bytes]]:
    out = []
    for f in files or []:
        if not f.filename:
            continue
        content = await f.read()
        if len(content) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, f"{f.filename} is larger than 200 MB")
        out.append((f.filename, content))
    return out


def _start(project_id: str) -> None:
    with _running_lock:
        if project_id in _running:
            return
        _running.add(project_id)

    def work() -> None:
        try:
            _studio().run(project_id)
        finally:
            with _running_lock:
                _running.discard(project_id)

    threading.Thread(target=work, name=f"studio-{project_id}", daemon=True).start()


def _summary(p: Project) -> dict[str, Any]:
    last = p.current_round()
    open_f = p.open_findings()
    return {
        "project_id": p.project_id,
        "title": p.input.title,
        "mode": p.input.mode,
        "status": p.status,
        "created_at": p.created_at,
        "updated_at": p.updated_at,
        "rounds": len(p.rounds),
        "max_rounds": p.input.max_rounds,
        "outcome": last.validator.outcome if last else None,
        "open_findings": {s: sum(1 for f in open_f if f.severity == s) for s in ("critical", "high", "medium", "low")},
        "last_event": p.trace[-1].model_dump() if p.trace else None,
    }


# --------------------------------------------------------------------------
# Pages and status
# --------------------------------------------------------------------------

@app.get("/", response_class=FileResponse)
async def index():
    return ROOT / "web" / "index.html"


@app.get("/health")
async def health() -> dict[str, Any]:
    cfg = _config()
    llm = _llm(cfg)
    try:
        llm_ok = llm.health()
    except Exception as exc:  # noqa: BLE001
        return {"status": "error", "detail": f"LLM health check failed: {exc}"}
    llm_settings = getattr(llm, "cfg", None)
    return {
        "status": "ok" if llm_ok else "partial",
        "llm": llm_ok,
        "provider": getattr(llm_settings, "provider_name", cfg.get("llm", {}).get("provider_name")),
        "model": getattr(llm_settings, "model", cfg.get("llm", {}).get("model")),
        "token_configured": bool(os.getenv("STUDIO_API_TOKEN")),
    }


@app.get("/api/frameworks")
async def frameworks() -> dict[str, Any]:
    return {"frameworks": [{"key": k, "label": v} for k, v in FRAMEWORK_LABELS.items() if k != "sound_practice"]}


# --------------------------------------------------------------------------
# Projects
# --------------------------------------------------------------------------

@app.get("/api/projects")
async def list_projects() -> dict[str, Any]:
    return {"projects": [_summary(p) for p in _studio().store.list()]}


@app.post("/api/projects", status_code=202)
async def create_project(
    mode: str = Form(...),
    title: str = Form(...),
    brief: str = Form(...),
    frameworks: list[str] = Form(default=[]),
    max_rounds: int = Form(default=3),
    challenger: str = Form(default="auto"),
    data: list[UploadFile] | None = File(default=None),
    regulations: list[UploadFile] | None = File(default=None),
    concept_papers: list[UploadFile] | None = File(default=None),
    package: list[UploadFile] | None = File(default=None),
    x_studio_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_token(x_studio_token)
    try:
        inp = ProjectInput(
            mode=mode, title=title.strip()[:200], brief=brief.strip(),  # type: ignore[arg-type]
            frameworks=[f for f in frameworks if f in FRAMEWORK_LABELS],
            max_rounds=max_rounds, challenger=challenger,  # type: ignore[arg-type]
        )
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    if not inp.title or not inp.brief:
        raise HTTPException(422, "Title and brief are required")
    try:
        project = _studio().create_project(
            inp,
            data=await _read_uploads(data),
            regulations=await _read_uploads(regulations),
            concept_papers=await _read_uploads(concept_papers),
            package=await _read_uploads(package),
        )
    except InputError as exc:
        raise HTTPException(422, str(exc)) from exc
    _start(project.project_id)
    return {"project_id": project.project_id, "status": "queued"}


@app.get("/api/projects/{project_id}")
async def get_project(project_id: str) -> dict[str, Any]:
    p = _load(project_id)
    return {"project": json.loads(p.model_dump_json()), "summary": _summary(p), "running": project_id in _running}


@app.get("/api/projects/{project_id}/stream")
async def stream_project(project_id: str, after: int = 0):
    _load(project_id)

    async def events():
        sent = after
        while True:
            p = _studio().store.load(project_id)
            for ev in p.trace[sent:]:
                yield f"event: trace\ndata: {ev.model_dump_json()}\n\n"
            sent = len(p.trace)
            if p.status not in ("queued", "running"):
                yield f"event: done\ndata: {json.dumps(_summary(p))}\n\n"
                return
            yield ": keep-alive\n\n"
            await asyncio.sleep(1.5)

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


@app.get("/api/projects/{project_id}/tree")
async def project_tree(project_id: str) -> dict[str, Any]:
    _load(project_id)
    pdir = _studio().store.project_dir(project_id)
    files = []
    for path in sorted(pdir.rglob("*")):
        if path.is_file() and path.suffix.lower() in DOWNLOADABLE and "__pycache__" not in path.parts and not path.name.startswith("."):
            rel = path.relative_to(pdir).as_posix()
            if rel == "project.json":
                continue
            files.append({"path": rel, "bytes": path.stat().st_size})
    return {"files": files}


@app.get("/api/projects/{project_id}/files/{rel_path:path}")
async def project_file(project_id: str, rel_path: str):
    _load(project_id)
    try:
        target = resolve_inside(_studio().store.project_dir(project_id), rel_path)
    except ValueError:
        raise HTTPException(400, "Invalid path") from None
    if not target.is_file() or target.suffix.lower() not in DOWNLOADABLE:
        raise HTTPException(404, "File not found")
    media = "text/plain; charset=utf-8" if target.suffix.lower() in (".py", ".txt", ".md", ".ini", ".xml", ".csv") else None
    return FileResponse(target, media_type=media, filename=target.name if target.suffix.lower() in (".docx", ".pdf") else None)


class SignoffRequest(BaseModel):
    name: str = Field(min_length=2, max_length=120)
    role: str = Field(default="", max_length=120)
    decision: str = Field(default="accept", pattern="^(accept|reject)$")
    comment: str = Field(default="", max_length=2000)


@app.post("/api/projects/{project_id}/signoff")
async def signoff(project_id: str, payload: SignoffRequest, x_studio_token: str | None = Header(default=None)) -> dict[str, Any]:
    _require_token(x_studio_token)
    _load(project_id)
    try:
        p = await asyncio.to_thread(_studio().signoff, project_id, Signoff(**payload.model_dump()))
    except InputError as exc:
        raise HTTPException(409, str(exc)) from exc
    return {"summary": _summary(p), "signoffs": [s.model_dump() for s in p.signoffs]}


# --------------------------------------------------------------------------
# Library
# --------------------------------------------------------------------------

@app.get("/api/library")
async def library() -> dict[str, Any]:
    lib = _studio().library
    docs = lib.documents() if lib else []
    return {"documents": [
        {"doc_id": d.doc_id, "title": d.title, "kind": d.kind, "framework": d.framework,
         "chunks": d.chunks, "segments": d.segments, "builtin": d.builtin, "projects": len(d.projects)}
        for d in docs
    ]}


@app.post("/api/library", status_code=201)
async def add_to_library(
    kind: str = Form(...),
    files: list[UploadFile] = File(...),
    x_studio_token: str | None = Header(default=None),
) -> dict[str, Any]:
    _require_token(x_studio_token)
    if kind not in ("regulation", "concept_paper", "reference"):
        raise HTTPException(422, "kind must be regulation, concept_paper or reference")
    lib = _studio().library
    if lib is None:
        raise HTTPException(503, "Library unavailable")
    import tempfile

    added = []
    for name, content in await _read_uploads(files):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / Path(name).name
            path.write_bytes(content)
            try:
                doc = await asyncio.to_thread(lib.add_document, path, kind=kind)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
        added.append({"doc_id": doc.doc_id, "title": doc.title, "chunks": doc.chunks})
    return {"added": added}
