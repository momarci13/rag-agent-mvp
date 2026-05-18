"""Task storage layer with persistence, search, and migration support.

Handles task lifecycle: save, load, list, search, and migration from legacy format.
"""
from __future__ import annotations

import json
import shutil
from dataclasses import asdict, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from agents.graph import RunState, Message


# Configuration
TASKS_DIR = Path(__file__).parent.parent / "output" / "tasks"
LEGACY_RUNS_DIR = Path(__file__).parent.parent / "output" / "runs"
INDEX_FILE = TASKS_DIR / "index.json"


def ensure_dirs() -> None:
    """Create necessary directories."""
    TASKS_DIR.mkdir(parents=True, exist_ok=True)


def _serialize_for_json(obj: Any) -> Any:
    """Convert objects that aren't JSON-serializable (datetime, Pydantic models, etc.)."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    elif isinstance(obj, Message):
        return {
            "role": obj.role,
            "content": obj.content,
            "timestamp": obj.timestamp.isoformat(),
            "iteration": obj.iteration,
        }
    elif isinstance(obj, (set, frozenset)):
        return list(obj)
    elif hasattr(obj, "model_dump"):
        # Pydantic v2 BaseModel
        return _serialize_for_json(obj.model_dump())
    elif hasattr(obj, "__dataclass_fields__"):
        return asdict(obj, dict_factory=lambda x: {
            k: _serialize_for_json(v) for k, v in x
        })
    elif isinstance(obj, dict):
        return {k: _serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_serialize_for_json(v) for v in obj]
    return obj


def _deserialize_message(data: dict) -> Message:
    """Reconstruct Message from dict."""
    return Message(
        role=data["role"],
        content=data["content"],
        timestamp=datetime.fromisoformat(data.get("timestamp", datetime.utcnow().isoformat())),
        iteration=data.get("iteration", 0),
    )


def _deserialize_runstate(data: dict) -> RunState:
    """Reconstruct RunState from dict, handling all fields including new ones."""
    # Convert messages back to Message objects
    messages = []
    if "messages" in data:
        for msg_data in data["messages"]:
            if isinstance(msg_data, dict):
                messages.append(_deserialize_message(msg_data))
            elif isinstance(msg_data, Message):
                messages.append(msg_data)
    
    # Convert datetime strings back to datetime objects
    created_at = data.get("created_at")
    if isinstance(created_at, str):
        created_at = datetime.fromisoformat(created_at)
    elif created_at is None:
        created_at = datetime.utcnow()
    
    updated_at = data.get("updated_at")
    if isinstance(updated_at, str):
        updated_at = datetime.fromisoformat(updated_at)
    elif updated_at is None:
        updated_at = datetime.utcnow()
    
    # Build RunState with all fields
    return RunState(
        task=data.get("task", ""),
        task_type=data.get("task_type", ""),
        subtasks=data.get("subtasks", []),
        artifacts=data.get("artifacts", []),
        critiques=data.get("critiques", []),
        iterations=data.get("iterations", 0),
        accepted=data.get("accepted", False),
        decoding=data.get("decoding"),
        task_id=data.get("task_id", ""),
        messages=messages,
        parent_run_id=data.get("parent_run_id"),
        tags=data.get("tags", []),
        created_at=created_at,
        updated_at=updated_at,
        branch_name=data.get("branch_name"),
        template_data=data.get("template_data"),
    )


def save_task(state: RunState, parent_id: Optional[str] = None) -> str:
    """Save task state to disk.
    
    Returns task_id.
    """
    ensure_dirs()
    
    # If parent_id provided, set it on state
    if parent_id:
        state.parent_run_id = parent_id
    
    # Update timestamp
    state.updated_at = datetime.utcnow()
    
    task_dir = TASKS_DIR / state.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metadata
    metadata = {
        "task_id": state.task_id,
        "title": state.task[:60] + "..." if len(state.task) > 60 else state.task,
        "status": "completed" if state.accepted else ("in-progress" if state.iterations > 0 else "planned"),
        "created_at": state.created_at.isoformat(),
        "updated_at": state.updated_at.isoformat(),
        "tags": state.tags,
        "parent_id": state.parent_run_id,
        "branch_name": state.branch_name,
    }
    (task_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    
    # Save full state snapshot
    state_data = _serialize_for_json(state)
    (task_dir / "state.json").write_text(json.dumps(state_data, indent=2), encoding="utf-8")
    
    # Rewrite conversation log from all messages to avoid duplicates
    conv_file = task_dir / "conversations.jsonl"
    with conv_file.open("w", encoding="utf-8") as fh:
        for msg in state.messages:
            msg_data = _serialize_for_json(msg)
            fh.write(json.dumps(msg_data) + "\n")
    
    # Create/update artifacts directory
    artifacts_dir = task_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    
    # Update global index
    _update_index(state)
    
    return state.task_id


def load_task(task_id: str) -> Optional[RunState]:
    """Load task state from disk.
    
    Returns RunState or None if not found.
    """
    task_dir = TASKS_DIR / task_id
    state_file = task_dir / "state.json"
    
    if not state_file.exists():
        return None
    
    data = json.loads(state_file.read_text(encoding="utf-8"))
    return _deserialize_runstate(data)


def list_tasks(
    limit: int = 50,
    offset: int = 0,
    sort_by: str = "updated_at",  # "created_at", "updated_at", "title"
    filters: Optional[dict] = None,
) -> tuple[list[dict], int]:
    """List tasks with pagination and filtering.
    
    Returns (tasks, total_count).
    """
    ensure_dirs()
    _ensure_index()
    
    index_data = json.loads(INDEX_FILE.read_text(encoding="utf-8")) if INDEX_FILE.exists() else {"tasks": []}
    tasks = index_data.get("tasks", [])
    
    # Apply filters
    if filters:
        if "status" in filters:
            tasks = [t for t in tasks if t.get("status") == filters["status"]]
        if "tags" in filters:
            filter_tags = set(filters["tags"]) if isinstance(filters["tags"], list) else {filters["tags"]}
            tasks = [t for t in tasks if any(tag in filter_tags for tag in t.get("tags", []))]
        if "parent_id" in filters:
            tasks = [t for t in tasks if t.get("parent_id") == filters["parent_id"]]
    
    # Sort
    reverse = sort_by.startswith("-")
    sort_key = sort_by.lstrip("-")
    try:
        tasks = sorted(tasks, key=lambda t: t.get(sort_key, ""), reverse=reverse)
    except Exception:
        pass
    
    total = len(tasks)
    tasks = tasks[offset : offset + limit]
    
    return tasks, total


def search_tasks(query: str, limit: int = 20) -> list[dict]:
    """Search tasks by keyword.
    
    Returns list of matching task metadata with relevance scores.
    """
    ensure_dirs()
    _ensure_index()
    
    index_data = json.loads(INDEX_FILE.read_text(encoding="utf-8")) if INDEX_FILE.exists() else {"tasks": []}
    tasks = index_data.get("tasks", [])
    
    query_lower = query.lower()
    results = []
    
    for task in tasks:
        score = 0
        title = task.get("title", "").lower()
        tags = [t.lower() for t in task.get("tags", [])]
        
        # Exact match in title
        if query_lower in title:
            score += 10
        # Partial matches
        for word in query_lower.split():
            if word in title:
                score += 5
            if any(word in tag for tag in tags):
                score += 3
        
        if score > 0:
            results.append((task, score))
    
    # Sort by relevance (descending)
    results = sorted(results, key=lambda x: x[1], reverse=True)[:limit]
    return [task for task, _ in results]


def branch_task(task_id: str, branch_name: Optional[str] = None, from_iteration: int = 0) -> str:
    """Create a branched copy of a task.
    
    Returns new task_id.
    """
    original = load_task(task_id)
    if not original:
        raise ValueError(f"Task {task_id} not found")
    
    # Create new task with same content up to specified iteration.
    # Filter artifacts by their stored iteration index, not by positional slice,
    # since multiple artifacts may be created per iteration.
    if from_iteration > 0:
        branch_artifacts = [
            a for a in original.artifacts
            if a.get("iteration", 0) < from_iteration
        ]
        branch_critiques = original.critiques[:from_iteration]
    else:
        branch_artifacts = []
        branch_critiques = []

    new_task = RunState(
        task=original.task,
        task_type=original.task_type,
        subtasks=original.subtasks,
        artifacts=branch_artifacts,
        critiques=branch_critiques,
        iterations=from_iteration,
        accepted=False,
        decoding=original.decoding,
        parent_run_id=task_id,
        tags=original.tags.copy(),
        branch_name=branch_name,
    )
    
    return save_task(new_task)


def export_template(task_id: str) -> dict:
    """Export task as a reusable template.
    
    Returns template data (excluding execution results).
    """
    task = load_task(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")
    
    template = {
        "task_description": task.task,
        "task_type": task.task_type,
        "decoding": task.decoding,
        "tags": task.tags,
        "subtasks": task.subtasks,
        "created_from_task": task_id,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    return template


def _update_index(state: RunState) -> None:
    """Update the task index file."""
    ensure_dirs()
    _ensure_index()
    
    index_data = json.loads(INDEX_FILE.read_text(encoding="utf-8")) if INDEX_FILE.exists() else {"tasks": []}
    tasks = index_data.get("tasks", [])
    
    # Find and update or append
    task_index = next((i for i, t in enumerate(tasks) if t["task_id"] == state.task_id), None)
    
    task_entry = {
        "task_id": state.task_id,
        "title": state.task[:60] + "..." if len(state.task) > 60 else state.task,
        "status": "completed" if state.accepted else ("in-progress" if state.iterations > 0 else "planned"),
        "created_at": state.created_at.isoformat(),
        "updated_at": state.updated_at.isoformat(),
        "tags": state.tags,
        "parent_id": state.parent_run_id,
        "branch_name": state.branch_name,
    }
    
    if task_index is not None:
        tasks[task_index] = task_entry
    else:
        tasks.append(task_entry)
    
    index_data["tasks"] = tasks
    INDEX_FILE.write_text(json.dumps(index_data, indent=2), encoding="utf-8")


def _ensure_index() -> None:
    """Ensure index file exists."""
    if not INDEX_FILE.exists():
        INDEX_FILE.write_text(json.dumps({"tasks": []}, indent=2), encoding="utf-8")


def migrate_legacy_runs() -> dict:
    """Migrate from legacy output/runs/run_*.json format to new tasks structure.
    
    Returns migration summary.
    """
    ensure_dirs()
    
    summary = {"migrated": 0, "skipped": 0, "failed": 0, "errors": []}
    
    if not LEGACY_RUNS_DIR.exists():
        return summary
    
    for run_file in sorted(LEGACY_RUNS_DIR.glob("run_*.json")):
        try:
            data = json.loads(run_file.read_text(encoding="utf-8"))
            
            # Deserialize to RunState (handles old format automatically)
            state = _deserialize_runstate(data)
            
            # If task_id not set, use a UUID
            if not state.task_id or state.task_id == "":
                from uuid import uuid4
                state.task_id = str(uuid4())
            
            # Save in new format
            save_task(state)
            summary["migrated"] += 1
            
        except Exception as e:
            summary["failed"] += 1
            summary["errors"].append(f"{run_file.name}: {str(e)}")
    
    return summary
