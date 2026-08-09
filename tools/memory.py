"""Conversation memory compression for long research sessions.

Compresses messages older than a rolling window into a compact LLM-generated
summary so that long task threads don't blow the 8192-token context limit.
The summary is persisted per-task and injected as a RAG context doc.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.llm import HostedLLM

logger = logging.getLogger(__name__)

TASKS_DIR = Path(__file__).parent.parent / "output" / "tasks"

COMPRESSION_THRESHOLD = 20  # compress when message count exceeds this
KEEP_LAST = 8                # always retain the N most-recent messages


@dataclass
class MemorySummary:
    task_id: str
    compressed_at: str
    message_count_before: int
    summary: str
    key_findings: list[str] = field(default_factory=list)
    open_questions: list[str] = field(default_factory=list)
    code_present: bool = False


def save_memory_summary(task_id: str, summary: MemorySummary) -> None:
    path = TASKS_DIR / task_id / "memory_summary.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(summary), indent=2), encoding="utf-8")


def load_memory_summary(task_id: str) -> MemorySummary | None:
    path = TASKS_DIR / task_id / "memory_summary.json"
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return MemorySummary(**{k: v for k, v in data.items() if k in MemorySummary.__dataclass_fields__})
    except Exception:
        return None


def _format_for_compression(messages: list[Any], task_desc: str) -> str:
    lines = [f"Research task: {task_desc[:200]}\n\nConversation history:"]
    for msg in messages:
        role = getattr(msg, "role", "?")
        content = getattr(msg, "content", "")
        if role == "system":
            continue
        lines.append(f"[{role.upper()}]: {content[:600]}")
    return "\n".join(lines)


def compress_old_messages(
    messages: list[Any],
    task_desc: str,
    task_id: str,
    llm: "HostedLLM",
    keep_last: int = KEEP_LAST,
) -> MemorySummary:
    """Compress messages[:-keep_last] into a MemorySummary via LLM JSON call."""
    to_compress = messages[:-keep_last] if len(messages) > keep_last else messages
    formatted = _format_for_compression(to_compress, task_desc)

    schema_hint = '{"summary": "string", "key_findings": ["string"], "open_questions": ["string"], "code_present": false}'
    prompt_msgs = [
        {
            "role": "system",
            "content": (
                "You are a research assistant. Summarize the conversation into compact JSON. "
                "Keys: summary (≤300 words), key_findings (list, each ≤20 words), "
                "open_questions (list, each ≤15 words), code_present (bool)."
            ),
        },
        {"role": "user", "content": formatted[:3000]},
    ]

    summary_text = ""
    key_findings: list[str] = []
    open_questions: list[str] = []
    code_present = False

    try:
        result = llm.chat_json(prompt_msgs, schema_hint=schema_hint, temperature=0.0)
        summary_text = str(result.get("summary", ""))[:1200]
        key_findings = [str(f) for f in (result.get("key_findings") or [])[:8]]
        open_questions = [str(q) for q in (result.get("open_questions") or [])[:5]]
        code_present = bool(result.get("code_present", False))
    except Exception as exc:
        logger.warning("[MEMORY] LLM compression failed (%s) — using text fallback", exc)
        summary_text = formatted[:800]
        code_present = any("```" in getattr(m, "content", "") for m in to_compress)

    sm = MemorySummary(
        task_id=task_id,
        compressed_at=datetime.utcnow().isoformat(),
        message_count_before=len(to_compress),
        summary=summary_text,
        key_findings=key_findings,
        open_questions=open_questions,
        code_present=code_present,
    )
    save_memory_summary(task_id, sm)
    logger.info("[MEMORY] Compressed %d messages → summary for task %s", len(to_compress), task_id[:8])
    return sm


def _render_summary_as_context(sm: MemorySummary) -> str:
    parts = [
        f"## Research Memory (summarising {sm.message_count_before} earlier messages)\n\n{sm.summary}"
    ]
    if sm.key_findings:
        parts.append("\n**Key findings so far:**\n" + "\n".join(f"- {f}" for f in sm.key_findings))
    if sm.open_questions:
        parts.append("\n**Open questions:**\n" + "\n".join(f"- {q}" for q in sm.open_questions))
    return "\n".join(parts)


def build_context_with_memory(
    task: Any,
    llm: "HostedLLM",
    max_messages: int = COMPRESSION_THRESHOLD,
) -> tuple[str, list]:
    """Return (memory_context_text, recent_messages).

    If the task has more than max_messages messages:
    - Compress the older portion via LLM (or reload a cached compression)
    - Return the summary as a context string + the last KEEP_LAST messages

    If under the threshold: return ("", all messages).
    The caller should inject the context string as a doc into the RAG doc list.
    """
    messages = getattr(task, "messages", [])
    if len(messages) <= max_messages:
        return "", messages

    # Reuse existing summary if it already covers at least this batch
    existing = load_memory_summary(task.task_id)
    if existing and existing.message_count_before >= len(messages) - KEEP_LAST - 2:
        return _render_summary_as_context(existing), messages[-KEEP_LAST:]

    sm = compress_old_messages(messages, task.task, task.task_id, llm)
    return _render_summary_as_context(sm), messages[-KEEP_LAST:]
