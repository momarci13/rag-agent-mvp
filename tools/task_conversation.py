"""Task conversation processing and refinement logic.

Handles:
- User message routing (Q&A, refinement, iteration)
- Artifact re-execution with code edits
- Task branching and variation creation
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any, Optional

from agents.graph import RunState, Message
from agents.llm import OllamaLLM
from agents.problem_decoder import decode_problem
from rag.hybrid import LiteHybridRAG
from tools.sandbox import run_code_sync
from tools import task_storage

logger = logging.getLogger(__name__)


def process_user_message(
    task_id: str,
    message_content: str,
    llm: OllamaLLM,
    rag: LiteHybridRAG,
    iteration: int = 0,
) -> tuple[str, list[dict], list[dict]]:
    """Process a user message and generate assistant response.
    
    Args:
        task_id: ID of the task to process
        message_content: User's message text
        llm: Language model instance
        rag: RAG system instance
        iteration: Which iteration this belongs to
    
    Returns:
        (assistant_response_text, new_artifacts, discovered_sources)
    """
    # Load task
    task = task_storage.load_task(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")
    
    # Add user message to task
    user_msg = Message(
        role="user",
        content=message_content,
        timestamp=datetime.utcnow(),
        iteration=iteration,
    )
    task.messages.append(user_msg)

    # Compress conversation memory if thread is long
    memory_context = ""
    try:
        from tools.memory import build_context_with_memory
        memory_context, _ = build_context_with_memory(task, llm)
    except Exception as exc:
        logger.debug("[CONV] Memory compression skipped: %s", exc)

    # Retrieve relevant context from RAG (with LLM query expansion)
    docs = rag.retrieve(message_content, k=3, llm=llm)

    # Auto-expand KB if retrieval quality is low
    try:
        from tools.kb_expansion import expand_kb_on_weak_retrieval
        docs, _expanded = expand_kb_on_weak_retrieval(
            message_content, docs, rag, llm, task_id,
        )
    except Exception as exc:
        logger.debug("[CONV] KB expansion skipped: %s", exc)

    # Inject memory context as a leading doc
    if memory_context:
        docs = [{"id": "memory_context", "text": memory_context,
                 "meta": {"kind": "memory", "source": "conversation"}}] + docs

    # Determine response type based on message content
    response_type = _classify_message(message_content, llm=llm)

    # Generate response
    try:
        if response_type == "refinement":
            response, artifacts, discovered_sources = _handle_refinement(task, task_id, message_content, docs, llm, rag)
        elif response_type == "question":
            response, artifacts, discovered_sources = _handle_question(task, message_content, docs, llm)
        else:  # new_iteration
            response, artifacts, discovered_sources = _handle_new_iteration(task, task_id, message_content, docs, llm, rag)
    except Exception as e:
        response = (
            f"I encountered an error processing your request: {e}\n\n"
            "Please try rephrasing, or start a new task."
        )
        artifacts = []
        discovered_sources = []
    
    # Add assistant response
    assistant_msg = Message(
        role="assistant",
        content=response,
        timestamp=datetime.utcnow(),
        iteration=iteration,
    )
    task.messages.append(assistant_msg)
    
    # Update task with new artifacts
    if artifacts:
        task.artifacts.extend(artifacts)
    
    # Save updated task
    task_storage.save_task(task)

    return response, artifacts, discovered_sources


def re_execute_artifact(
    task_id: str,
    artifact_id: str,
    edited_code: str,
    artifact_index: int = -1,
) -> dict:
    """Re-execute an artifact with edited code.
    
    Args:
        task_id: Task ID
        artifact_id: Artifact ID (for reference)
        edited_code: Modified code to execute
        artifact_index: Index in artifacts list (-1 for last)
    
    Returns:
        {
            "status": "success" | "error",
            "stdout": str,
            "stderr": str,
            "returncode": int,
            "execution_time": float,
        }
    """
    task = task_storage.load_task(task_id)
    if not task:
        raise ValueError(f"Task {task_id} not found")
    
    if artifact_index < 0:
        artifact_index = len(task.artifacts) + artifact_index
    
    if artifact_index < 0 or artifact_index >= len(task.artifacts):
        raise ValueError(f"Artifact index {artifact_index} out of range")
    
    artifact = task.artifacts[artifact_index]
    
    # Execute code
    result = run_code_sync(
        code=edited_code,
        timeout_s=30,
        mem_mb=512,
    )
    
    # Create new artifact version with results
    new_artifact = {
        "id": f"{artifact_id}_v{len(task.artifacts)}",
        "type": artifact.get("type", "ds"),
        "payload": result,
        "raw": edited_code,
        "iteration": task.iterations,
        "edited": True,
        "edited_at": datetime.utcnow().isoformat(),
    }
    
    # Append to artifacts
    task.artifacts.append(new_artifact)
    
    # Add message to thread
    msg = Message(
        role="system",
        content=f"User edited and re-ran artifact {artifact_id}. Return code: {result['returncode']}",
        timestamp=datetime.utcnow(),
        iteration=task.iterations,
    )
    task.messages.append(msg)
    
    # Save
    task_storage.save_task(task)
    
    return result


def branch_task(
    task_id: str,
    branch_name: Optional[str] = None,
    from_iteration: int = 0,
) -> str:
    """Branch a task to create a variation.
    
    Args:
        task_id: Original task ID
        branch_name: Name for the branch
        from_iteration: Which iteration to branch from (0 = start fresh)
    
    Returns:
        new_task_id
    """
    new_task_id = task_storage.branch_task(
        task_id,
        branch_name=branch_name,
        from_iteration=from_iteration,
    )
    
    # Log branching action
    task = task_storage.load_task(new_task_id)
    if task:
        msg = Message(
            role="system",
            content=f"Branch created from task {task_id}" + (f" (branch: {branch_name})" if branch_name else ""),
            timestamp=datetime.utcnow(),
            iteration=0,
        )
        task.messages.append(msg)
        task_storage.save_task(task)
    
    return new_task_id


def _classify_message(content: str, llm: OllamaLLM | None = None) -> str:
    """Classify user message type.

    Returns: "question" | "refinement" | "new_iteration"
    """
    if llm is not None:
        try:
            msgs = [
                {
                    "role": "system",
                    "content": (
                        "Classify the following user message as exactly one of: "
                        "question, refinement, new_iteration. "
                        "Reply with the single word only, no punctuation."
                    ),
                },
                {"role": "user", "content": content[:500]},
            ]
            result = llm.chat(msgs, temperature=0.0).strip().lower()
            if result in ("question", "refinement", "new_iteration"):
                return result
        except Exception:
            pass

    # Keyword fallback
    content_lower = content.lower()
    iteration_keywords = [
        "run again", "execute again", "retry", "restart",
        "new run", "from scratch", "start over",
    ]
    refinement_keywords = [
        "refine", "improve", "adjust", "modify", "change",
        "fix", "update", "edit", "tweak", "try again",
        "better", "more", "less", "use ema", "use sma",
    ]
    question_keywords = [
        "what", "how", "why", "where", "when",
        "can you", "could you", "explain", "understand",
        "tell me", "show me", "interpret",
    ]

    if any(kw in content_lower for kw in iteration_keywords):
        return "new_iteration"
    elif any(kw in content_lower for kw in refinement_keywords):
        return "refinement"
    elif any(kw in content_lower for kw in question_keywords):
        return "question"
    return "question"


_FOLLOW_UP_PROMPT = (
    "\n\n---\n"
    "*Would you like me to generate a PDF report, execute the code, or make further modifications?*"
)


def _handle_question(
    task: RunState,
    message: str,
    docs: list[dict],
    llm: OllamaLLM,
) -> tuple[str, list[dict], list[dict]]:
    """Handle a clarification/information question. Returns (response_text, [], [])."""
    from agents import roles

    prompt = (
        f"Original task: {task.task}\n"
        f"Task type: {task.task_type}\n"
        f"Iteration: {task.iterations}\n\n"
        f"User question: {message}"
    )
    msgs = roles.build_messages("DS", prompt, context_docs=docs)
    response = llm.chat(msgs) + _FOLLOW_UP_PROMPT
    return response, [], []


def _handle_refinement(
    task: RunState,
    task_id: str,
    message: str,
    docs: list[dict],
    llm: OllamaLLM,
    rag: LiteHybridRAG,
) -> tuple[str, list[dict], list[dict]]:
    """Handle a refinement request — regenerates the artifact with user feedback.

    Returns (response_text, [new_artifact], discovered_sources).
    """
    from agents import roles
    from tools import source_search

    discovered_sources: list[dict] = []
    try:
        results = source_search.multi_source_search(task.task, n_papers=3, llm=llm)
        discovered_sources = [r.to_dict() for r in results]
        hop0 = [r for r in results if r.hop == 0]
        if hop0:
            rag.ingest_papers(hop0)
        source_search.save_pending_sources(task_id, results)
        ctx_parts = [f"# {r.title}\n{r.abstract}" for r in hop0[:3] if r.abstract]
        if ctx_parts:
            docs = [{"id": "source_context", "text": "\n\n".join(ctx_parts),
                     "meta": {"kind": "scholar", "source": "multi_source"}}] + docs
    except Exception:
        pass

    last_artifact = task.artifacts[-1] if task.artifacts else None
    feedback = f"Refinement request: {message}"
    if last_artifact and last_artifact.get("raw"):
        feedback += f"\n\nPrevious code:\n{last_artifact['raw'][:2000]}"

    if task.task_type == "trading_research":
        spec = roles.design_strategy(llm, task.task, docs)
        new_artifact = {
            "type": "quant",
            "payload": {"spec": spec.model_dump(), "backtest": None},
            "iteration": task.iterations,
            "refined": True,
        }
        response = (
            f"Refined strategy generated: **{spec.name}**.\n\n"
            f"Signal: `{spec.signal_code}`"
            + _FOLLOW_UP_PROMPT
        )
    else:
        code_md = roles.analyze(llm, task.task, docs, feedback=feedback, decoding=task.decoding)
        new_artifact = {
            "type": "ds",
            "payload": {"code": code_md, "stdout": "", "stderr": "", "returncode": None},
            "raw": code_md,
            "iteration": task.iterations,
            "refined": True,
        }
        code_preview = "\n".join(code_md.splitlines()[:60])
        response = (
            f"Here is the refined code:\n\n```python\n{code_preview}\n```\n\n"
            f"The analysis has been updated based on your feedback."
            + _FOLLOW_UP_PROMPT
        )

    return response, [new_artifact], discovered_sources


def _handle_new_iteration(
    task: RunState,
    task_id: str,
    message: str,
    docs: list[dict],
    llm: OllamaLLM,
    rag: LiteHybridRAG,
) -> tuple[str, list[dict], list[dict]]:
    """Handle request to start a new iteration — runs multi-source search then re-executes.

    Returns (response_text, [new_artifact], discovered_sources).
    """
    from agents import roles
    from tools import source_search

    discovered_sources: list[dict] = []
    try:
        results = source_search.multi_source_search(task.task, n_papers=3, llm=llm)
        discovered_sources = [r.to_dict() for r in results]
        hop0 = [r for r in results if r.hop == 0]
        if hop0:
            rag.ingest_papers(hop0)
        source_search.save_pending_sources(task_id, results)
        ctx_parts = [f"# {r.title}\n{r.abstract}" for r in hop0[:3] if r.abstract]
        if ctx_parts:
            docs.insert(0, {
                "id": "source_context",
                "text": "\n\n".join(ctx_parts),
                "meta": {"kind": "scholar", "source": "multi_source"},
            })
    except Exception:
        pass

    feedback = (
        f"New iteration request: {message}\n"
        f"Previous iterations completed: {task.iterations}"
    )
    code_md = roles.analyze(llm, task.task, docs, feedback=feedback, decoding=task.decoding)

    new_artifact = {
        "type": "ds",
        "payload": {"code": code_md, "stdout": "", "stderr": "", "returncode": None},
        "raw": code_md,
        "iteration": task.iterations + 1,
    }
    code_preview = "\n".join(code_md.splitlines()[:60])
    response = (
        f"New iteration started. Here is the updated code:\n\n```python\n{code_preview}\n```\n\n"
        f"This incorporates your feedback and any newly retrieved research."
        + _FOLLOW_UP_PROMPT
    )
    return response, [new_artifact], discovered_sources
