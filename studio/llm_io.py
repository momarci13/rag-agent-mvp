"""Structured LLM calls: every response is validated against a Pydantic model."""
from __future__ import annotations

import json
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class JsonLLM(Protocol):
    def chat_json(
        self,
        messages: list[dict[str, Any]],
        schema_hint: str = "",
        *,
        temperature: float = 0.0,
        max_retries: int = 2,
        task_complexity: str | None = None,
        role: str | None = None,
        strict: bool = False,
    ) -> dict[str, Any]: ...


class AgentOutputError(RuntimeError):
    """The model did not return output matching the expected schema."""


def clip(text: str, limit: int) -> str:
    text = text or ""
    if len(text) <= limit:
        return text
    head = int(limit * 0.7)
    return text[:head] + f"\n...[{len(text) - limit} characters omitted]...\n" + text[-(limit - head):]


def to_json(obj: Any, limit: int = 20000) -> str:
    if isinstance(obj, BaseModel):
        text = obj.model_dump_json(indent=1, exclude_none=True)
    else:
        text = json.dumps(obj, indent=1, default=str)
    return clip(text, limit)


def ask(
    llm: JsonLLM,
    *,
    role: str,
    system: str,
    user: str,
    model: type[T],
    attempts: int = 2,
    temperature: float = 0.1,
) -> T:
    """Call the LLM and parse into ``model``; on schema errors, retry once with
    the validation error so the model can correct itself."""
    schema = json.dumps(model.model_json_schema())
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    last_error = ""
    for _ in range(attempts):
        raw = llm.chat_json(messages, schema, temperature=temperature, role=role, task_complexity="complex")
        try:
            return model.model_validate(raw)
        except ValidationError as exc:
            last_error = str(exc)[:3000]
            messages = messages + [
                {"role": "assistant", "content": json.dumps(raw)[:20000]},
                {"role": "user", "content": f"That JSON does not match the schema:\n{last_error}\nReturn corrected JSON only."},
            ]
    raise AgentOutputError(f"{role}: output did not match {model.__name__}: {last_error}")
