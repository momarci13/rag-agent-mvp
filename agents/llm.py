"""Direct OpenAI API client used by every agent role.

The application talks to OpenAI's hosted Chat Completions API. It never loads
or runs a local language model. Agent specialisation is expressed through
role prompts and optional per-role model routes (``role_models``).

DRAFT/SUPPORT NOTE: this client underlies the bank risk-validation agent team
as well as the pre-existing quant/research agents. Nothing here should be
read as making any agent's output authoritative -- see agents/risk_schemas.py
for the disclaimer baked into every validation report.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import httpx2
import openai


class ModelSelectionStrategy(Enum):
    """How configured model routes are ordered."""

    PRIORITY = "priority"
    COMPLEXITY_BASED = "complexity_based"
    USER_SPECIFIED = "user_specified"


@dataclass(frozen=True)
class ModelSpec:
    """A model route exposed by OpenAI (e.g. a reasoning vs. fast tier)."""

    name: str
    priority: int = 1
    capabilities: list[str] | None = None


@dataclass
class LLMConfig:
    """Configuration for :class:`HostedLLM`."""

    model: str = "gpt-5.1-mini"
    base_url: str | None = None
    api_key: str = ""
    provider_name: str = "OpenAI"
    require_api_key: bool = True
    temperature: float = 0.2
    timeout_s: int = 180
    max_output_tokens: int = 8192
    models: list[ModelSpec] | None = None
    selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.PRIORITY
    fallback_timeout_s: int = 90
    role_models: dict[str, str] = field(default_factory=dict)


class HostedLLM:
    """Small, synchronous OpenAI client shared by every agent role.

    Exposes the same ``chat``/``chat_json``/``chat_with_fallback`` shape the
    agents already depend on, keeping every role independent of the raw
    OpenAI SDK. ``transport`` accepts an ``httpx2.BaseTransport`` (the OpenAI
    SDK's own HTTP stack) purely to let tests inject a mock transport with no
    live network calls.
    """

    def __init__(
        self,
        cfg: LLMConfig,
        *,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        self.cfg = cfg
        self.base_url = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
        http_client = httpx2.Client(timeout=cfg.timeout_s, transport=transport)
        client_kwargs: dict[str, Any] = {
            "api_key": cfg.api_key or "sk-not-configured",
            "http_client": http_client,
            # Retries/fallback are handled explicitly by chat_with_fallback and
            # chat_json's bounded retry loop; disable the SDK's own automatic
            # retries so each attempt maps to exactly one HTTP request.
            "max_retries": 0,
        }
        if cfg.base_url:
            client_kwargs["base_url"] = cfg.base_url
        self._client = openai.OpenAI(**client_kwargs)
        if not cfg.models:
            cfg.models = [ModelSpec(name=cfg.model, priority=1)]

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HostedLLM":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def select_models(self, task_complexity: str = "medium") -> list[ModelSpec]:
        """Return the ordered model fallback chain.

        Priority is deterministic. For ``complexity_based`` routing, simple
        tasks prefer routes tagged ``fast`` and complex tasks prefer routes
        tagged ``reasoning``; configured priority resolves remaining ties.
        """

        candidates = list(self.cfg.models or [])
        candidates.sort(key=lambda item: item.priority)
        if self.cfg.selection_strategy != ModelSelectionStrategy.COMPLEXITY_BASED:
            return candidates

        preferred = "reasoning" if task_complexity == "complex" else "fast"
        return sorted(
            candidates,
            key=lambda item: (
                preferred not in (item.capabilities or []),
                item.priority,
            ),
        )

    def _model_for_role(self, role: str | None) -> str | None:
        if role is None:
            return None
        return self.cfg.role_models.get(role.lower())

    def chat_with_fallback(
        self,
        messages: list[dict[str, Any]],
        task_complexity: str = "medium",
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        response_schema: dict[str, Any] | None = None,
        stop: list[str] | None = None,
        role: str | None = None,
    ) -> str:
        """Call the preferred model and fail over through configured routes."""

        role_model = self._model_for_role(role)
        chain = [ModelSpec(role_model, priority=0)] if role_model else []
        chain.extend(m for m in self.select_models(task_complexity) if m.name != role_model)
        last_error: Exception | None = None

        for index, model_spec in enumerate(chain):
            try:
                timeout = self.cfg.timeout_s if index == 0 else self.cfg.fallback_timeout_s
                return self.chat(
                    messages,
                    temperature=temperature,
                    json_mode=json_mode,
                    response_schema=response_schema,
                    stop=stop,
                    model=model_spec.name,
                    timeout_s=timeout,
                )
            except (openai.OpenAIError, KeyError, ValueError) as exc:
                last_error = exc

        raise RuntimeError(
            f"All {self.cfg.provider_name} model routes failed. Last error: {last_error}"
        )

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        response_schema: dict[str, Any] | None = None,
        stop: list[str] | None = None,
        model: str | None = None,
        role: str | None = None,
        timeout_s: int | None = None,
    ) -> str:
        """Send one non-streaming chat-completions request.

        ``response_schema``, when given a JSON Schema dict, requests OpenAI's
        strict structured-outputs mode instead of the looser ``json_object``
        mode (which ``json_mode=True`` alone requests). Use
        :func:`make_strict_json_schema` to prepare a Pydantic-derived schema
        for strict mode first.
        """

        selected_model = model or self._model_for_role(role) or self.cfg.model
        request_kwargs: dict[str, Any] = {
            "model": selected_model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": self.cfg.max_output_tokens,
            "timeout": timeout_s or self.cfg.timeout_s,
        }
        if stop:
            request_kwargs["stop"] = stop
        if response_schema is not None:
            request_kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": response_schema, "strict": True},
            }
        elif json_mode:
            request_kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**request_kwargs)
        content = response.choices[0].message.content
        if not isinstance(content, str):
            raise ValueError(
                f"{self.cfg.provider_name} response did not contain text content"
            )
        return content

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
    ) -> dict[str, Any]:
        """Request and validate a JSON object with bounded retries.

        ``strict=True`` additionally requests OpenAI's strict structured
        outputs mode using ``schema_hint`` (expected to be
        ``json.dumps(SomeModel.model_json_schema())``) as the schema, cutting
        down on malformed-JSON retries for schemas without optional-field
        constraints the strict-mode transform can't express (e.g.
        ``Field(min_length=...)``, which Pydantic's own ``model_validate``
        below still enforces regardless of this flag). Falls back to loose
        ``json_object`` mode if ``schema_hint`` isn't valid JSON.
        """

        request_messages = list(messages)
        response_schema: dict[str, Any] | None = None
        if schema_hint:
            request_messages.append({
                "role": "system",
                "content": f"Respond with JSON only. Schema hint:\n{schema_hint}",
            })
            if strict:
                try:
                    response_schema = make_strict_json_schema(json.loads(schema_hint))
                except (json.JSONDecodeError, TypeError):
                    response_schema = None
        if task_complexity is None:
            user_text = next(
                (str(m.get("content", "")) for m in reversed(request_messages) if m.get("role") == "user"),
                "",
            )
            task_complexity = self.estimate_task_complexity(user_text)

        last_error: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                text = self.chat_with_fallback(
                    request_messages,
                    task_complexity,
                    temperature=temperature,
                    json_mode=True,
                    response_schema=response_schema,
                    role=role,
                )
                parsed = json.loads(_strip_json_fence(text))
                if not isinstance(parsed, dict):
                    raise ValueError("Expected a JSON object")
                return parsed
            except (json.JSONDecodeError, openai.OpenAIError, RuntimeError, ValueError) as exc:
                last_error = exc
                time.sleep(0.25 * (attempt + 1))
        raise RuntimeError(
            f"Failed to obtain valid JSON from {self.cfg.provider_name} after "
            f"{max_retries + 1} attempts: {last_error}"
        )

    def health(self) -> bool:
        """Return whether the OpenAI API is reachable and the key is authorized."""

        if self.cfg.require_api_key and not self.cfg.api_key:
            return False
        try:
            self._client.models.list()
            return True
        except openai.OpenAIError:
            return False

    @staticmethod
    def estimate_task_complexity(task: str) -> str:
        words = len(task.split())
        complex_terms = {
            "analyze", "optimize", "model", "predict", "strategy", "research",
            "portfolio", "regime", "backtest", "causal",
        }
        if words > 50 or any(term in task.lower() for term in complex_terms):
            return "complex"
        if words < 10:
            return "simple"
        return "medium"


def make_strict_json_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Rewrite a JSON Schema (as produced by Pydantic's ``model_json_schema()``)
    to satisfy OpenAI's strict structured-outputs mode: every object must set
    ``additionalProperties: false`` and list every property as required
    (Pydantic already expresses optional fields as nullable ``anyOf`` unions,
    which strict mode accepts as long as they're also marked required).
    Provider-neutral -- operates on any JSON Schema dict, not just risk
    schemas, so this stays in the shared LLM client rather than a domain module.
    """
    schema = dict(schema)
    defs = schema.get("$defs")
    if isinstance(defs, dict):
        schema["$defs"] = {name: make_strict_json_schema(sub) for name, sub in defs.items()}
    if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
        schema["properties"] = {
            key: make_strict_json_schema(value) if isinstance(value, dict) else value
            for key, value in schema["properties"].items()
        }
        schema["additionalProperties"] = False
        schema["required"] = list(schema["properties"].keys())
    if isinstance(schema.get("items"), dict):
        schema["items"] = make_strict_json_schema(schema["items"])
    for key in ("anyOf", "allOf", "oneOf"):
        if isinstance(schema.get(key), list):
            schema[key] = [
                make_strict_json_schema(item) if isinstance(item, dict) else item
                for item in schema[key]
            ]
    return schema


def _strip_json_fence(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        first_newline = stripped.find("\n")
        if first_newline >= 0:
            stripped = stripped[first_newline + 1 : -3].strip()
    return stripped
