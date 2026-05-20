"""Ollama client with JSON-mode + retry.

One model, many roles: we switch behaviour via system prompts rather than
loading different models — a 7B Q4 fits in VRAM; two don't.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

import httpx


class ModelSelectionStrategy(Enum):
    COMPLEXITY_BASED = "complexity_based"
    RESOURCE_BASED = "resource_based"
    USER_SPECIFIED = "user_specified"


@dataclass
class ModelSpec:
    name: str
    priority: int = 1
    min_vram_gb: int = 4
    capabilities: list[str] | None = None  # e.g., ["math", "reasoning"]


@dataclass
class LLMConfig:
    model: str  # Backward compatibility: primary model
    host: str = "http://localhost:11434"
    num_ctx: int = 8192
    temperature: float = 0.2
    timeout_s: int = 180
    # New fields for multi-model support
    models: list[ModelSpec] | None = None
    selection_strategy: ModelSelectionStrategy = ModelSelectionStrategy.COMPLEXITY_BASED
    fallback_timeout_s: int = 60


class OllamaLLM:
    """Minimal Ollama client. Uses the /api/chat endpoint directly so we
    don't need the `ollama` package at runtime (though it's in requirements
    for convenience)."""

    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self._client = httpx.Client(timeout=cfg.timeout_s)
        # If no models list, create from single model for backward compatibility
        if not cfg.models:
            self.cfg.models = [ModelSpec(name=cfg.model, priority=1, min_vram_gb=4)]

    def _get_available_vram_gb(self) -> int:
        """Estimate available VRAM. Simple heuristic: assume 8GB for now."""
        # TODO: Implement proper VRAM detection using GPUtil or nvidia-ml-py
        return 8  # Placeholder

    def select_models(self, task_complexity: str = "medium") -> list[ModelSpec]:
        """Select primary and fallback models based on strategy."""
        available_vram = self._get_available_vram_gb()
        candidates = [m for m in self.cfg.models if m.min_vram_gb <= available_vram]
        if not candidates:
            # Fallback to any model if none fit
            candidates = self.cfg.models

        # Sort by priority (lower number = higher priority)
        candidates.sort(key=lambda m: m.priority)

        if self.cfg.selection_strategy == ModelSelectionStrategy.COMPLEXITY_BASED:
            # For complex tasks, prefer larger models; simple tasks, smaller
            if task_complexity == "complex":
                return candidates  # Try best first
            else:
                return list(reversed(candidates))  # Try smaller first
        elif self.cfg.selection_strategy == ModelSelectionStrategy.RESOURCE_BASED:
            return candidates  # Already sorted by priority assuming larger = higher priority
        else:  # USER_SPECIFIED or default
            return candidates

    def chat_with_fallback(
        self,
        messages: list[dict],
        task_complexity: str = "medium",
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        stop: list[str] | None = None,
    ) -> str:
        """Chat with automatic fallback to alternative models on failure."""
        model_chain = self.select_models(task_complexity)
        last_error = None

        for i, model_spec in enumerate(model_chain):
            original_model = self.cfg.model
            try:
                self.cfg.model = model_spec.name
                # First attempt uses full timeout; fallbacks use shorter timeout
                self._client.timeout = self.cfg.timeout_s if i == 0 else self.cfg.fallback_timeout_s

                result = self.chat(messages, temperature=temperature, json_mode=json_mode, stop=stop)
                return result
            except Exception as e:
                last_error = e
                print(f"Model {model_spec.name} failed: {e}. Trying next...")
                continue
            finally:
                self.cfg.model = original_model
                self._client.timeout = self.cfg.timeout_s

        raise RuntimeError(f"All models failed. Last error: {last_error}")

    def chat(
        self,
        messages: list[dict],
        *,
        temperature: float | None = None,
        json_mode: bool = False,
        stop: list[str] | None = None,
    ) -> str:
        payload: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.cfg.temperature if temperature is None else temperature,
                "num_ctx": self.cfg.num_ctx,
            },
        }
        if stop:
            payload["options"]["stop"] = stop
        if json_mode:
            payload["format"] = "json"

        r = self._client.post(f"{self.cfg.host}/api/chat", json=payload)
        r.raise_for_status()
        data = r.json()
        return data["message"]["content"]

    def chat_json(
        self,
        messages: list[dict],
        schema_hint: str = "",
        *,
        temperature: float = 0.0,
        max_retries: int = 2,
        task_complexity: str | None = None,
    ) -> dict:
        """Ask the model for JSON and parse it with retries on malformed output."""
        if schema_hint:
            messages = messages + [{
                "role": "system",
                "content": f"Respond with JSON only. Schema hint:\n{schema_hint}",
            }]
        if task_complexity is None:
            # Estimate from the last user message
            user_msg = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
            task_complexity = self.estimate_task_complexity(user_msg)

        last_err = None
        for attempt in range(max_retries + 1):
            try:
                txt = self.chat_with_fallback(messages, task_complexity, temperature=temperature, json_mode=True)
                return json.loads(txt)
            except (json.JSONDecodeError, httpx.HTTPError) as e:
                last_err = e
                time.sleep(0.5 * (attempt + 1))
        raise RuntimeError(f"Failed to get valid JSON after {max_retries+1} tries: {last_err}")

    def health(self) -> bool:
        """Return True if the Ollama service is reachable."""
        try:
            r = self._client.get(f"{self.cfg.host}/api/tags", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def estimate_task_complexity(self, task: str) -> str:
        """Simple heuristic to estimate task complexity."""
        words = len(task.split())
        has_complex_keywords = any(kw in task.lower() for kw in [
            "analyze", "optimize", "model", "predict", "strategy", "research", "complex"
        ])
        if words > 50 or has_complex_keywords:
            return "complex"
        elif words < 20:
            return "simple"
        else:
            return "medium"
