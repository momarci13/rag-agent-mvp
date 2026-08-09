"""Embedding backends for retrieval.

Production inference is routed through a hosted OpenAI-compatible provider.
The deterministic hash backend exists only for unit tests and offline
structural validation.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Protocol, Sequence

import httpx
import numpy as np


class EmbeddingClient(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray: ...


class HostedEmbeddings:
    """OpenAI-compatible client for hosted embedding inference."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: int = 120,
        batch_size: int = 64,
        provider_name: str = "OpenRouter",
        require_api_key: bool = True,
        health_path: str = "/key",
        extra_headers: dict[str, str] | None = None,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url
            or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        ).rstrip("/")
        self.api_key = (
            api_key if api_key is not None else os.getenv("OPENROUTER_API_KEY", "")
        )
        self.timeout_s = timeout_s
        self.batch_size = max(1, batch_size)
        self.provider_name = provider_name
        self.require_api_key = require_api_key
        self.health_path = health_path
        self.extra_headers = dict(extra_headers or {})
        self._client = httpx.Client(timeout=timeout_s, transport=transport)

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self.extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        del show_progress_bar
        values = list(texts)
        if not values:
            return np.empty((0, 0), dtype=np.float32)

        rows: list[list[float]] = []
        for start in range(0, len(values), self.batch_size):
            batch = values[start : start + self.batch_size]
            response = self._client.post(
                f"{self.base_url}/embeddings",
                headers=self._headers(),
                json={"model": self.model, "input": batch, "encoding_format": "float"},
            )
            response.raise_for_status()
            payload = response.json()
            data = sorted(payload["data"], key=lambda item: int(item.get("index", 0)))
            if len(data) != len(batch):
                raise ValueError(
                    f"{self.provider_name} returned an unexpected embedding count"
                )
            rows.extend(item["embedding"] for item in data)

        result = np.asarray(rows, dtype=np.float32)
        if result.ndim != 2:
            raise ValueError(f"{self.provider_name} returned malformed embeddings")
        if normalize_embeddings:
            norms = np.linalg.norm(result, axis=1, keepdims=True)
            result = result / np.maximum(norms, 1e-12)
        return result

    def health(self) -> bool:
        if self.require_api_key and not self.api_key:
            return False
        try:
            response = self._client.get(
                f"{self.base_url}/{self.health_path.lstrip('/')}",
                headers=self._headers(),
                timeout=min(self.timeout_s, 8),
            )
            return response.status_code == 200
        except httpx.HTTPError:
            return False

    def close(self) -> None:
        self._client.close()


class DeterministicHashEmbeddings:
    """Dependency-free deterministic vectors for tests, never production."""

    def __init__(self, dimensions: int = 128) -> None:
        self.dimensions = dimensions

    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray:
        del show_progress_bar
        matrix = np.zeros((len(texts), self.dimensions), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"[A-Za-z0-9_]+", text.lower()):
                digest = hashlib.sha256(token.encode("utf-8")).digest()
                column = int.from_bytes(digest[:4], "big") % self.dimensions
                sign = 1.0 if digest[4] & 1 else -1.0
                matrix[row, column] += sign
        if normalize_embeddings and len(texts):
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.maximum(norms, 1e-12)
        return matrix
