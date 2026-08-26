"""Embedding backends for retrieval.

Production inference is routed through the direct OpenAI embeddings API.
The deterministic hash backend exists only for unit tests and offline
structural validation.
"""
from __future__ import annotations

import hashlib
import os
import re
from typing import Any, Protocol, Sequence

import httpx2
import numpy as np
import openai


class EmbeddingClient(Protocol):
    def encode(
        self,
        texts: Sequence[str],
        *,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> np.ndarray: ...


class HostedEmbeddings:
    """Direct OpenAI client for hosted embedding inference."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout_s: int = 120,
        batch_size: int = 64,
        provider_name: str = "OpenAI",
        require_api_key: bool = True,
        transport: httpx2.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.base_url = (
            base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        ).rstrip("/")
        self.api_key = (
            api_key if api_key is not None else os.getenv("OPENAI_API_KEY", "")
        )
        self.timeout_s = timeout_s
        self.batch_size = max(1, batch_size)
        self.provider_name = provider_name
        self.require_api_key = require_api_key
        http_client = httpx2.Client(timeout=timeout_s, transport=transport)
        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key or "sk-not-configured",
            "http_client": http_client,
            "max_retries": 0,
        }
        if base_url:
            client_kwargs["base_url"] = base_url
        self._client = openai.OpenAI(**client_kwargs)

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
            response = self._client.embeddings.create(
                model=self.model, input=batch, encoding_format="float",
            )
            data = sorted(response.data, key=lambda item: item.index)
            if len(data) != len(batch):
                raise ValueError(
                    f"{self.provider_name} returned an unexpected embedding count"
                )
            rows.extend(item.embedding for item in data)

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
            self._client.models.list()
            return True
        except openai.OpenAIError:
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
