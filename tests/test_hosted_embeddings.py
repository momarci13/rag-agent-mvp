import json

import httpx
import numpy as np

from rag.embeddings import HostedEmbeddings


def test_openrouter_free_embeddings_openai_contract():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "data": [
                {"index": 0, "embedding": [3.0, 4.0]},
                {"index": 1, "embedding": [0.0, 2.0]},
            ]
        })

    client = HostedEmbeddings(
        "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        base_url="https://openrouter.ai/api/v1",
        api_key="sk-or-test",
        transport=httpx.MockTransport(handler),
    )
    vectors = client.encode(["alpha", "beta"])

    assert captured["path"] == "/api/v1/embeddings"
    assert captured["auth"] == "Bearer sk-or-test"
    assert captured["body"]["model"] == "nvidia/llama-nemotron-embed-vl-1b-v2:free"
    assert vectors.shape == (2, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])


def test_openrouter_embeddings_health_requires_api_key():
    client = HostedEmbeddings(
        "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        api_key="",
    )
    assert client.health() is False


def test_openrouter_embeddings_health_validates_current_key_endpoint():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        return httpx.Response(200, json={"data": {"is_free_tier": True}})

    client = HostedEmbeddings(
        "nvidia/llama-nemotron-embed-vl-1b-v2:free",
        api_key="sk-or-test",
        transport=httpx.MockTransport(handler),
    )
    assert client.health() is True
    assert captured["path"] == "/api/v1/key"
