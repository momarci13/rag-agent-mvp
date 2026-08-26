import json

import httpx2
import numpy as np

from rag.embeddings import HostedEmbeddings


def test_openai_embeddings_contract():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json={
            "data": [
                {"index": 0, "embedding": [3.0, 4.0]},
                {"index": 1, "embedding": [0.0, 2.0]},
            ]
        })

    client = HostedEmbeddings(
        "text-embedding-3-large",
        api_key="sk-test",
        transport=httpx2.MockTransport(handler),
    )
    vectors = client.encode(["alpha", "beta"])

    assert captured["path"] == "/v1/embeddings"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["model"] == "text-embedding-3-large"
    assert vectors.shape == (2, 2)
    assert np.allclose(np.linalg.norm(vectors, axis=1), [1.0, 1.0])


def test_openai_embeddings_health_requires_api_key():
    client = HostedEmbeddings(
        "text-embedding-3-large",
        api_key="",
    )
    assert client.health() is False


def test_openai_embeddings_health_validates_via_models_list():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["path"] = request.url.path
        return httpx2.Response(200, json={"object": "list", "data": []})

    client = HostedEmbeddings(
        "text-embedding-3-large",
        api_key="sk-test",
        transport=httpx2.MockTransport(handler),
    )
    assert client.health() is True
    assert captured["path"] == "/v1/models"


def test_openai_embeddings_health_returns_false_on_error():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"error": {"message": "invalid key"}})

    client = HostedEmbeddings(
        "text-embedding-3-large",
        api_key="sk-bad",
        transport=httpx2.MockTransport(handler),
    )
    assert client.health() is False
