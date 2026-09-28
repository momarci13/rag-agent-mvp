from fastapi.testclient import TestClient

import server


def test_quant_api_is_disabled_without_server_token(monkeypatch):
    monkeypatch.delenv("TRADER_API_TOKEN", raising=False)
    response = TestClient(server.app).post(
        "/api/quant-team/run",
        json={"task": "research SPY"},
    )
    assert response.status_code == 503


def test_quant_api_rejects_wrong_server_token(monkeypatch):
    monkeypatch.setenv("TRADER_API_TOKEN", "correct-secret")
    response = TestClient(server.app).post(
        "/api/quant-team/run",
        headers={"X-Trader-Token": "wrong-secret"},
        json={"task": "research SPY"},
    )
    assert response.status_code == 401


def test_health_is_partial_when_hosted_inference_is_down(monkeypatch):
    class DownLLM:
        def health(self):
            return False

    class EmptyRAG:
        def __len__(self):
            return 0

    monkeypatch.setattr(server, "_llm", lambda _cfg: DownLLM())
    monkeypatch.setattr(server, "_rag", lambda _cfg: EmptyRAG())
    response = TestClient(server.app).get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "partial"
