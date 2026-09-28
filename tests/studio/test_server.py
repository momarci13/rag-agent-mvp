from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import server

pytestmark = pytest.mark.slow
TOKEN = {"X-Studio-Token": "secret"}


@pytest.fixture()
def client(monkeypatch, studio):
    monkeypatch.setenv("STUDIO_API_TOKEN", "secret")
    monkeypatch.setattr(server, "_studio", lambda: studio)
    monkeypatch.setattr(server, "_start", lambda pid: studio.run(pid))
    return TestClient(server.app)


def test_mutations_need_a_configured_token(monkeypatch, client):
    monkeypatch.delenv("STUDIO_API_TOKEN")
    r = client.post("/api/projects", data={"mode": "develop_and_validate", "title": "t", "brief": "b"})
    assert r.status_code == 503
    monkeypatch.setenv("STUDIO_API_TOKEN", "secret")
    r = client.post("/api/projects", headers={"X-Studio-Token": "wrong"},
                    data={"mode": "develop_and_validate", "title": "t", "brief": "b"})
    assert r.status_code == 401


def test_project_lifecycle_over_http(client, loans_csv):
    r = client.post(
        "/api/projects", headers=TOKEN,
        data={"mode": "develop_and_validate", "title": "Mortgage PD", "brief": "PD model",
              "frameworks": ["eu_banking", "bogus"], "max_rounds": "2", "challenger": "off"},
        files=[("data", ("loans.csv", loans_csv, "text/csv"))],
    )
    assert r.status_code == 202, r.text
    pid = r.json()["project_id"]

    body = client.get(f"/api/projects/{pid}").json()
    assert body["project"]["input"]["frameworks"] == ["eu_banking"]
    assert body["summary"]["status"] == "awaiting_signoff"
    assert client.get("/api/projects").json()["projects"][0]["project_id"] == pid

    files = [f["path"] for f in client.get(f"/api/projects/{pid}/tree").json()["files"]]
    doc = next(f for f in files if f.endswith(".docx") and "modelling_document" in f)
    assert client.get(f"/api/projects/{pid}/files/{doc}").status_code == 200
    assert client.get(f"/api/projects/{pid}/files/..%2F..%2Fsecret.txt").status_code in (400, 404)
    assert client.get("/api/projects/zzzzzzzzzzzz").status_code == 404

    stream = client.get(f"/api/projects/{pid}/stream")
    assert "event: done" in stream.text

    r = client.post(f"/api/projects/{pid}/signoff", headers=TOKEN, json={"name": "Jane Validator", "role": "Head"})
    assert r.status_code == 200 and r.json()["summary"]["status"] == "signed_off"
    r = client.post(f"/api/projects/{pid}/signoff", headers=TOKEN, json={"name": "Jane Validator"})
    assert r.status_code == 409


def test_bad_upload_is_rejected(client):
    r = client.post("/api/projects", headers=TOKEN,
                    data={"mode": "develop_and_validate", "title": "t", "brief": "b"},
                    files=[("data", ("x.exe", b"MZ", "application/octet-stream"))])
    assert r.status_code == 422


def test_library_lists_builtin_references(client):
    docs = client.get("/api/library").json()["documents"]
    assert any(d["framework"] == "eu_ai_act" for d in docs)
