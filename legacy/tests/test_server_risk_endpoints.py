"""FastAPI TestClient coverage for the /api/risk-validation/* endpoints:
auth enforcement and the run -> approve -> download lifecycle."""

import json
from dataclasses import dataclass, field

from fastapi.testclient import TestClient

import server
from agents.llm import ModelSpec


@dataclass
class _FakeLLMConfig:
    models: list[ModelSpec] = field(default_factory=lambda: [ModelSpec(name="gpt-5.1-mini", priority=1)])


class _FakeLLM:
    cfg = _FakeLLMConfig()

    def health(self):
        return True

    def chat_json(self, messages, schema_hint="", **kwargs):
        if '"scope"' in schema_hint and '"methodology"' in schema_hint:
            return {"scope": "Scope text", "methodology": "Methodology text", "overall_conclusion": "Conclusion text"}
        if '"agrees"' in schema_hint:
            return {"agrees": True, "rationale": ""}
        if '"accept"' in schema_hint:
            return {"accept": True, "issues": [], "suggested_revisions": []}
        return {"findings": [{
            "domain": "credit_risk", "area": "Conceptual soundness", "verdict": "compliant",
            "severity": "observation", "description": "Case file review looks reasonable and well documented.",
        }]}


class _FakeRAG:
    def retrieve(self, task, k=8, llm=None):
        return [{"id": "doc1", "text": "placeholder regulatory text"}]


_CREDIT_PAYLOAD = {
    "domain": "credit_risk",
    "inputs": {
        "model_id": "PD-RETAIL-01", "model_type": "PD", "exposure_class": "retail",
        "portfolio_segment": "mortgages", "estimation_approach": "internal_ratings_based",
        "population_stability_index": 0.30, "gini_coefficient": 0.35,
        "backtesting_exceptions_count": 10, "backtesting_observations_count": 100,
        "override_rate_pct": 0.15,
    },
}


def _patch_backends(monkeypatch, tmp_path):
    from tools.risk_storage import RiskStorage

    monkeypatch.setattr(server, "_llm", lambda cfg: _FakeLLM())
    monkeypatch.setattr(server, "_regulatory_rag", lambda cfg: _FakeRAG())
    storage = RiskStorage(tmp_path / "risk_validation.db")
    monkeypatch.setattr(server, "_risk_storage", lambda cfg: storage)


def test_risk_run_is_disabled_without_server_token(monkeypatch):
    monkeypatch.delenv("RISK_VALIDATION_API_TOKEN", raising=False)
    response = TestClient(server.app).post("/api/risk-validation/run", json=_CREDIT_PAYLOAD)
    assert response.status_code == 503


def test_risk_run_rejects_wrong_server_token(monkeypatch):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    response = TestClient(server.app).post(
        "/api/risk-validation/run",
        headers={"X-Risk-Token": "wrong-secret"},
        json=_CREDIT_PAYLOAD,
    )
    assert response.status_code == 401


def test_risk_run_rejects_invalid_domain(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    _patch_backends(monkeypatch, tmp_path)
    response = TestClient(server.app).post(
        "/api/risk-validation/run",
        headers={"X-Risk-Token": "correct-secret"},
        json={"domain": "not_a_domain", "inputs": {}},
    )
    assert response.status_code == 400


def test_risk_run_rejects_invalid_inputs_for_domain(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    _patch_backends(monkeypatch, tmp_path)
    response = TestClient(server.app).post(
        "/api/risk-validation/run",
        headers={"X-Risk-Token": "correct-secret"},
        json={"domain": "credit_risk", "inputs": {"model_id": "X"}},  # missing required fields
    )
    assert response.status_code == 422


def test_report_download_404s_for_unknown_run(monkeypatch):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    response = TestClient(server.app).get("/api/risk-validation/does-not-exist/report.pptx")
    assert response.status_code == 404


def _run_and_wait(client: TestClient, headers: dict, payload: dict = _CREDIT_PAYLOAD) -> dict:
    """POST /run and fetch the completed result. TestClient runs
    BackgroundTasks synchronously as part of handling the request, so the
    risk-validation run has already finished by the time the POST returns."""
    run_resp = client.post("/api/risk-validation/run", headers=headers, json=payload)
    assert run_resp.status_code == 200
    queued = run_resp.json()
    assert queued["status"] == "queued"
    run_id = queued["run_id"]
    status_resp = client.get(f"/api/risk-validation/{run_id}")
    assert status_resp.status_code == 200
    return status_resp.json()


def test_run_returns_queued_status_immediately(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    _patch_backends(monkeypatch, tmp_path)
    response = TestClient(server.app).post(
        "/api/risk-validation/run", headers={"X-Risk-Token": "correct-secret"}, json=_CREDIT_PAYLOAD,
    )
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "queued"
    assert "run_id" in data


def test_stream_reports_completed_after_run_finishes(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    _patch_backends(monkeypatch, tmp_path / "db")
    monkeypatch.setattr(server, "OUTPUT_RISK_VALIDATION", tmp_path / "files")
    client = TestClient(server.app)
    headers = {"X-Risk-Token": "correct-secret"}

    run_resp = client.post("/api/risk-validation/run", headers=headers, json=_CREDIT_PAYLOAD)
    run_id = run_resp.json()["run_id"]

    with client.stream("GET", f"/api/risk-validation/{run_id}/stream") as stream_resp:
        assert stream_resp.status_code == 200
        body = ""
        for chunk in stream_resp.iter_text():
            body += chunk
            if "event: completed" in body:
                break
        assert "event: completed" in body


def test_history_endpoint_returns_prior_cycles(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    _patch_backends(monkeypatch, tmp_path)
    client = TestClient(server.app)
    headers = {"X-Risk-Token": "correct-secret"}

    _run_and_wait(client, headers)
    _run_and_wait(client, headers)

    response = client.get(
        "/api/risk-validation/history",
        params={"entity": "PD-RETAIL-01 (mortgages)", "domain": "credit_risk"},
    )
    assert response.status_code == 200
    data = response.json()
    assert len(data["reports"]) == 2


_MATERIAL_CHANGE_PAYLOAD = {
    "domain": "credit_risk",
    "inputs": {
        "model_id": "PD-RETAIL-01", "model_type": "PD", "exposure_class": "retail",
        "portfolio_segment": "mortgages", "estimation_approach": "internal_ratings_based",
        "model_change_type": "material", "model_change_pre_approval_obtained": False,
        "moc_framework_documented": False,
    },
}


def test_new_gates_and_recommendation_surface_in_run_json(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    _patch_backends(monkeypatch, tmp_path / "db")
    monkeypatch.setattr(server, "OUTPUT_RISK_VALIDATION", tmp_path / "files")
    client = TestClient(server.app)
    headers = {"X-Risk-Token": "correct-secret"}

    run_data = _run_and_wait(client, headers, payload=_MATERIAL_CHANGE_PAYLOAD)
    assert run_data["gate_passed"] is False
    areas = {f["area"] for f in run_data["findings"]}
    assert "Model change management" in areas
    assert "Margin of Conservatism" in areas
    report = run_data["report"]
    assert report["recommendation"] == "reject"
    change_finding = next(f for f in run_data["findings"] if f["area"] == "Model change management")
    assert change_finding["regulatory_reference"].startswith("Commission Delegated Regulation (EU) No 529/2014")
    assert change_finding["regulatory_reference_structured"] is not None


def test_preparer_cannot_sign_off_via_api(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    _patch_backends(monkeypatch, tmp_path / "db")
    monkeypatch.setattr(server, "OUTPUT_RISK_VALIDATION", tmp_path / "files")
    client = TestClient(server.app)
    headers = {"X-Risk-Token": "correct-secret"}

    payload = {**_CREDIT_PAYLOAD, "preparer": "Alice Preparer"}
    run_data = _run_and_wait(client, headers, payload=payload)
    resp = client.post(
        f"/api/risk-validation/{run_data['run_id']}/approve",
        headers=headers,
        json={"approval_token": run_data["approval_token"], "signed_off_by": "alice preparer"},
    )
    assert resp.status_code == 403


def test_full_run_approve_download_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    _patch_backends(monkeypatch, tmp_path / "db")
    monkeypatch.setattr(server, "OUTPUT_RISK_VALIDATION", tmp_path / "files")

    client = TestClient(server.app)
    headers = {"X-Risk-Token": "correct-secret"}

    run_data = _run_and_wait(client, headers)
    run_id = run_data["run_id"]
    token = run_data["approval_token"]
    assert run_data["gate_passed"] is False  # PSI/gini/backtesting/override breaches
    assert len(run_data["findings"]) > 1
    assert token

    status_resp = client.get(f"/api/risk-validation/{run_id}")
    assert status_resp.status_code == 200
    assert status_resp.json()["run_id"] == run_id

    pptx_resp = client.get(f"/api/risk-validation/{run_id}/report.pptx")
    assert pptx_resp.status_code == 200
    assert len(pptx_resp.content) > 1000

    docx_resp = client.get(f"/api/risk-validation/{run_id}/report.docx")
    assert docx_resp.status_code == 200

    bad_ext_resp = client.get(f"/api/risk-validation/{run_id}/report.txt")
    assert bad_ext_resp.status_code == 400

    approve_resp = client.post(
        f"/api/risk-validation/{run_id}/approve",
        headers=headers,
        json={"approval_token": token, "signed_off_by": "jane.validator"},
    )
    assert approve_resp.status_code == 200
    approved_report = approve_resp.json()
    assert approved_report["signed_off_by"] == "jane.validator"

    reuse_resp = client.post(
        f"/api/risk-validation/{run_id}/approve",
        headers=headers,
        json={"approval_token": token, "signed_off_by": "jane.validator"},
    )
    assert reuse_resp.status_code == 403

    final_docx_resp = client.get(f"/api/risk-validation/{run_id}/report.docx")
    assert final_docx_resp.status_code == 200
    assert b"jane.validator" in final_docx_resp.content or len(final_docx_resp.content) > 1000


def test_rollup_builds_deck_from_explicit_run_ids(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    _patch_backends(monkeypatch, tmp_path / "db")
    monkeypatch.setattr(server, "OUTPUT_RISK_VALIDATION", tmp_path / "files")

    client = TestClient(server.app)
    headers = {"X-Risk-Token": "correct-secret"}

    run_id = _run_and_wait(client, headers)["run_id"]

    rollup_resp = client.post(
        "/api/risk-validation/rollup", headers=headers, json={"run_ids": [run_id]},
    )
    assert rollup_resp.status_code == 200
    rollup_data = rollup_resp.json()
    assert rollup_data["report_count"] == 1

    download_resp = client.get(rollup_data["download_url"])
    assert download_resp.status_code == 200
    assert len(download_resp.content) > 1000


def test_rollup_404s_when_no_reports_match(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    _patch_backends(monkeypatch, tmp_path)
    response = TestClient(server.app).post(
        "/api/risk-validation/rollup",
        headers={"X-Risk-Token": "correct-secret"},
        json={"run_ids": ["does-not-exist"]},
    )
    assert response.status_code == 404


def test_rollup_download_404s_for_unknown_rollup(monkeypatch):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    response = TestClient(server.app).get("/api/risk-validation/rollup/does-not-exist")
    assert response.status_code == 404
