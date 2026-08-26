"""FastAPI TestClient coverage for the /api/risk-validation/* endpoints:
auth enforcement and the run -> approve -> download lifecycle."""

import json

from fastapi.testclient import TestClient

import server


class _FakeLLM:
    def health(self):
        return True

    def chat_json(self, messages, schema_hint="", **kwargs):
        if '"scope"' in schema_hint and '"methodology"' in schema_hint:
            return {"scope": "Scope text", "methodology": "Methodology text", "overall_conclusion": "Conclusion text"}
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


def _patch_backends(monkeypatch):
    monkeypatch.setattr(server, "_llm", lambda cfg: _FakeLLM())
    monkeypatch.setattr(server, "_regulatory_rag", lambda cfg: _FakeRAG())


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


def test_risk_run_rejects_invalid_domain(monkeypatch):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    _patch_backends(monkeypatch)
    response = TestClient(server.app).post(
        "/api/risk-validation/run",
        headers={"X-Risk-Token": "correct-secret"},
        json={"domain": "not_a_domain", "inputs": {}},
    )
    assert response.status_code == 400


def test_risk_run_rejects_invalid_inputs_for_domain(monkeypatch):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    _patch_backends(monkeypatch)
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


def test_full_run_approve_download_lifecycle(monkeypatch, tmp_path):
    monkeypatch.setenv("RISK_VALIDATION_API_TOKEN", "correct-secret")
    _patch_backends(monkeypatch)
    monkeypatch.setattr(server, "OUTPUT_RISK_VALIDATION", tmp_path)

    client = TestClient(server.app)
    headers = {"X-Risk-Token": "correct-secret"}

    run_resp = client.post("/api/risk-validation/run", headers=headers, json=_CREDIT_PAYLOAD)
    assert run_resp.status_code == 200
    run_data = run_resp.json()
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
