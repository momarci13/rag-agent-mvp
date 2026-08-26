"""End-to-end tests for the risk-validation orchestrator with a mocked LLM
and a stubbed RAG (no live network calls)."""

import json

import httpx2
import pytest

from agents.llm import HostedLLM, LLMConfig
from agents.quant_team import RAGAgent
from agents.risk_schemas import CreditModelValidationInputs, ModelRiskValidationInputs
from agents.risk_validation_team import (
    CreditRiskValidationAgent,
    ModelRiskValidationAgent,
    NonCreditRiskValidationAgent,
    ReportComposerAgent,
    RiskValidationOrchestrator,
    ValidationGateAgent,
)


class _FakeRAG:
    def retrieve(self, task, k=8, llm=None):
        return [{"id": "doc1", "text": "Placeholder regulatory reference material."}]


def _mock_llm(handler) -> HostedLLM:
    return HostedLLM(LLMConfig(model="gpt-5.1-mini", api_key="sk-test"), transport=httpx2.MockTransport(handler))


def _default_handler(request: httpx2.Request) -> httpx2.Response:
    body = json.loads(request.content)
    schema_hint = next(
        (m["content"] for m in body["messages"] if m["role"] == "system" and "schema hint" in m["content"].lower()),
        "",
    )
    if '"scope"' in schema_hint and '"methodology"' in schema_hint:
        content = json.dumps({
            "scope": "Scope text", "methodology": "Methodology text", "overall_conclusion": "Conclusion text",
        })
    else:
        content = json.dumps({"findings": [{
            "domain": "credit_risk", "area": "Conceptual soundness", "verdict": "compliant",
            "severity": "observation", "description": "Model design appears sound based on the case file.",
            "evidence": ["doc1"],
        }]})
    return httpx2.Response(200, json={"choices": [{"message": {"content": content}}]})


def _orchestrator(llm: HostedLLM) -> RiskValidationOrchestrator:
    return RiskValidationOrchestrator(
        rag_agent=RAGAgent(_FakeRAG(), llm),
        credit_agent=CreditRiskValidationAgent(llm),
        noncredit_agent=NonCreditRiskValidationAgent(llm),
        model_agent=ModelRiskValidationAgent(llm),
        gate_agent=ValidationGateAgent(),
        composer_agent=ReportComposerAgent(llm),
    )


def test_orchestrator_run_dispatches_by_domain_and_appends_gate_findings():
    llm = _mock_llm(_default_handler)
    orch = _orchestrator(llm)
    inputs = CreditModelValidationInputs(
        model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
        population_stability_index=0.30, gini_coefficient=0.55,
    )
    run = orch.run("credit_risk", inputs)

    assert run.domain == "credit_risk"
    agents_in_trace = [t.agent for t in run.trace]
    assert agents_in_trace == ["rag", "credit_validator", "gate", "composer", "approval"]
    assert any(f.severity == "critical" for f in run.findings)  # PSI breach from the gate
    assert any(f.area == "Conceptual soundness" for f in run.findings)  # LLM-drafted finding
    assert run.gate_passed is False
    assert run.report is not None
    assert run.report.overall_rating == "non_compliant"
    assert run.approval_token is not None


def test_orchestrator_run_model_risk_domain():
    llm = _mock_llm(_default_handler)
    orch = _orchestrator(llm)
    inputs = ModelRiskValidationInputs(
        model_id="M1", model_name="IFRS9 ECL Retail", model_tier="tier_1_high_materiality",
        model_owner="Credit Risk",
    )
    run = orch.run("model_risk", inputs)
    assert run.domain == "model_risk"
    assert "model_validator" in [t.agent for t in run.trace]
    assert run.report.overall_rating in {"low", "medium", "high", "unacceptable"}


def test_execute_consumes_token_and_signs_off():
    llm = _mock_llm(_default_handler)
    orch = _orchestrator(llm)
    inputs = CreditModelValidationInputs(
        model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
    )
    run = orch.run("credit_risk", inputs)
    report = orch.execute(run, run.approval_token, "jane.validator")
    assert report.signed_off_by == "jane.validator"
    assert report.signed_off_at is not None


def test_execute_rejects_wrong_token():
    llm = _mock_llm(_default_handler)
    orch = _orchestrator(llm)
    inputs = CreditModelValidationInputs(
        model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
    )
    run = orch.run("credit_risk", inputs)
    with pytest.raises(PermissionError):
        orch.execute(run, "wrong-token", "jane.validator")


def test_execute_rejects_reused_token():
    llm = _mock_llm(_default_handler)
    orch = _orchestrator(llm)
    inputs = CreditModelValidationInputs(
        model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
    )
    run = orch.run("credit_risk", inputs)
    token = run.approval_token
    orch.execute(run, token, "jane.validator")
    with pytest.raises(PermissionError):
        orch.execute(run, token, "jane.validator")
