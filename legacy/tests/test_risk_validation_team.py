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
from tools.risk_storage import RiskStorage


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
    elif '"agrees"' in schema_hint:
        content = json.dumps({"agrees": True, "rationale": "consistent with the case file"})
    elif '"accept"' in schema_hint:
        content = json.dumps({"accept": True, "issues": [], "suggested_revisions": []})
    else:
        content = json.dumps({"findings": [{
            "domain": "credit_risk", "area": "Conceptual soundness", "verdict": "compliant",
            "severity": "observation", "description": "Model design appears sound based on the case file.",
            "evidence": ["doc1"],
        }]})
    return httpx2.Response(200, json={"choices": [{"message": {"content": content}}]})


def _orchestrator(llm: HostedLLM, tmp_path) -> RiskValidationOrchestrator:
    storage = RiskStorage(tmp_path / "risk_validation.db")
    return RiskValidationOrchestrator(
        llm=llm,
        rag_agent=RAGAgent(_FakeRAG(), llm),
        credit_agent=CreditRiskValidationAgent(llm),
        noncredit_agent=NonCreditRiskValidationAgent(llm),
        model_agent=ModelRiskValidationAgent(llm),
        gate_agent=ValidationGateAgent(),
        composer_agent=ReportComposerAgent(llm),
        storage=storage,
    )


def test_orchestrator_run_dispatches_by_domain_and_appends_gate_findings(tmp_path):
    llm = _mock_llm(_default_handler)
    orch = _orchestrator(llm, tmp_path)
    inputs = CreditModelValidationInputs(
        model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
        population_stability_index=0.30, gini_coefficient=0.55,
    )
    run = orch.run("credit_risk", inputs)

    assert run.domain == "credit_risk"
    agents_in_trace = [t.agent for t in run.trace]
    assert agents_in_trace == ["rag", "credit_validator", "critique", "gate", "composer", "approval"]
    assert any(f.severity == "critical" for f in run.findings)  # PSI breach from the gate
    assert any(f.area == "Conceptual soundness" for f in run.findings)  # LLM-drafted finding
    assert run.gate_passed is False
    assert run.report is not None
    assert run.report.overall_rating == "non_compliant"
    assert run.approval_token is not None
    # persisted -- a fresh load from storage returns the same run
    assert orch.storage.load_run(run.run_id).run_id == run.run_id


def test_orchestrator_run_cross_checks_critical_llm_drafted_findings(tmp_path):
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        schema_hint = next(
            (m["content"] for m in body["messages"] if m["role"] == "system" and "schema hint" in m["content"].lower()),
            "",
        )
        if '"scope"' in schema_hint and '"methodology"' in schema_hint:
            content = json.dumps({"scope": "s", "methodology": "m", "overall_conclusion": "c"})
        elif '"agrees"' in schema_hint:
            content = json.dumps({"agrees": True})
        elif '"accept"' in schema_hint:
            content = json.dumps({"accept": True, "issues": [], "suggested_revisions": []})
        else:
            content = json.dumps({"findings": [{
                "domain": "credit_risk", "area": "Use test", "verdict": "non_compliant",
                "severity": "high", "description": "Evidence in the case file suggests the model is not genuinely used in credit decisions.",
            }]})
        return httpx2.Response(200, json={"choices": [{"message": {"content": content}}]})

    llm = _mock_llm(handler)
    orch = _orchestrator(llm, tmp_path)
    inputs = CreditModelValidationInputs(
        model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
    )
    run = orch.run("credit_risk", inputs)
    assert "cross_model_check" in [t.agent for t in run.trace]
    assert any(f.area == "Use test" and f.secondary_review_flag is False for f in run.findings)


def test_orchestrator_run_model_risk_domain(tmp_path):
    llm = _mock_llm(_default_handler)
    orch = _orchestrator(llm, tmp_path)
    inputs = ModelRiskValidationInputs(
        model_id="M1", model_name="IFRS9 ECL Retail", model_tier="tier_1_high_materiality",
        model_owner="Credit Risk",
    )
    run = orch.run("model_risk", inputs)
    assert run.domain == "model_risk"
    assert "model_validator" in [t.agent for t in run.trace]
    assert run.report.overall_rating in {"low", "medium", "high", "unacceptable"}


def test_orchestrator_run_revises_findings_when_critique_rejects(tmp_path):
    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        schema_hint = next(
            (m["content"] for m in body["messages"] if m["role"] == "system" and "schema hint" in m["content"].lower()),
            "",
        )
        if '"scope"' in schema_hint and '"methodology"' in schema_hint:
            content = json.dumps({"scope": "s", "methodology": "m", "overall_conclusion": "c"})
        elif '"agrees"' in schema_hint:
            content = json.dumps({"agrees": True})
        elif '"accept"' in schema_hint:
            content = json.dumps({"accept": False, "issues": ["severity mismatch"], "suggested_revisions": ["lower severity"]})
        else:
            content = json.dumps({"findings": [{
                "domain": "credit_risk", "area": "Conceptual soundness", "verdict": "compliant",
                "severity": "observation", "description": "Revised finding after critique feedback was applied.",
            }]})
        return httpx2.Response(200, json={"choices": [{"message": {"content": content}}]})

    llm = _mock_llm(handler)
    orch = _orchestrator(llm, tmp_path)
    inputs = CreditModelValidationInputs(
        model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
    )
    run = orch.run("credit_risk", inputs)
    critique_trace = [t for t in run.trace if t.agent == "critique"][0]
    assert critique_trace.status == "revised"
    assert any("Revised finding" in f.description for f in run.findings)


def test_execute_consumes_token_and_signs_off(tmp_path):
    llm = _mock_llm(_default_handler)
    orch = _orchestrator(llm, tmp_path)
    inputs = CreditModelValidationInputs(
        model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
    )
    run = orch.run("credit_risk", inputs)
    report = orch.execute(run.run_id, run.approval_token, "jane.validator")
    assert report.signed_off_by == "jane.validator"
    assert report.signed_off_at is not None
    assert len(report.signoffs) == 1


def test_execute_rejects_wrong_token(tmp_path):
    llm = _mock_llm(_default_handler)
    orch = _orchestrator(llm, tmp_path)
    inputs = CreditModelValidationInputs(
        model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
    )
    run = orch.run("credit_risk", inputs)
    with pytest.raises(PermissionError):
        orch.execute(run.run_id, "wrong-token", "jane.validator")


def test_execute_rejects_reused_token(tmp_path):
    llm = _mock_llm(_default_handler)
    orch = _orchestrator(llm, tmp_path)
    inputs = CreditModelValidationInputs(
        model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
    )
    run = orch.run("credit_risk", inputs)
    token = run.approval_token
    orch.execute(run.run_id, token, "jane.validator")
    with pytest.raises(PermissionError):
        orch.execute(run.run_id, token, "jane.validator")


def test_execute_requires_all_signoffs_before_finalizing(tmp_path):
    llm = _mock_llm(_default_handler)
    storage = RiskStorage(tmp_path / "risk_validation.db")
    orch = RiskValidationOrchestrator(
        llm=llm,
        rag_agent=RAGAgent(_FakeRAG(), llm),
        credit_agent=CreditRiskValidationAgent(llm),
        noncredit_agent=NonCreditRiskValidationAgent(llm),
        model_agent=ModelRiskValidationAgent(llm),
        gate_agent=ValidationGateAgent(),
        composer_agent=ReportComposerAgent(llm),
        storage=storage,
        required_signoffs=2,
    )
    inputs = CreditModelValidationInputs(
        model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
    )
    run = orch.run("credit_risk", inputs)
    assert run.report.required_signoffs == 2

    report = orch.execute(run.run_id, run.approval_token, "jane.preparer", role="preparer")
    assert report.signed_off_by is None  # only one of two signoffs collected
    assert len(report.signoffs) == 1

    token2, _ = storage.issue_approval(run.run_id, report)
    report2 = orch.execute(run.run_id, token2, "john.reviewer", role="reviewer")
    assert report2.signed_off_by == "john.reviewer"
    assert len(report2.signoffs) == 2


def test_execute_rejects_signoff_by_the_preparer(tmp_path):
    llm = _mock_llm(_default_handler)
    orch = _orchestrator(llm, tmp_path)
    inputs = CreditModelValidationInputs(
        model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
    )
    run = orch.run("credit_risk", inputs, preparer="Alice Preparer")
    assert run.report.preparer == "Alice Preparer"
    with pytest.raises(PermissionError):
        orch.execute(run.run_id, run.approval_token, "alice preparer")  # case/space-insensitive match


def test_execute_rejects_duplicate_signoff_identity(tmp_path):
    llm = _mock_llm(_default_handler)
    storage = RiskStorage(tmp_path / "risk_validation.db")
    orch = RiskValidationOrchestrator(
        llm=llm, rag_agent=RAGAgent(_FakeRAG(), llm),
        credit_agent=CreditRiskValidationAgent(llm), noncredit_agent=NonCreditRiskValidationAgent(llm),
        model_agent=ModelRiskValidationAgent(llm), gate_agent=ValidationGateAgent(),
        composer_agent=ReportComposerAgent(llm), storage=storage, required_signoffs=2,
    )
    inputs = CreditModelValidationInputs(
        model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
    )
    run = orch.run("credit_risk", inputs)
    orch.execute(run.run_id, run.approval_token, "jane.validator", role="preparer")
    token2, _ = storage.issue_approval(run.run_id, storage.load_run(run.run_id).report)
    with pytest.raises(PermissionError):
        orch.execute(run.run_id, token2, "jane.validator", role="reviewer")  # same identity


def test_tier_1_model_run_requires_two_signoffs(tmp_path):
    llm = _mock_llm(_default_handler)
    storage = RiskStorage(tmp_path / "risk_validation.db")
    orch = RiskValidationOrchestrator(
        llm=llm, rag_agent=RAGAgent(_FakeRAG(), llm),
        credit_agent=CreditRiskValidationAgent(llm), noncredit_agent=NonCreditRiskValidationAgent(llm),
        model_agent=ModelRiskValidationAgent(llm), gate_agent=ValidationGateAgent(),
        composer_agent=ReportComposerAgent(llm), storage=storage, required_signoffs=1,
        tier_required_signoffs={"tier_1_high_materiality": 2},
    )
    inputs = ModelRiskValidationInputs(
        model_id="M1", model_name="IFRS9 ECL Retail", model_tier="tier_1_high_materiality",
        model_owner="Credit Risk",
    )
    run = orch.run("model_risk", inputs)
    assert run.report.required_signoffs == 2


def test_report_recommendation_is_derived_from_gate_result(tmp_path):
    llm = _mock_llm(_default_handler)
    orch = _orchestrator(llm, tmp_path)
    inputs = CreditModelValidationInputs(
        model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
        population_stability_index=0.30,  # critical gate finding
    )
    run = orch.run("credit_risk", inputs)
    assert run.report.recommendation == "reject"
    assert run.report.conditions == []  # reject carries no conditions


def test_run_history_feeds_trend_check_on_second_cycle(tmp_path):
    llm = _mock_llm(_default_handler)
    orch = _orchestrator(llm, tmp_path)
    for gini in [0.60, 0.55, 0.50]:
        inputs = CreditModelValidationInputs(
            model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
            portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
            gini_coefficient=gini,
        )
        run = orch.run("credit_risk", inputs)

    # 4th cycle, still declining and still above the 0.40 absolute minimum
    inputs = CreditModelValidationInputs(
        model_id="PD-RETAIL-01", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
        gini_coefficient=0.45,
    )
    run = orch.run("credit_risk", inputs)
    assert any(f.area == "Discriminatory power" and "declined" in f.description for f in run.findings)
