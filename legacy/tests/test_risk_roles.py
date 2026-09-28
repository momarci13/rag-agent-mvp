"""Unit tests for agents/risk_roles.py's grounding, critique, and
cross-model verification helpers, using a mocked HostedLLM transport."""

import json

import httpx2
import pytest

from agents.llm import HostedLLM, LLMConfig, ModelSpec
from agents.risk_roles import critique_findings, draft_credit_findings, revise_findings, verify_high_severity_finding
from agents.risk_schemas import CreditModelValidationInputs, ValidationFinding
from agents.schemas import Critique


def _mock_llm(handler, **cfg_kwargs) -> HostedLLM:
    return HostedLLM(LLMConfig(model="gpt-5.1-mini", api_key="sk-test", **cfg_kwargs), transport=httpx2.MockTransport(handler))


_CREDIT_INPUTS = CreditModelValidationInputs(
    model_id="PD-1", model_type="PD", exposure_class="retail",
    portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
)


def test_draft_credit_findings_strips_ungrounded_evidence_ids():
    def handler(request: httpx2.Request) -> httpx2.Response:
        content = json.dumps({"findings": [{
            "domain": "credit_risk", "area": "Data quality", "verdict": "compliant",
            "severity": "observation", "description": "Data appears complete and reasonably documented.",
            "evidence": ["doc1", "doc-not-retrieved"],
        }]})
        return httpx2.Response(200, json={"choices": [{"message": {"content": content}}]})

    llm = _mock_llm(handler)
    docs = [{"id": "doc1", "text": "Data appears complete and reasonably documented in the case file."}]
    findings = draft_credit_findings(llm, _CREDIT_INPUTS, docs)

    assert len(findings) == 1
    assert findings[0].evidence == ["doc1"]  # the unretrieved ID was stripped


def test_draft_credit_findings_flags_low_evidence_overlap():
    def handler(request: httpx2.Request) -> httpx2.Response:
        content = json.dumps({"findings": [{
            "domain": "credit_risk", "area": "Data quality", "verdict": "compliant",
            "severity": "observation", "description": "Completely unrelated statement about weather patterns.",
            "evidence": ["doc1"],
        }]})
        return httpx2.Response(200, json={"choices": [{"message": {"content": content}}]})

    llm = _mock_llm(handler)
    docs = [{"id": "doc1", "text": "EBA GL 2017-11 discusses PD model calibration and backtesting exceptions."}]
    findings = draft_credit_findings(llm, _CREDIT_INPUTS, docs)

    assert findings[0].evidence_grounded is False


def test_draft_credit_findings_marks_grounded_when_overlap_is_high():
    def handler(request: httpx2.Request) -> httpx2.Response:
        content = json.dumps({"findings": [{
            "domain": "credit_risk", "area": "Calibration and back-testing", "verdict": "compliant",
            "severity": "observation", "description": "Backtesting exceptions and calibration were reviewed per EBA GL 2017-11.",
            "evidence": ["doc1"],
        }]})
        return httpx2.Response(200, json={"choices": [{"message": {"content": content}}]})

    llm = _mock_llm(handler)
    docs = [{"id": "doc1", "text": "EBA GL 2017-11 discusses PD model calibration and backtesting exceptions review."}]
    findings = draft_credit_findings(llm, _CREDIT_INPUTS, docs)

    assert findings[0].evidence_grounded is True


def test_draft_credit_findings_no_evidence_leaves_grounded_flag_none():
    def handler(request: httpx2.Request) -> httpx2.Response:
        content = json.dumps({"findings": [{
            "domain": "credit_risk", "area": "Data quality", "verdict": "compliant",
            "severity": "observation", "description": "Assessment based on general case file review.",
            "evidence": [],
        }]})
        return httpx2.Response(200, json={"choices": [{"message": {"content": content}}]})

    llm = _mock_llm(handler)
    findings = draft_credit_findings(llm, _CREDIT_INPUTS, docs=[])
    assert findings[0].evidence_grounded is None


def test_critique_findings_returns_parsed_critique():
    def handler(request: httpx2.Request) -> httpx2.Response:
        content = json.dumps({"accept": False, "issues": ["severity/verdict mismatch"], "suggested_revisions": ["lower severity to medium"]})
        return httpx2.Response(200, json={"choices": [{"message": {"content": content}}]})

    llm = _mock_llm(handler)
    findings = [ValidationFinding(
        domain="credit_risk", area="x", verdict="compliant", severity="critical",
        description="Contradictory finding for testing purposes here.",
    )]
    critique = critique_findings(llm, findings)
    assert isinstance(critique, Critique)
    assert critique.accept is False
    assert critique.suggested_revisions == ["lower severity to medium"]


def test_revise_findings_returns_corrected_findings():
    def handler(request: httpx2.Request) -> httpx2.Response:
        content = json.dumps({"findings": [{
            "domain": "credit_risk", "area": "x", "verdict": "partially_compliant",
            "severity": "medium", "description": "Corrected finding after critique feedback applied.",
        }]})
        return httpx2.Response(200, json={"choices": [{"message": {"content": content}}]})

    llm = _mock_llm(handler)
    original = [ValidationFinding(
        domain="credit_risk", area="x", verdict="compliant", severity="critical",
        description="Contradictory finding for testing purposes here.",
    )]
    critique = Critique(accept=False, issues=["mismatch"], suggested_revisions=["lower severity"])
    revised = revise_findings(llm, original, critique)
    assert revised[0].severity == "medium"


def test_verify_high_severity_finding_returns_true_without_fallback_tier():
    llm = _mock_llm(lambda request: httpx2.Response(500))  # should never be called
    finding = ValidationFinding(
        domain="credit_risk", area="x", verdict="non_compliant", severity="critical",
        description="A description long enough to pass validation checks.",
    )
    assert verify_high_severity_finding(llm, finding, "case context") is True


def test_verify_high_severity_finding_uses_fallback_model_and_returns_agreement():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        captured["model"] = body["model"]
        return httpx2.Response(200, json={"choices": [{"message": {"content": json.dumps({"agrees": False, "rationale": "disagree"})}}]})

    llm = _mock_llm(
        handler,
        models=[ModelSpec(name="gpt-5.1", priority=1), ModelSpec(name="gpt-5.1-mini", priority=2)],
    )
    finding = ValidationFinding(
        domain="credit_risk", area="x", verdict="non_compliant", severity="critical",
        description="A description long enough to pass validation checks.",
    )
    result = verify_high_severity_finding(llm, finding, "case context")
    assert result is False
    assert captured["model"] == "gpt-5.1-mini"


def test_verify_high_severity_finding_fails_open_on_error():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(500, json={"error": {"message": "boom"}})

    llm = _mock_llm(
        handler,
        models=[ModelSpec(name="gpt-5.1", priority=1), ModelSpec(name="gpt-5.1-mini", priority=2)],
    )
    finding = ValidationFinding(
        domain="credit_risk", area="x", verdict="non_compliant", severity="critical",
        description="A description long enough to pass validation checks.",
    )
    assert verify_high_severity_finding(llm, finding, "case context") is True
