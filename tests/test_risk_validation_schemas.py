"""Pydantic validation tests for agents/risk_schemas.py."""

import pytest
from pydantic import ValidationError

from agents.risk_schemas import (
    CreditModelValidationInputs,
    LiquidityRiskMetrics,
    MarketRiskMetrics,
    ModelRiskValidationInputs,
    ModelStabilityMetrics,
    NonCreditRiskValidationInputs,
    OperationalRiskMetrics,
    RiskValidationRun,
    ValidationFinding,
    ValidationReport,
    worst_severity,
)


def test_validation_finding_requires_min_length_description():
    with pytest.raises(ValidationError):
        ValidationFinding(domain="credit_risk", area="x", verdict="compliant", severity="low", description="short")


def test_validation_finding_rejects_invalid_enum_values():
    with pytest.raises(ValidationError):
        ValidationFinding(
            domain="credit_risk", area="x", verdict="maybe", severity="low",
            description="A description long enough to pass validation.",
        )
    with pytest.raises(ValidationError):
        ValidationFinding(
            domain="not_a_domain", area="x", verdict="compliant", severity="low",
            description="A description long enough to pass validation.",
        )


def test_validation_report_carries_disclaimer_by_default():
    report = ValidationReport(
        domain="credit_risk", title="t", scope="s", methodology="m",
        entity_under_review="e", reporting_period="2026Q2", overall_rating="compliant",
    )
    assert "DRAFT" in report.disclaimer
    assert report.requires_human_signoff is True
    assert report.signed_off_by is None


def test_validation_report_overall_rating_accepts_both_domain_vocabularies():
    # compliant/non-compliant vocabulary (credit / non-credit)
    r1 = ValidationReport(
        domain="credit_risk", title="t", scope="s", methodology="m",
        entity_under_review="e", reporting_period="p", overall_rating="non_compliant",
    )
    assert r1.overall_rating == "non_compliant"
    # low/medium/high/unacceptable vocabulary (model risk)
    r2 = ValidationReport(
        domain="model_risk", title="t", scope="s", methodology="m",
        entity_under_review="e", reporting_period="p", overall_rating="unacceptable",
    )
    assert r2.overall_rating == "unacceptable"


def test_worst_severity_picks_the_most_severe():
    findings = [
        ValidationFinding(domain="credit_risk", area="a", verdict="compliant", severity="low", description="fine, nothing notable here."),
        ValidationFinding(domain="credit_risk", area="b", verdict="non_compliant", severity="critical", description="serious breach detected here."),
        ValidationFinding(domain="credit_risk", area="c", verdict="partially_compliant", severity="medium", description="minor issue detected here."),
    ]
    assert worst_severity(findings) == "critical"
    assert worst_severity([]) is None


def test_risk_validation_run_defaults():
    run = RiskValidationRun(domain="model_risk")
    assert run.run_id
    assert run.gate_passed is False
    assert run.findings == []
    assert run.report is None
    assert run.approval_token is None


def test_credit_model_validation_inputs_requires_enum_fields():
    with pytest.raises(ValidationError):
        CreditModelValidationInputs(
            model_id="X", model_type="not_a_type", exposure_class="retail",
            portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
        )
    inputs = CreditModelValidationInputs(
        model_id="X", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
    )
    assert inputs.population_stability_index is None


def test_non_credit_risk_validation_inputs_market_metrics():
    inputs = NonCreditRiskValidationInputs(
        risk_type="market", business_unit="Trading", reporting_date="2026-06-30",
        market_metrics=MarketRiskMetrics(var_backtesting_exceptions=5, var_backtesting_observations=250),
    )
    assert inputs.market_metrics.var_backtesting_exceptions == 5
    assert inputs.operational_metrics is None


def test_non_credit_risk_validation_inputs_operational_and_liquidity_metrics():
    op = OperationalRiskMetrics(loss_event_count=3, gross_loss_amount=250000.0, event_category="internal_fraud")
    assert op.recovery_amount == 0.0
    liq = LiquidityRiskMetrics(lcr_pct=95.0, nsfr_pct=110.0)
    assert liq.lcr_pct == 95.0


def test_model_risk_validation_inputs_with_stability_metrics():
    inputs = ModelRiskValidationInputs(
        model_id="M1", model_name="IFRS9 ECL Retail", model_tier="tier_1_high_materiality",
        model_owner="Credit Risk", activities_performed=["stability_testing", "benchmarking"],
        stability_metrics=ModelStabilityMetrics(psi=0.3, gini=0.35, psi_threshold_breached=True),
    )
    assert inputs.stability_metrics.psi_threshold_breached is True
    with pytest.raises(ValidationError):
        ModelRiskValidationInputs(
            model_id="M1", model_name="X", model_tier="not_a_tier",
            model_owner="Credit Risk",
        )
