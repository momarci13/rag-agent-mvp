"""Pydantic validation tests for agents/risk_schemas.py."""

import pytest
from pydantic import ValidationError

from agents.regulatory_refs import reference_for
from agents.risk_schemas import (
    CREDIT_VALIDATION_AREAS,
    CreditModelValidationInputs,
    LiquidityRiskMetrics,
    MarketRiskMetrics,
    ModelRiskValidationInputs,
    ModelStabilityMetrics,
    NonCreditRiskValidationInputs,
    OperationalRiskMetrics,
    PriorFindingStatus,
    RatingGradeObservation,
    RegulatoryReference,
    RiskValidationRun,
    SignoffRecord,
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


def test_validation_finding_defaults_for_new_v2_fields():
    finding = ValidationFinding(
        domain="credit_risk", area="x", verdict="compliant", severity="low",
        description="A description long enough to pass validation.",
    )
    assert finding.finding_status == "open"
    assert finding.evidence_grounded is None
    assert finding.secondary_review_flag is False
    assert finding.regulatory_reference_structured is None


def test_validation_finding_accepts_structured_regulatory_reference():
    finding = ValidationFinding(
        domain="credit_risk", area="x", verdict="non_compliant", severity="high",
        description="A description long enough to pass validation.",
        regulatory_reference_structured=RegulatoryReference(template_code="C 07.00", row="010", paragraph="45"),
    )
    assert finding.regulatory_reference_structured.template_code == "C 07.00"


def test_validation_report_default_requires_single_signoff():
    report = ValidationReport(
        domain="credit_risk", title="t", scope="s", methodology="m",
        entity_under_review="e", reporting_period="p", overall_rating="compliant",
    )
    assert report.required_signoffs == 1
    assert report.signoffs == []


def test_validation_report_accepts_multiple_signoff_records():
    report = ValidationReport(
        domain="credit_risk", title="t", scope="s", methodology="m",
        entity_under_review="e", reporting_period="p", overall_rating="compliant",
        required_signoffs=2,
        signoffs=[SignoffRecord(by="jane", role="preparer"), SignoffRecord(by="john", role="reviewer")],
    )
    assert len(report.signoffs) == 2
    assert report.signoffs[1].role == "reviewer"


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
    assert inputs.jurisdiction == "EU"
    assert inputs.ifrs9_stage is None
    assert inputs.single_name_concentration_pct is None


def test_credit_model_validation_inputs_ifrs9_staging_and_concentration_fields():
    inputs = CreditModelValidationInputs(
        model_id="X", model_type="IFRS9_ECL", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
        jurisdiction="HU", ifrs9_stage="stage_2", sicr_trigger_flag=False,
        days_past_due=5, single_name_concentration_pct=7.5,
    )
    assert inputs.jurisdiction == "HU"
    assert inputs.ifrs9_stage == "stage_2"
    assert inputs.sicr_trigger_flag is False
    with pytest.raises(ValidationError):
        CreditModelValidationInputs(
            model_id="X", model_type="PD", exposure_class="retail",
            portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
            ifrs9_stage="not_a_stage",
        )


def test_non_credit_risk_validation_inputs_market_metrics():
    inputs = NonCreditRiskValidationInputs(
        risk_type="market", business_unit="Trading", reporting_date="2026-06-30",
        market_metrics=MarketRiskMetrics(var_backtesting_exceptions=5, var_backtesting_observations=250),
    )
    assert inputs.market_metrics.var_backtesting_exceptions == 5
    assert inputs.operational_metrics is None
    assert inputs.jurisdiction == "EU"


def test_non_credit_risk_validation_inputs_operational_and_liquidity_metrics():
    op = OperationalRiskMetrics(loss_event_count=3, gross_loss_amount=250000.0, event_category="internal_fraud")
    assert op.recovery_amount == 0.0
    liq = LiquidityRiskMetrics(lcr_pct=95.0, nsfr_pct=110.0)
    assert liq.lcr_pct == 95.0


def test_new_report_fields_default_safely():
    report = ValidationReport(
        domain="credit_risk", title="t", scope="s", methodology="m",
        entity_under_review="e", reporting_period="p", overall_rating="compliant",
    )
    assert report.recommendation == "not_a_recommendation"
    assert report.conditions == []
    assert report.follow_up_on_prior_findings == []
    assert report.validation_sample == "" and report.materiality_rationale == ""
    assert report.deviations_from_policy == []
    assert report.preparer == ""


def test_report_recommendation_rejects_unknown_value():
    with pytest.raises(ValidationError):
        ValidationReport(
            domain="credit_risk", title="t", scope="s", methodology="m",
            entity_under_review="e", reporting_period="p", overall_rating="compliant",
            recommendation="definitely_approve",
        )


def test_prior_finding_status_enum():
    pf = PriorFindingStatus(finding_reference="credit_risk:gini", area="Discriminatory power", status="overdue")
    assert pf.status == "overdue"
    with pytest.raises(ValidationError):
        PriorFindingStatus(finding_reference="x", status="unknown")


def test_credit_validation_areas_include_the_new_eba_gl_2017_16_areas():
    for area in ("Margin of Conservatism", "Downturn LGD estimation", "Model change management"):
        assert area in CREDIT_VALIDATION_AREAS


def test_rating_grade_observation_bounds():
    RatingGradeObservation(grade="A", predicted_pd=0.01, obligors=100, observed_defaults=1)
    with pytest.raises(ValidationError):
        RatingGradeObservation(grade="A", predicted_pd=1.5, obligors=100, observed_defaults=1)
    with pytest.raises(ValidationError):
        RatingGradeObservation(grade="A", predicted_pd=0.01, obligors=-1, observed_defaults=0)


def test_regulatory_refs_helper_populates_structured_reference():
    ref = reference_for("Calibration and back-testing", "credit_risk")
    assert ref.paragraph and ref.paragraph.startswith("EBA/GL/2017/16")
    assert reference_for("some unmapped area").paragraph is None


def test_credit_inputs_accept_new_optional_estimation_fields():
    inputs = CreditModelValidationInputs(
        model_id="X", model_type="LGD", exposure_class="corporate",
        portfolio_segment="large corp", estimation_approach="internal_ratings_based",
        rating_grade_observations=[RatingGradeObservation(grade="A", predicted_pd=0.01, obligors=100, observed_defaults=2)],
        lgd_predicted_mean=0.3, lgd_observed_mean=0.35, downturn_lgd_applied=True,
        moc_framework_documented=True, moc_total_pct=0.05, data_completeness_pct=99.0,
        model_change_type="non_material", validation_function_independent=True,
    )
    assert inputs.rating_grade_observations[0].grade == "A"
    assert inputs.model_change_type == "non_material"


def test_market_metrics_accept_frtb_fields():
    m = MarketRiskMetrics(
        var_backtesting_exceptions=3, var_backtesting_observations=250,
        desk_id="RATES-1", expected_shortfall_975=0.031, stressed_es=0.05,
        var_backtesting_exceptions_975=6, var_backtesting_observations_975=250,
        pla_spearman_correlation=0.82, pla_ks_statistic=0.08,
    )
    assert m.desk_id == "RATES-1"
    assert m.es_horizon_days == 10


def test_model_risk_validation_inputs_with_stability_metrics():
    inputs = ModelRiskValidationInputs(
        model_id="M1", model_name="IFRS9 ECL Retail", model_tier="tier_1_high_materiality",
        model_owner="Credit Risk", activities_performed=["stability_testing", "benchmarking"],
        stability_metrics=ModelStabilityMetrics(psi=0.3, gini=0.35, psi_threshold_breached=True),
    )
    assert inputs.stability_metrics.psi_threshold_breached is True
    assert inputs.jurisdiction == "EU"
    with pytest.raises(ValidationError):
        ModelRiskValidationInputs(
            model_id="M1", model_name="X", model_tier="not_a_tier",
            model_owner="Credit Risk",
        )
