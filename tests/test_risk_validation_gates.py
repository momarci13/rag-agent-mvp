"""Pure unit tests for the deterministic ValidationGateAgent.

No LLM/network mocking needed -- these are plain-Python threshold checks,
mirroring tests/test_risk.py's direct-math-assertion style.
"""

from datetime import date, timedelta

from agents.risk_schemas import (
    CreditModelValidationInputs,
    LiquidityRiskMetrics,
    MarketRiskMetrics,
    ModelRiskValidationInputs,
    ModelStabilityMetrics,
    NonCreditRiskValidationInputs,
    OperationalRiskMetrics,
    RatingGradeObservation,
    ValidationReport,
)
from studio.regulatory_gates import ValidationGateAgent, ValidationThresholds

THRESHOLDS = ValidationThresholds()
GATE = ValidationGateAgent(THRESHOLDS)


def _credit_inputs(**overrides):
    base = dict(
        model_id="PD-1", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
    )
    base.update(overrides)
    return CreditModelValidationInputs(**base)


def test_credit_gate_passes_with_compliant_metrics():
    inputs = _credit_inputs(
        population_stability_index=0.05, gini_coefficient=0.55,
        backtesting_exceptions_count=1, backtesting_observations_count=100,
        override_rate_pct=0.02,
    )
    findings, gate_passed = GATE.run("credit_risk", inputs)
    assert findings == []
    assert gate_passed is True


def test_credit_gate_flags_psi_breach_as_critical():
    inputs = _credit_inputs(population_stability_index=0.30)
    findings, gate_passed = GATE.run("credit_risk", inputs)
    assert any(f.severity == "critical" and "Population Stability" in f.description for f in findings)
    assert gate_passed is False


def test_credit_gate_flags_low_gini_as_high():
    inputs = _credit_inputs(gini_coefficient=0.10)
    findings, gate_passed = GATE.run("credit_risk", inputs)
    assert any(f.severity == "high" and f.area == "Discriminatory power" for f in findings)
    assert gate_passed is True  # high, not critical -- gate still "passes" (no critical)


def test_credit_gate_flags_backtesting_exception_rate():
    inputs = _credit_inputs(backtesting_exceptions_count=20, backtesting_observations_count=100)
    findings, _ = GATE.run("credit_risk", inputs)
    assert any(f.area == "Calibration and back-testing" and f.severity == "high" for f in findings)


def test_credit_gate_flags_override_rate():
    inputs = _credit_inputs(override_rate_pct=0.25)
    findings, gate_passed = GATE.run("credit_risk", inputs)
    assert any(f.area == "Override analysis" for f in findings)
    assert gate_passed is True  # medium severity, not critical


def test_non_credit_market_gate_green_zone_produces_no_finding():
    inputs = NonCreditRiskValidationInputs(
        risk_type="market", business_unit="Trading", reporting_date="2026-06-30",
        market_metrics=MarketRiskMetrics(var_backtesting_exceptions=2, var_backtesting_observations=250),
    )
    findings, gate_passed = GATE.run("non_credit_risk", inputs)
    assert findings == []
    assert gate_passed is True


def test_non_credit_market_gate_amber_zone():
    inputs = NonCreditRiskValidationInputs(
        risk_type="market", business_unit="Trading", reporting_date="2026-06-30",
        market_metrics=MarketRiskMetrics(var_backtesting_exceptions=6, var_backtesting_observations=250),
    )
    findings, gate_passed = GATE.run("non_credit_risk", inputs)
    assert len(findings) == 1
    assert findings[0].severity == "medium"
    assert gate_passed is True


def test_non_credit_market_gate_red_zone_is_critical():
    inputs = NonCreditRiskValidationInputs(
        risk_type="market", business_unit="Trading", reporting_date="2026-06-30",
        market_metrics=MarketRiskMetrics(var_backtesting_exceptions=12, var_backtesting_observations=250),
    )
    findings, gate_passed = GATE.run("non_credit_risk", inputs)
    assert len(findings) == 1
    assert findings[0].severity == "critical"
    assert gate_passed is False


def test_non_credit_operational_gate_material_loss():
    inputs = NonCreditRiskValidationInputs(
        risk_type="operational", business_unit="Ops", reporting_date="2026-06-30",
        operational_metrics=OperationalRiskMetrics(
            loss_event_count=1, gross_loss_amount=500000.0, recovery_amount=50000.0,
            event_category="internal_fraud",
        ),
    )
    findings, gate_passed = GATE.run("non_credit_risk", inputs)
    assert len(findings) == 1
    assert findings[0].severity == "high"
    assert gate_passed is True  # high, not critical


def test_non_credit_liquidity_gate_lcr_and_nsfr_breach():
    inputs = NonCreditRiskValidationInputs(
        risk_type="liquidity", business_unit="Treasury", reporting_date="2026-06-30",
        liquidity_metrics=LiquidityRiskMetrics(lcr_pct=80.0, nsfr_pct=90.0),
    )
    findings, gate_passed = GATE.run("non_credit_risk", inputs)
    areas = {f.area for f in findings}
    assert "Liquidity Coverage Ratio" in areas
    assert "Net Stable Funding Ratio" in areas
    assert gate_passed is False  # LCR breach is critical


def test_model_risk_gate_psi_and_gini_breach():
    inputs = ModelRiskValidationInputs(
        model_id="M1", model_name="IFRS9 ECL Retail", model_tier="tier_1_high_materiality",
        model_owner="Credit Risk",
        stability_metrics=ModelStabilityMetrics(psi=0.35, gini=0.20),
    )
    findings, gate_passed = GATE.run("model_risk", inputs)
    areas = {f.area for f in findings}
    assert "Stability testing" in areas
    assert "Outcomes analysis" in areas
    assert gate_passed is False


def test_model_risk_gate_no_stability_metrics_produces_no_findings():
    inputs = ModelRiskValidationInputs(
        model_id="M1", model_name="X", model_tier="tier_3_low_materiality", model_owner="Owner",
    )
    findings, gate_passed = GATE.run("model_risk", inputs)
    assert findings == []
    assert gate_passed is True


# ---------- Kupiec statistical test path (observations >= 30) ----------

def test_credit_gate_large_sample_uses_kupiec_test_and_flags_significant_excess():
    inputs = _credit_inputs(backtesting_exceptions_count=15, backtesting_observations_count=100)
    findings, _ = GATE.run("credit_risk", inputs)
    match = [f for f in findings if f.area == "Calibration and back-testing"]
    assert len(match) == 1
    assert "Kupiec POF test" in match[0].description


def test_credit_gate_large_sample_does_not_flag_marginal_excess():
    # observed rate slightly above threshold but not statistically significant
    # at this small a deviation with 100 observations.
    inputs = _credit_inputs(backtesting_exceptions_count=6, backtesting_observations_count=100)
    findings, _ = GATE.run("credit_risk", inputs)
    assert not any(f.area == "Calibration and back-testing" for f in findings)


def test_credit_gate_small_sample_falls_back_to_flat_rate():
    inputs = _credit_inputs(backtesting_exceptions_count=2, backtesting_observations_count=20)
    findings, _ = GATE.run("credit_risk", inputs)
    match = [f for f in findings if f.area == "Calibration and back-testing"]
    assert len(match) == 1
    assert "p-value" not in match[0].description
    assert "below" in match[0].description  # small-sample-size note present


def test_non_credit_market_gate_small_sample_falls_back_to_raw_count():
    inputs = NonCreditRiskValidationInputs(
        risk_type="market", business_unit="Trading", reporting_date="2026-06-30",
        market_metrics=MarketRiskMetrics(var_backtesting_exceptions=6, var_backtesting_observations=20),
    )
    findings, gate_passed = GATE.run("non_credit_risk", inputs)
    assert len(findings) == 1
    assert "p-value" not in findings[0].description


# ---------- concentration checks ----------

def test_credit_gate_flags_single_name_concentration():
    inputs = _credit_inputs(single_name_concentration_pct=8.0)
    findings, gate_passed = GATE.run("credit_risk", inputs)
    assert any(f.area == "Portfolio concentration risk" for f in findings)
    assert gate_passed is True  # high, not critical


def test_non_credit_liquidity_gate_flags_funding_concentration():
    inputs = NonCreditRiskValidationInputs(
        risk_type="liquidity", business_unit="Treasury", reporting_date="2026-06-30",
        liquidity_metrics=LiquidityRiskMetrics(lcr_pct=120.0, nsfr_pct=120.0, concentration_of_funding_pct=40.0),
    )
    findings, _ = GATE.run("non_credit_risk", inputs)
    assert any(f.area == "Funding concentration" for f in findings)


# ---------- IFRS9 staging consistency ----------

def test_credit_gate_flags_unevidenced_stage2_classification():
    inputs = _credit_inputs(
        model_type="IFRS9_ECL", ifrs9_stage="stage_2", sicr_trigger_flag=False, days_past_due=5,
    )
    findings, _ = GATE.run("credit_risk", inputs)
    assert any(f.area == "IFRS9 staging consistency" for f in findings)


def test_credit_gate_does_not_flag_stage2_with_sicr_trigger():
    inputs = _credit_inputs(
        model_type="IFRS9_ECL", ifrs9_stage="stage_2", sicr_trigger_flag=True,
    )
    findings, _ = GATE.run("credit_risk", inputs)
    assert not any(f.area == "IFRS9 staging consistency" for f in findings)


def test_credit_gate_does_not_flag_stage2_with_material_arrears():
    inputs = _credit_inputs(
        model_type="IFRS9_ECL", ifrs9_stage="stage_2", sicr_trigger_flag=False, days_past_due=45,
    )
    findings, _ = GATE.run("credit_risk", inputs)
    assert not any(f.area == "IFRS9 staging consistency" for f in findings)


# ---------- jurisdiction / tier threshold resolution ----------

def test_resolve_applies_jurisdiction_override_only_to_listed_fields():
    thresholds = ValidationThresholds(
        credit_max_psi=0.25, credit_min_gini=0.40,
        credit_jurisdiction_overrides={"HU": {"max_psi": 0.10}},
    )
    resolved = thresholds.resolve(jurisdiction="HU")
    assert resolved.credit_max_psi == 0.10
    assert resolved.credit_min_gini == 0.40  # untouched
    assert thresholds.resolve(jurisdiction="DE") is thresholds  # no override -> same instance


def test_resolve_applies_tier_override():
    thresholds = ValidationThresholds(model_max_psi=0.25, model_tier_overrides={"tier_1_high_materiality": {"max_psi": 0.10}})
    resolved = thresholds.resolve(tier="tier_1_high_materiality")
    assert resolved.model_max_psi == 0.10
    assert thresholds.resolve(tier="tier_3_low_materiality") is thresholds


def test_credit_gate_uses_jurisdiction_specific_threshold():
    thresholds = ValidationThresholds(credit_max_psi=0.25, credit_jurisdiction_overrides={"HU": {"max_psi": 0.10}})
    gate = ValidationGateAgent(thresholds)
    inputs = _credit_inputs(jurisdiction="HU", population_stability_index=0.15)
    findings, gate_passed = gate.run("credit_risk", inputs)
    # 0.15 is below the base 0.25 but above the HU-specific 0.10 override
    assert any(f.severity == "critical" for f in findings)
    assert gate_passed is False


def test_model_risk_gate_uses_tier_specific_threshold():
    thresholds = ValidationThresholds(model_max_psi=0.25, model_tier_overrides={"tier_1_high_materiality": {"max_psi": 0.10}})
    gate = ValidationGateAgent(thresholds)
    inputs = ModelRiskValidationInputs(
        model_id="M1", model_name="X", model_tier="tier_1_high_materiality", model_owner="Owner",
        stability_metrics=ModelStabilityMetrics(psi=0.15),
    )
    findings, gate_passed = gate.run("model_risk", inputs)
    assert any(f.severity == "critical" for f in findings)
    assert gate_passed is False


# ---------- trend-aware checks ----------

def _report_with_gini(gini: float) -> ValidationReport:
    return ValidationReport(
        domain="credit_risk", title="t", scope="s", methodology="m",
        entity_under_review="e", reporting_period="p", overall_rating="compliant",
        quantitative_results={"gini_coefficient": gini},
    )


def test_credit_gate_flags_declining_gini_trend_even_above_threshold():
    history = [_report_with_gini(g) for g in [0.60, 0.55, 0.50]]
    inputs = _credit_inputs(gini_coefficient=0.45)  # still well above min_gini=0.40
    findings, _ = GATE.run("credit_risk", inputs, history=history)
    assert any(f.area == "Discriminatory power" and "declined" in f.description for f in findings)


def test_credit_gate_does_not_flag_trend_with_insufficient_history():
    history = [_report_with_gini(g) for g in [0.60, 0.55]]  # only 2 prior cycles, need 3
    inputs = _credit_inputs(gini_coefficient=0.50)
    findings, _ = GATE.run("credit_risk", inputs, history=history)
    assert not any(f.area == "Discriminatory power" and "declined" in f.description for f in findings)


def test_credit_gate_does_not_flag_trend_when_not_monotonic():
    history = [_report_with_gini(g) for g in [0.50, 0.60, 0.55]]
    inputs = _credit_inputs(gini_coefficient=0.58)
    findings, _ = GATE.run("credit_risk", inputs, history=history)
    assert not any(f.area == "Discriminatory power" and "declined" in f.description for f in findings)


# ---------- structured regulatory reference ----------

def test_gate_findings_carry_a_structured_regulatory_reference():
    inputs = _credit_inputs(population_stability_index=0.30)
    findings, _ = GATE.run("credit_risk", inputs)
    psi = next(f for f in findings if f.finding_reference == "credit_risk:psi")
    assert psi.regulatory_reference.startswith("EBA/GL/2017/16")
    assert psi.regulatory_reference_structured is not None
    assert psi.remediation_deadline_days == 30  # critical -> 30d default policy


# ---------- PD calibration (Jeffreys / binomial) ----------

def test_pd_calibration_flags_optimistic_grade():
    grades = [
        RatingGradeObservation(grade="A", predicted_pd=0.005, obligors=1000, observed_defaults=50),
        RatingGradeObservation(grade="B", predicted_pd=0.02, obligors=500, observed_defaults=12),
    ]
    inputs = _credit_inputs(rating_grade_observations=grades)
    findings, _ = GATE.run("credit_risk", inputs)
    assert any(f.finding_reference == "credit_risk:pd-calibration" for f in findings)


def test_pd_calibration_passes_when_grades_are_well_calibrated():
    grades = [
        RatingGradeObservation(grade="A", predicted_pd=0.005, obligors=1000, observed_defaults=5),
        RatingGradeObservation(grade="B", predicted_pd=0.02, obligors=500, observed_defaults=10),
    ]
    inputs = _credit_inputs(rating_grade_observations=grades)
    findings, _ = GATE.run("credit_risk", inputs)
    assert not any(f.finding_reference == "credit_risk:pd-calibration" for f in findings)


# ---------- LGD back-testing / downturn LGD / MoC / data quality ----------

def test_lgd_backtesting_flags_underestimation_outside_tolerance():
    inputs = _credit_inputs(model_type="LGD", lgd_predicted_mean=0.30, lgd_observed_mean=0.55,
                            lgd_observation_count=400, downturn_lgd_applied=True, downturn_lgd_addon_pct=5.0)
    findings, _ = GATE.run("credit_risk", inputs)
    assert any(f.finding_reference == "credit_risk:lgd-backtesting" and f.severity == "high" for f in findings)


def test_downturn_lgd_missing_is_flagged_for_lgd_model():
    inputs = _credit_inputs(model_type="LGD", downturn_lgd_applied=False)
    findings, _ = GATE.run("credit_risk", inputs)
    assert any(f.area == "Downturn LGD estimation" for f in findings)


def test_downturn_lgd_not_flagged_for_pd_model_without_signal():
    inputs = _credit_inputs()  # model_type PD, nothing set
    findings, _ = GATE.run("credit_risk", inputs)
    assert not any(f.area == "Downturn LGD estimation" for f in findings)


def test_moc_missing_framework_is_high():
    inputs = _credit_inputs(moc_framework_documented=False)
    findings, _ = GATE.run("credit_risk", inputs)
    assert any(f.area == "Margin of Conservatism" and f.severity == "high" for f in findings)


def test_data_quality_shortfall_is_flagged():
    inputs = _credit_inputs(data_completeness_pct=80.0, historical_observation_period_years=3)
    findings, _ = GATE.run("credit_risk", inputs)
    dq = [f for f in findings if f.finding_reference == "credit_risk:data-quality"]
    assert len(dq) == 1 and dq[0].severity == "high"


def test_representativeness_not_assessed_is_medium():
    inputs = _credit_inputs(representativeness_assessed=False)
    findings, _ = GATE.run("credit_risk", inputs)
    assert any(f.finding_reference == "credit_risk:representativeness" and f.severity == "medium" for f in findings)


# ---------- model change materiality (Delegated Reg 529/2014) ----------

def test_material_model_change_without_pre_approval_is_critical_and_fails_gate():
    inputs = _credit_inputs(model_change_type="material", model_change_pre_approval_obtained=False)
    findings, gate_passed = GATE.run("credit_risk", inputs)
    assert any(f.area == "Model change management" and f.severity == "critical" for f in findings)
    assert gate_passed is False


def test_material_model_change_with_pre_approval_is_not_flagged():
    inputs = _credit_inputs(model_change_type="material", model_change_pre_approval_obtained=True)
    findings, gate_passed = GATE.run("credit_risk", inputs)
    assert not any(f.area == "Model change management" for f in findings)
    assert gate_passed is True


# ---------- governance & lifecycle ----------

def test_validation_function_not_independent_is_flagged():
    inputs = _credit_inputs(validation_function_independent=False)
    findings, _ = GATE.run("credit_risk", inputs)
    assert any(f.finding_reference == "credit_risk:validation-independence" for f in findings)


def test_revalidation_overdue_is_flagged_high():
    stale = (date.today() - timedelta(days=800)).isoformat()
    inputs = _credit_inputs(last_validation_date=stale)
    findings, _ = GATE.run("credit_risk", inputs)
    rc = [f for f in findings if f.finding_reference == "credit_risk:revalidation-cadence"]
    assert len(rc) == 1 and rc[0].severity == "high"


def test_revalidation_within_window_is_low_severity():
    soon_last = (date.today() - timedelta(days=330)).isoformat()  # due in ~35 days (365 cadence)
    inputs = _credit_inputs(last_validation_date=soon_last)
    findings, gate_passed = GATE.run("credit_risk", inputs)
    rc = [f for f in findings if f.finding_reference == "credit_risk:revalidation-cadence"]
    assert len(rc) == 1 and rc[0].severity == "low"
    assert gate_passed is True


def test_revalidation_not_checked_without_dates():
    inputs = _credit_inputs()
    findings, _ = GATE.run("credit_risk", inputs)
    assert not any(f.finding_reference == "credit_risk:revalidation-cadence" for f in findings)


def test_model_risk_revalidation_uses_tier_cadence():
    stale = (date.today() - timedelta(days=400)).isoformat()
    inputs = ModelRiskValidationInputs(
        model_id="M1", model_name="X", model_tier="tier_1_high_materiality", model_owner="Owner",
        last_validation_date=stale,
    )
    findings, _ = GATE.run("model_risk", inputs)
    assert any(f.finding_reference == "model_risk:revalidation-cadence" and f.severity == "high" for f in findings)


# ---------- FRTB market risk: ES / PLA / dual-level backtesting ----------

def _market_inputs(**mkw):
    return NonCreditRiskValidationInputs(
        risk_type="market", business_unit="Rates Desk", reporting_date="2026-06-30",
        market_metrics=MarketRiskMetrics(var_backtesting_exceptions=2, var_backtesting_observations=250, **mkw),
    )


def test_pla_red_zone_is_critical():
    inputs = _market_inputs(desk_id="RATES-1", pla_spearman_correlation=0.55, pla_ks_statistic=0.30)
    findings, gate_passed = GATE.run("non_credit_risk", inputs)
    assert any(f.area == "P&L attribution" and f.severity == "critical" for f in findings)
    assert gate_passed is False


def test_pla_amber_zone_is_medium():
    inputs = _market_inputs(desk_id="RATES-1", pla_spearman_correlation=0.74, pla_ks_statistic=0.11)
    findings, gate_passed = GATE.run("non_credit_risk", inputs)
    assert any(f.area == "P&L attribution" and f.severity == "medium" for f in findings)
    assert gate_passed is True


def test_expected_shortfall_missing_on_frtb_submission_is_medium():
    inputs = _market_inputs(desk_id="RATES-1", pla_spearman_correlation=0.95, pla_ks_statistic=0.03)
    findings, _ = GATE.run("non_credit_risk", inputs)
    assert any(f.area == "Expected shortfall" and f.severity == "medium" for f in findings)


def test_expected_shortfall_not_required_for_legacy_market_payload():
    inputs = _market_inputs()  # no FRTB fields
    findings, _ = GATE.run("non_credit_risk", inputs)
    assert not any(f.area == "Expected shortfall" for f in findings)


def test_dual_level_backtesting_flags_worse_975_series():
    inputs = NonCreditRiskValidationInputs(
        risk_type="market", business_unit="Rates Desk", reporting_date="2026-06-30",
        market_metrics=MarketRiskMetrics(
            var_backtesting_exceptions=2, var_backtesting_observations=250,
            var_backtesting_exceptions_975=20, var_backtesting_observations_975=250,
            expected_shortfall_975=0.031,
        ),
    )
    findings, gate_passed = GATE.run("non_credit_risk", inputs)
    var_findings = [f for f in findings if f.area == "VaR backtesting"]
    assert any("97.5%" in f.description and f.severity == "critical" for f in var_findings)
    assert gate_passed is False


# ---------- prior-findings carry-forward ----------

def test_overdue_prior_finding_is_escalated():
    old = ValidationReport(
        domain="credit_risk", title="t", scope="s", methodology="m",
        entity_under_review="e", reporting_period="2025Q1", overall_rating="non_compliant",
        generated_at=(date.today() - timedelta(days=400)).isoformat() + "T00:00:00+00:00",
    )
    from agents.risk_schemas import ValidationFinding
    old.findings = [ValidationFinding(
        domain="credit_risk", area="Discriminatory power", verdict="non_compliant", severity="high",
        description="Gini below the configured minimum in the prior cycle.",
        finding_reference="credit_risk:gini", finding_status="open", remediation_deadline_days=90,
    )]
    inputs = _credit_inputs(gini_coefficient=0.55)  # currently fine
    findings, gate_passed = GATE.run("credit_risk", inputs, history=[old])
    overdue = [f for f in findings if "overdue" in f.finding_reference]
    assert len(overdue) == 1 and overdue[0].severity == "critical"  # high -> critical escalation
    assert gate_passed is False
