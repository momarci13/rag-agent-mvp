"""Property-based tests for ValidationGateAgent using hypothesis.

Sweeps threshold boundaries automatically rather than relying only on the
hand-picked boundary cases in tests/test_risk_validation_gates.py.
"""

from hypothesis import given, settings
from hypothesis import strategies as st

from agents.risk_schemas import (
    CreditModelValidationInputs,
    LiquidityRiskMetrics,
    MarketRiskMetrics,
    NonCreditRiskValidationInputs,
)
from studio.regulatory_gates import ValidationGateAgent, ValidationThresholds
from tools.statistical_tests import traffic_light_zone

THRESHOLDS = ValidationThresholds()
GATE = ValidationGateAgent(THRESHOLDS)

_finite_float = lambda **kw: st.floats(allow_nan=False, allow_infinity=False, **kw)


def _credit_inputs(**overrides):
    base = dict(
        model_id="PD-1", model_type="PD", exposure_class="retail",
        portfolio_segment="mortgages", estimation_approach="internal_ratings_based",
    )
    base.update(overrides)
    return CreditModelValidationInputs(**base)


@given(psi=_finite_float(min_value=0.0, max_value=1.0))
@settings(max_examples=50)
def test_property_psi_above_threshold_always_critical(psi):
    inputs = _credit_inputs(population_stability_index=psi)
    findings, gate_passed = GATE.run("credit_risk", inputs)
    breach = psi > THRESHOLDS.credit_max_psi
    critical_found = any(f.severity == "critical" and f.area == "Calibration and back-testing" for f in findings)
    assert critical_found == breach
    assert gate_passed == (not breach)


@given(gini=_finite_float(min_value=-1.0, max_value=1.0))
@settings(max_examples=50)
def test_property_gini_below_threshold_always_high(gini):
    inputs = _credit_inputs(gini_coefficient=gini)
    findings, _ = GATE.run("credit_risk", inputs)
    breach = gini < THRESHOLDS.credit_min_gini
    high_found = any(f.severity == "high" and f.area == "Discriminatory power" for f in findings)
    assert high_found == breach


@given(concentration=_finite_float(min_value=0.0, max_value=100.0))
@settings(max_examples=50)
def test_property_concentration_above_threshold_always_flagged(concentration):
    inputs = _credit_inputs(single_name_concentration_pct=concentration)
    findings, _ = GATE.run("credit_risk", inputs)
    breach = concentration > THRESHOLDS.credit_max_single_name_concentration_pct
    flagged = any(f.area == "Portfolio concentration risk" for f in findings)
    assert flagged == breach


@given(lcr=_finite_float(min_value=0.0, max_value=300.0), nsfr=_finite_float(min_value=0.0, max_value=300.0))
@settings(max_examples=50)
def test_property_lcr_and_nsfr_breaches_are_independent(lcr, nsfr):
    inputs = NonCreditRiskValidationInputs(
        risk_type="liquidity", business_unit="Treasury", reporting_date="2026-06-30",
        liquidity_metrics=LiquidityRiskMetrics(lcr_pct=lcr, nsfr_pct=nsfr),
    )
    findings, gate_passed = GATE.run("non_credit_risk", inputs)
    lcr_breach = lcr < THRESHOLDS.liquidity_min_lcr_pct
    nsfr_breach = nsfr < THRESHOLDS.liquidity_min_nsfr_pct
    assert any(f.area == "Liquidity Coverage Ratio" for f in findings) == lcr_breach
    assert any(f.area == "Net Stable Funding Ratio" for f in findings) == nsfr_breach
    # LCR breaches are critical -- gate fails whenever LCR breaches, regardless of NSFR.
    assert gate_passed == (not lcr_breach)


@given(
    exceptions=st.integers(min_value=0, max_value=50),
    observations=st.integers(min_value=30, max_value=250),
)
@settings(max_examples=50)
def test_property_market_gate_matches_traffic_light_zone(exceptions, observations):
    exceptions = min(exceptions, observations)  # exceptions can't exceed observations
    inputs = NonCreditRiskValidationInputs(
        risk_type="market", business_unit="Trading", reporting_date="2026-06-30",
        market_metrics=MarketRiskMetrics(var_backtesting_exceptions=exceptions, var_backtesting_observations=observations),
    )
    findings, gate_passed = GATE.run("non_credit_risk", inputs)
    zone = traffic_light_zone(exceptions, observations, 0.01)

    if zone == "green":
        assert findings == []
        assert gate_passed is True
    elif zone == "red":
        assert any(f.severity == "critical" for f in findings)
        assert gate_passed is False
    else:  # yellow
        assert any(f.severity == "medium" for f in findings)
        assert gate_passed is True


@given(
    findings_present=st.booleans(),
    psi=_finite_float(min_value=0.0, max_value=1.0),
)
@settings(max_examples=30)
def test_property_gate_passed_iff_no_critical_finding(findings_present, psi):
    inputs = _credit_inputs(population_stability_index=psi if findings_present else None)
    findings, gate_passed = GATE.run("credit_risk", inputs)
    assert gate_passed == (not any(f.severity == "critical" for f in findings))


@given(pre_approved=st.sampled_from([True, False, None]))
@settings(max_examples=10)
def test_property_material_change_without_pre_approval_is_always_critical(pre_approved):
    inputs = _credit_inputs(model_change_type="material", model_change_pre_approval_obtained=pre_approved)
    findings, gate_passed = GATE.run("credit_risk", inputs)
    flagged = any(f.area == "Model change management" and f.severity == "critical" for f in findings)
    assert flagged == (pre_approved is not True)
    if pre_approved is not True:
        assert gate_passed is False


@given(
    spearman=_finite_float(min_value=-1.0, max_value=1.0),
    ks=_finite_float(min_value=0.0, max_value=1.0),
)
@settings(max_examples=50)
def test_property_pla_red_zone_is_always_critical(spearman, ks):
    from tools.statistical_tests import pla_test

    inputs = NonCreditRiskValidationInputs(
        risk_type="market", business_unit="Desk", reporting_date="2026-06-30",
        market_metrics=MarketRiskMetrics(
            var_backtesting_exceptions=1, var_backtesting_observations=250,
            desk_id="D1", expected_shortfall_975=0.03,
            pla_spearman_correlation=spearman, pla_ks_statistic=ks,
        ),
    )
    findings, gate_passed = GATE.run("non_credit_risk", inputs)
    zone = pla_test(spearman, ks)
    pla_findings = [f for f in findings if f.area == "P&L attribution"]
    if zone == "red":
        assert pla_findings and pla_findings[0].severity == "critical"
        assert gate_passed is False
    elif zone == "amber":
        assert pla_findings and pla_findings[0].severity == "medium"
    else:
        assert not pla_findings
