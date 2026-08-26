"""Pure unit tests for the deterministic ValidationGateAgent.

No LLM/network mocking needed -- these are plain-Python threshold checks,
mirroring tests/test_risk.py's direct-math-assertion style.
"""

from agents.risk_schemas import (
    CreditModelValidationInputs,
    LiquidityRiskMetrics,
    MarketRiskMetrics,
    ModelRiskValidationInputs,
    ModelStabilityMetrics,
    NonCreditRiskValidationInputs,
    OperationalRiskMetrics,
)
from agents.risk_validation_team import ValidationGateAgent, ValidationThresholds

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
