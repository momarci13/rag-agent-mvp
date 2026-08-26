"""Typed schemas for the bank risk-validation agent team.

DRAFT / SUPPORT TOOL ONLY -- READ BEFORE USE.

Every object defined in this module represents a *draft* produced for a
human risk validator to review, not a certified or compliant regulatory
deliverable. In particular:

- The regulatory structures referenced here (COREP/FINREP-style template
  numbering, EBA GL 2017-11 headings, ECB TRIM assessment areas, generic MNB
  circular conventions) are placeholder/representative structures inferred
  from public naming conventions. They are NOT verified against actual
  source PDFs of the regulations/guidelines and must be reviewed against real
  regulatory source documents before any real use.
- The thresholds used by the deterministic validation gates
  (agents/risk_validation_team.py::ValidationGateAgent) are illustrative
  defaults sourced from configs/config.yaml's ``risk_validation`` block, not
  bank-approved or regulator-approved risk appetite limits.
- No output produced from these schemas may be submitted to ECB/EBA/MNB, or
  used as a final regulatory deliverable, without human validator sign-off.

See ``ValidationReport.disclaimer`` for the text baked into every generated
report/deck.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

DISCLAIMER_TEXT = (
    "DRAFT -- For internal validation review only, not a regulatory "
    "submission. This report was produced by an AI-assisted drafting tool. "
    "Regulatory references and thresholds are placeholder/illustrative and "
    "have not been verified against source ECB/EBA/MNB documents. It must be "
    "reviewed and signed off by a qualified human risk validator before any "
    "internal or external use."
)


# ---------- Shared envelope (domain-agnostic) ----------

Severity = Literal["critical", "high", "medium", "low", "observation"]
ValidationVerdict = Literal[
    "compliant", "partially_compliant", "non_compliant", "not_applicable"
]
RiskDomain = Literal["credit_risk", "non_credit_risk", "model_risk"]

_SEVERITY_RANK: dict[str, int] = {
    "critical": 4,
    "high": 3,
    "medium": 2,
    "low": 1,
    "observation": 0,
}


class AgentTrace(BaseModel):
    """One step in a risk-validation run, kept independent of the quant
    team's AgentTrace so this domain has no dependency on agents/quant_team.py."""

    agent: str
    status: str
    summary: str


class ValidationFinding(BaseModel):
    finding_id: str = Field(default_factory=lambda: str(uuid4()))
    domain: RiskDomain
    area: str = Field(description="Validation area, e.g. 'PD Model Calibration'")
    regulatory_reference: str = Field(
        default="", description="Placeholder-style citation, e.g. 'EBA/GL/2017/11 Title IV'"
    )
    verdict: ValidationVerdict
    severity: Severity
    description: str = Field(min_length=10)
    evidence: list[str] = Field(default_factory=list, description="RAG source IDs")
    recommendation: str = ""
    remediation_deadline_days: int | None = None
    owner: str = ""


class ValidationReport(BaseModel):
    report_id: str = Field(default_factory=lambda: str(uuid4()))
    domain: RiskDomain
    title: str
    scope: str
    methodology: str
    entity_under_review: str = Field(
        description="Model name, portfolio segment, or exposure class under review"
    )
    reporting_period: str
    findings: list[ValidationFinding] = Field(default_factory=list)
    quantitative_results: dict[str, Any] = Field(default_factory=dict)
    # Kept as a plain string rather than a shared Literal: credit/non-credit
    # domains rate compliant/partially_compliant/non_compliant while model
    # risk convention (ECB TRIM style) rates low/medium/high/unacceptable.
    # Each specialist agent validates against its own domain-appropriate
    # enum before constructing the report.
    overall_rating: str
    overall_conclusion: str = ""
    prepared_by: str = "AI Risk Validation Agent (draft)"
    requires_human_signoff: bool = True
    signed_off_by: str | None = None
    signed_off_at: str | None = None
    generated_at: str = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    disclaimer: str = DISCLAIMER_TEXT


class RiskValidationRun(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    domain: RiskDomain
    inputs: dict[str, Any] = Field(default_factory=dict)
    findings: list[ValidationFinding] = Field(default_factory=list)
    gate_passed: bool = False
    report: ValidationReport | None = None
    trace: list[AgentTrace] = Field(default_factory=list)
    # Sign-off token gating report finalisation/export -- issued by
    # RiskValidationOrchestrator.run() against the draft report, consumed by
    # RiskValidationOrchestrator.execute(). Never a substitute for the bank's
    # actual four-eyes/committee sign-off process.
    approval_token: str | None = None
    approval_expires_at_epoch: float | None = None


def worst_severity(findings: list[ValidationFinding]) -> Severity | None:
    """Return the most severe rating present, or ``None`` if there are no findings."""

    if not findings:
        return None
    return max(findings, key=lambda f: _SEVERITY_RANK[f.severity]).severity


# ---------- Credit risk validation ----------

ExposureClass = Literal[
    "retail", "corporate", "institutions", "sovereign", "equity", "securitisation", "other"
]
CreditRiskModelType = Literal["PD", "LGD", "EAD", "rating_scorecard", "IFRS9_ECL"]

# EBA GL 2017-11-style validation checklist. Shared by the drafting prompt
# (agents/risk_roles.py) and the deterministic gate so both reference the
# same canonical list of areas.
CREDIT_VALIDATION_AREAS: list[str] = [
    "Conceptual soundness",
    "Data quality",
    "Discriminatory power",
    "Calibration and back-testing",
    "Override analysis",
    "IT implementation",
    "Use test",
    "Ongoing monitoring",
]


class CreditModelValidationInputs(BaseModel):
    model_id: str
    model_type: CreditRiskModelType
    exposure_class: ExposureClass
    portfolio_segment: str
    estimation_approach: Literal["internal_ratings_based", "standardised", "hybrid"]
    last_recalibration_date: str | None = None
    population_stability_index: float | None = None
    gini_coefficient: float | None = None
    ks_statistic: float | None = None
    backtesting_exceptions_count: int | None = None
    backtesting_observations_count: int | None = None
    override_rate_pct: float | None = None


# ---------- Non-credit risk validation (market / operational / liquidity) ----------

NonCreditRiskType = Literal["market", "operational", "liquidity"]


class MarketRiskMetrics(BaseModel):
    var_confidence_level: float = 0.99
    var_horizon_days: int = 1
    var_backtesting_exceptions: int
    var_backtesting_observations: int
    traffic_light_zone: Literal["green", "yellow", "red"] | None = None
    stressed_var: float | None = None


class OperationalRiskMetrics(BaseModel):
    loss_event_count: int
    gross_loss_amount: float
    recovery_amount: float = 0.0
    event_category: str = Field(
        description="Basel-style event type, e.g. 'internal_fraud', 'external_fraud', 'execution_delivery'"
    )
    control_effectiveness_rating: Literal["effective", "partially_effective", "ineffective"] | None = None


class LiquidityRiskMetrics(BaseModel):
    lcr_pct: float | None = None
    nsfr_pct: float | None = None
    survival_horizon_days: int | None = None
    concentration_of_funding_pct: float | None = None


class NonCreditRiskValidationInputs(BaseModel):
    risk_type: NonCreditRiskType
    business_unit: str
    reporting_date: str
    market_metrics: MarketRiskMetrics | None = None
    operational_metrics: OperationalRiskMetrics | None = None
    liquidity_metrics: LiquidityRiskMetrics | None = None


# ---------- Model risk validation (banking model-risk-management sense --
# distinct from agents/quant_team.py's trading-model risk gates) ----------

ModelTier = Literal["tier_1_high_materiality", "tier_2_medium_materiality", "tier_3_low_materiality"]

ModelValidationActivity = Literal[
    "conceptual_soundness_review",
    "data_quality_assessment",
    "outcomes_analysis",
    "benchmarking",
    "sensitivity_analysis",
    "stability_testing",
    "implementation_testing",
    "ongoing_monitoring_review",
]

ModelRiskRating = Literal["low", "medium", "high", "unacceptable"]


class ModelStabilityMetrics(BaseModel):
    psi: float | None = None
    gini: float | None = None
    ks_statistic: float | None = None
    psi_threshold_breached: bool | None = None


class ModelRiskValidationInputs(BaseModel):
    model_id: str
    model_name: str
    model_tier: ModelTier
    model_owner: str
    activities_performed: list[ModelValidationActivity] = Field(default_factory=list)
    stability_metrics: ModelStabilityMetrics | None = None
    benchmarking_results: str | None = None
    last_validation_date: str | None = None
    next_scheduled_validation_date: str | None = None
