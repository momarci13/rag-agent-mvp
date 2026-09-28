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

# Deterministically derived from overall_rating + gate_passed (see
# agents/risk_validation_team.py::_derive_recommendation). Kept deliberately
# conservative: the LLM never sets this -- a draft support tool does not issue
# an approve/reject regulatory conclusion.
ValidationRecommendation = Literal[
    "approve", "approve_with_conditions", "reject", "not_a_recommendation"
]

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


class RegulatoryReference(BaseModel):
    """Structured citation, for once a real taxonomy/DPM has been ingested to
    replace data/regulatory/'s placeholders. All fields optional since most
    findings will only populate a subset (e.g. a paragraph reference but no
    template row/column)."""

    template_code: str | None = None
    row: str | None = None
    column: str | None = None
    paragraph: str | None = None
    note: str | None = None


FindingStatus = Literal["open", "closed"]


class ValidationFinding(BaseModel):
    finding_id: str = Field(default_factory=lambda: str(uuid4()))
    domain: RiskDomain
    area: str = Field(description="Validation area, e.g. 'PD Model Calibration'")
    regulatory_reference: str = Field(
        default="", description="Placeholder-style citation, e.g. 'EBA/GL/2017/11 Title IV'"
    )
    regulatory_reference_structured: RegulatoryReference | None = None
    verdict: ValidationVerdict
    severity: Severity
    description: str = Field(min_length=10)
    evidence: list[str] = Field(default_factory=list, description="RAG source IDs")
    recommendation: str = ""
    remediation_deadline_days: int | None = None
    owner: str = ""
    # Remediation tracking (open/closed carried forward across validation
    # cycles) and LLM-output-quality flags -- see agents/risk_roles.py.
    finding_status: FindingStatus = "open"
    # Stable slug (default "{domain}:{area}") used to match a finding to the
    # same issue in a prior validation cycle -- see
    # agents/risk_validation_team.py::ValidationGateAgent._check_prior_findings.
    finding_reference: str = ""
    first_raised_period: str = ""
    prior_finding_id: str | None = None
    evidence_grounded: bool | None = Field(
        default=None,
        description="False when cited evidence text has low overlap with the finding's "
        "description (informational plausibility heuristic, not a hard block).",
    )
    secondary_review_flag: bool = Field(
        default=False,
        description="Set when a fallback-tier model disagreed with a critical/high finding "
        "during cross-model verification; flags it for extra human attention.",
    )


class SignoffRecord(BaseModel):
    by: str
    role: str = ""
    at: str = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())


class PriorFindingStatus(BaseModel):
    """Status, in the current cycle, of a finding raised in a previous
    validation cycle -- the 'follow-up on previous findings' section a
    supervisory validation report is expected to carry."""

    finding_reference: str
    area: str = ""
    status: Literal["resolved", "open", "overdue"]
    note: str = ""


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
    # Deterministically derived (agents/risk_validation_team.py::
    # _derive_recommendation) from overall_rating + gate_passed -- NOT written
    # by the LLM. Defaults to "not_a_recommendation" so a draft never reads as
    # a supervisory approval decision.
    recommendation: ValidationRecommendation = "not_a_recommendation"
    conditions: list[str] = Field(
        default_factory=list,
        description="Concrete conditions attached to an 'approve_with_conditions' recommendation.",
    )
    follow_up_on_prior_findings: list[PriorFindingStatus] = Field(default_factory=list)
    validation_sample: str = Field(
        default="", description="Data/sample used: reference dates, volumes, exclusions."
    )
    materiality_rationale: str = Field(
        default="", description="Why this model tier / validation scope was applied."
    )
    deviations_from_policy: list[str] = Field(default_factory=list)
    preparer: str = Field(
        default="",
        description="Person/role that prepared the case file. A sign-off by this "
        "same identity is rejected (preparer must differ from validator).",
    )
    prepared_by: str = "AI Risk Validation Agent (draft)"
    requires_human_signoff: bool = True
    # Multi-level ("four-eyes") sign-off: the report only becomes final once
    # len(signoffs) >= required_signoffs. signed_off_by/signed_off_at are set
    # to the LAST signoff once that threshold is met, kept for backward
    # compatibility with callers that only look at a single sign-off pair.
    required_signoffs: int = 1
    signoffs: list[SignoffRecord] = Field(default_factory=list)
    signed_off_by: str | None = None
    signed_off_at: str | None = None
    generated_at: str = Field(default_factory=lambda: dt.datetime.now(dt.timezone.utc).isoformat())
    disclaimer: str = DISCLAIMER_TEXT


class RiskValidationRun(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    domain: RiskDomain
    inputs: dict[str, Any] = Field(default_factory=dict)
    preparer: str = Field(
        default="",
        description="Identity that prepared the case file; propagated to the "
        "report and enforced as distinct from every sign-off identity.",
    )
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

# EBA/GL/2017/16-style IRB validation checklist (PD/LGD estimation and the
# treatment of defaulted exposures). Shared by the drafting prompt
# (agents/risk_roles.py) and the deterministic gate so both reference the
# same canonical list of areas. The area strings are mapped to regulatory
# instruments in agents/regulatory_refs.py::AREA_TO_REFERENCE.
CREDIT_VALIDATION_AREAS: list[str] = [
    "Conceptual soundness",
    "Data quality",
    "Discriminatory power",
    "Calibration and back-testing",
    "Margin of Conservatism",
    "Downturn LGD estimation",
    "Override analysis",
    "Model change management",
    "IT implementation",
    "Use test",
    "Ongoing monitoring",
]

# Materiality classification of a model change/extension in the sense of
# Commission Delegated Regulation (EU) No 529/2014.
ModelChangeType = Literal["none", "non_material", "material"]


class RatingGradeObservation(BaseModel):
    """One rating grade's realised outcome for PD calibration back-testing
    (predicted PD vs observed default rate at grade level)."""

    grade: str
    predicted_pd: float = Field(ge=0.0, le=1.0)
    obligors: int = Field(ge=0)
    observed_defaults: int = Field(ge=0)


class CreditModelValidationInputs(BaseModel):
    model_id: str
    model_type: CreditRiskModelType
    exposure_class: ExposureClass
    portfolio_segment: str
    estimation_approach: Literal["internal_ratings_based", "standardised", "hybrid"]
    jurisdiction: str = Field(
        default="EU", description="ISO-ish jurisdiction code used to resolve threshold overrides, e.g. 'EU', 'HU'"
    )
    last_recalibration_date: str | None = None
    # Model lifecycle / governance (EBA GL / ECB EGIM general topics).
    last_validation_date: str | None = None
    next_scheduled_validation_date: str | None = None
    validation_function_independent: bool | None = Field(
        default=None,
        description="Attestation that the validation function is independent of model development.",
    )
    population_stability_index: float | None = None
    gini_coefficient: float | None = None
    ks_statistic: float | None = None
    backtesting_exceptions_count: int | None = None
    backtesting_observations_count: int | None = None
    # PD calibration back-testing at rating-grade level (Jeffreys / binomial).
    rating_grade_observations: list[RatingGradeObservation] | None = None
    # LGD / EAD(CCF) back-testing: predicted vs realised means.
    lgd_predicted_mean: float | None = None
    lgd_observed_mean: float | None = None
    lgd_observation_count: int | None = None
    ccf_predicted_mean: float | None = None
    ccf_observed_mean: float | None = None
    ccf_observation_count: int | None = None
    # Downturn LGD (EBA/GL/2019/03).
    downturn_lgd_applied: bool | None = None
    downturn_lgd_addon_pct: float | None = None
    # Margin of Conservatism framework (EBA/GL/2017/16 section 4.4).
    moc_framework_documented: bool | None = None
    moc_total_pct: float | None = None
    moc_category_a_pct: float | None = Field(
        default=None, description="MoC for data/methodological deficiencies (category A)."
    )
    moc_category_b_pct: float | None = Field(
        default=None, description="MoC for relevant changes / general estimation error (category B)."
    )
    # Data quality (EBA/GL/2017/16, BCBS 239).
    data_completeness_pct: float | None = None
    data_accuracy_pct: float | None = None
    historical_observation_period_years: float | None = None
    data_deficiencies_count: int | None = None
    # Representativeness of the development sample vs the application portfolio.
    representativeness_assessed: bool | None = None
    application_vs_development_psi: float | None = None
    # Model change management (Delegated Regulation (EU) 529/2014).
    model_change_type: ModelChangeType | None = None
    model_change_pre_approval_obtained: bool | None = None
    override_rate_pct: float | None = None
    single_name_concentration_pct: float | None = Field(
        default=None, description="Largest single-name exposure as % of portfolio/segment"
    )
    # IFRS 9 staging (only meaningful when model_type == "IFRS9_ECL")
    ifrs9_stage: Literal["stage_1", "stage_2", "stage_3"] | None = None
    sicr_trigger_flag: bool | None = Field(
        default=None, description="Whether a Significant Increase in Credit Risk trigger fired"
    )
    days_past_due: int | None = None


# ---------- Non-credit risk validation (market / operational / liquidity) ----------

NonCreditRiskType = Literal["market", "operational", "liquidity"]


class MarketRiskMetrics(BaseModel):
    # Existing fields are the 99% one-day VaR back-testing series.
    var_confidence_level: float = 0.99
    var_horizon_days: int = 1
    var_backtesting_exceptions: int
    var_backtesting_observations: int
    traffic_light_zone: Literal["green", "yellow", "red"] | None = None
    stressed_var: float | None = None
    desk_id: str | None = Field(
        default=None, description="Trading desk under review (FRTB back-testing / PLA are per-desk)."
    )
    # FRTB (BCBS d457 / CRR Art. 325bf-325bg): Expected Shortfall at 97.5%.
    expected_shortfall_975: float | None = None
    es_horizon_days: int = 10
    stressed_es: float | None = None
    # Optional second back-testing series at 97.5% (run alongside the 99% one).
    var_backtesting_exceptions_975: int | None = None
    var_backtesting_observations_975: int | None = None
    # P&L attribution test inputs: either the summary statistics directly, or
    # the raw daily series for the gate helper to compute Spearman + KS from.
    pla_spearman_correlation: float | None = None
    pla_ks_statistic: float | None = None
    hypothetical_pnl: list[float] | None = None
    risk_theoretical_pnl: list[float] | None = None


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
    jurisdiction: str = "EU"
    last_validation_date: str | None = None
    next_scheduled_validation_date: str | None = None
    validation_function_independent: bool | None = None
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
    jurisdiction: str = "EU"
    activities_performed: list[ModelValidationActivity] = Field(default_factory=list)
    stability_metrics: ModelStabilityMetrics | None = None
    benchmarking_results: str | None = None
    last_validation_date: str | None = None
    next_scheduled_validation_date: str | None = None
    validation_function_independent: bool | None = Field(
        default=None,
        description="Attestation that internal validation is independent of model development "
        "(ECB guide to internal models, general topics).",
    )
