"""Bank Risk Validation Department agent team.

Mirrors agents/quant_team.py's shape: an explicit sequence of typed agents
with Pydantic handoffs, orchestrated by a single class that appends an
:class:`~agents.risk_schemas.AgentTrace` per step for auditability.

DRAFT / SUPPORT TOOL ONLY -- see agents/risk_schemas.py's module docstring
for the disclaimer that ships with every generated report. In particular:
the LLM never decides whether a quantitative threshold has been breached --
:class:`ValidationGateAgent` does that in plain Python from configured
thresholds, exactly like agents/quant_team.py::QuantRiskAgent's "LLM
proposes, code enforces" split. And no report may be treated as final
without a human validator consuming the sign-off token issued by
:meth:`RiskValidationOrchestrator.run` and completing
:meth:`RiskValidationOrchestrator.execute`.

The gate's structure follows current EBA/ECB expectations at the level of
*validation areas and statistical tests* -- EBA/GL/2017/16 (PD/LGD
estimation), EBA/GL/2019/03 (downturn LGD), Delegated Regulation (EU)
529/2014 (model changes), the ECB guide to internal models, BCBS d457 (FRTB
market risk) and BCBS 239 (risk data) -- but every threshold is an
illustrative placeholder and every citation is a drafting aid, not a
verified legal cross-reference (agents/regulatory_refs.py).
"""
from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from tools.risk_storage import RiskStorage
from tools.statistical_tests import (
    backtesting_plus_factor,
    binomial_pd_test,
    jeffreys_test,
    kupiec_pof_test,
    normal_mean_test,
    pla_statistics,
    pla_test,
    traffic_light_zone,
)

from . import risk_roles
from .llm import HostedLLM
from .quant_team import RAGAgent  # generic; reused as-is
from .regulatory_refs import citation_text, reference_for
from .risk_schemas import (
    AgentTrace,
    CreditModelValidationInputs,
    ModelRiskValidationInputs,
    NonCreditRiskValidationInputs,
    PriorFindingStatus,
    RiskDomain,
    RiskValidationRun,
    ValidationFinding,
    ValidationReport,
    worst_severity,
)

RiskCaseInputs = CreditModelValidationInputs | NonCreditRiskValidationInputs | ModelRiskValidationInputs

# Kupiec-style statistical tests are only meaningful with enough observations;
# below this, fall back to the flat exception-rate/count comparison.
_MIN_OBSERVATIONS_FOR_STATISTICAL_TEST = 30

# Minimum number of consecutive validation cycles (current + prior) required
# before a monotonic trend finding is raised.
_TREND_MIN_CYCLES = 4

# One-notch severity escalation for an overdue prior-cycle finding.
_SEVERITY_ESCALATION = {"medium": "high", "high": "critical", "critical": "critical"}

_MARKET_REC_RED = "Escalate to model risk committee; review VaR model assumptions."
_MARKET_REC_MONITOR = "Monitor closely; document rationale for each exception."

# Maps a jurisdiction/tier override block's raw config keys (as written in
# configs/config.yaml's risk_validation.credit_risk.jurisdictions.<CODE> /
# risk_validation.model_risk.tiers.<tier>) to the matching ValidationThresholds
# field name.
_CREDIT_OVERRIDE_FIELD_MAP = {
    "max_psi": "credit_max_psi",
    "min_gini": "credit_min_gini",
    "max_backtesting_exception_rate": "credit_max_backtesting_exception_rate",
    "max_override_rate_pct": "credit_max_override_rate_pct",
    "max_single_name_concentration_pct": "credit_max_single_name_concentration_pct",
    "pd_calibration_significance": "credit_pd_calibration_significance",
    "min_data_completeness_pct": "credit_min_data_completeness_pct",
    "min_data_accuracy_pct": "credit_min_data_accuracy_pct",
    "min_observation_period_years": "credit_min_observation_period_years",
    "revalidation_days": "credit_revalidation_days",
}
_MODEL_TIER_OVERRIDE_FIELD_MAP = {
    "max_psi": "model_max_psi",
    "min_gini": "model_min_gini",
}


def _default_tier_revalidation_days() -> dict[str, int]:
    return {
        "tier_1_high_materiality": 365,
        "tier_2_medium_materiality": 730,
        "tier_3_low_materiality": 1095,
    }


def _default_remediation_deadline_days() -> dict[str, int]:
    return {"critical": 30, "high": 90, "medium": 180, "low": 365, "observation": 365}


@dataclass(frozen=True)
class ValidationThresholds:
    """Illustrative default thresholds -- see configs/config.yaml's
    ``risk_validation`` block. Must be replaced with the bank's actual
    validation-policy-owned limits before any real use."""

    credit_max_psi: float = 0.25
    credit_min_gini: float = 0.40
    credit_max_backtesting_exception_rate: float = 0.05
    credit_backtesting_significance_level: float = 0.05
    credit_pd_calibration_significance: float = 0.05
    credit_lgd_backtesting_significance: float = 0.05
    credit_lgd_backtesting_tolerance: float = 0.10
    credit_min_moc_pct: float = 0.0
    credit_min_data_completeness_pct: float = 95.0
    credit_min_data_accuracy_pct: float = 98.0
    credit_min_observation_period_years: float = 5.0
    credit_revalidation_days: int = 365
    credit_revalidation_warning_days: int = 60
    credit_max_override_rate_pct: float = 0.10
    credit_max_single_name_concentration_pct: float = 5.0
    market_max_var_exceptions_250d: int = 4
    market_max_var_exceptions_red_250d: int = 9
    market_kupiec_significance_level: float = 0.05
    market_es_confidence_level: float = 0.975
    market_require_expected_shortfall: bool = True
    market_pla_spearman_green: float = 0.80
    market_pla_spearman_amber: float = 0.70
    market_pla_ks_green: float = 0.09
    market_pla_ks_amber: float = 0.12
    operational_material_loss_threshold: float = 100000.0
    liquidity_min_lcr_pct: float = 100.0
    liquidity_min_nsfr_pct: float = 100.0
    liquidity_max_funding_concentration_pct: float = 25.0
    model_max_psi: float = 0.25
    model_min_gini: float = 0.40
    model_revalidation_warning_days: int = 60
    model_tier_revalidation_days: dict[str, int] = field(default_factory=_default_tier_revalidation_days)
    remediation_deadline_days: dict[str, int] = field(default_factory=_default_remediation_deadline_days)
    # Raw override maps merged on top of the flat fields above by resolve().
    # Keys are jurisdiction codes / model tiers; values are the matching
    # raw sub-dicts straight out of configs/config.yaml.
    credit_jurisdiction_overrides: dict[str, dict[str, float]] = field(default_factory=dict)
    model_tier_overrides: dict[str, dict[str, float]] = field(default_factory=dict)

    def resolve(self, *, jurisdiction: str | None = None, tier: str | None = None) -> "ValidationThresholds":
        """Return a copy with jurisdiction/tier-specific overrides merged in.
        Unlisted fields fall back to this instance's (base) values."""
        overrides: dict[str, float] = {}
        if jurisdiction and jurisdiction in self.credit_jurisdiction_overrides:
            for key, value in self.credit_jurisdiction_overrides[jurisdiction].items():
                mapped = _CREDIT_OVERRIDE_FIELD_MAP.get(key)
                if mapped:
                    overrides[mapped] = value
        if tier and tier in self.model_tier_overrides:
            for key, value in self.model_tier_overrides[tier].items():
                mapped = _MODEL_TIER_OVERRIDE_FIELD_MAP.get(key)
                if mapped:
                    overrides[mapped] = value
        if not overrides:
            return self
        return dataclasses.replace(self, **overrides)


def _retrieval_query(domain: RiskDomain, inputs: RiskCaseInputs) -> str:
    if isinstance(inputs, CreditModelValidationInputs):
        return (
            f"{inputs.model_type} model validation {inputs.exposure_class} exposure class "
            "EBA GL 2017/16 PD LGD estimation credit risk model validation"
        )
    if isinstance(inputs, NonCreditRiskValidationInputs):
        if inputs.risk_type == "market":
            return (
                f"market risk validation {inputs.business_unit} FRTB BCBS d457 expected "
                "shortfall VaR backtesting P&L attribution"
            )
        return f"{inputs.risk_type} risk validation {inputs.business_unit} ECB guide to internal models"
    return (
        f"model risk validation {inputs.model_tier} ECB guide to internal models EGIM "
        "internal validation governance"
    )


class CreditRiskValidationAgent:
    def __init__(self, llm: HostedLLM) -> None:
        self.llm = llm

    def run(self, inputs: CreditModelValidationInputs, docs: list[dict[str, Any]]) -> list[ValidationFinding]:
        return risk_roles.draft_credit_findings(self.llm, inputs, docs)


class NonCreditRiskValidationAgent:
    def __init__(self, llm: HostedLLM) -> None:
        self.llm = llm

    def run(self, inputs: NonCreditRiskValidationInputs, docs: list[dict[str, Any]]) -> list[ValidationFinding]:
        return risk_roles.draft_noncredit_findings(self.llm, inputs, docs)


class ModelRiskValidationAgent:
    def __init__(self, llm: HostedLLM) -> None:
        self.llm = llm

    def run(self, inputs: ModelRiskValidationInputs, docs: list[dict[str, Any]]) -> list[ValidationFinding]:
        return risk_roles.draft_model_risk_findings(self.llm, inputs, docs)


def _parse_iso_date(value: str | None) -> date | None:
    if not value:
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except ValueError:
        return None


class ValidationGateAgent:
    """Deterministic, non-LLM threshold checks. The LLM never gets to decide
    whether a hard numeric threshold was breached; this class does."""

    def __init__(self, thresholds: ValidationThresholds | None = None) -> None:
        self.thresholds = thresholds or ValidationThresholds()

    # ---------- finding factory (structured citation + remediation deadline) ----------

    def _finding(
        self,
        *,
        domain: RiskDomain,
        area: str,
        verdict: str,
        severity: str,
        description: str,
        recommendation: str = "",
        remediation_deadline_days: int | None = None,
        finding_reference: str | None = None,
    ) -> ValidationFinding:
        deadline = (
            remediation_deadline_days
            if remediation_deadline_days is not None
            else self.thresholds.remediation_deadline_days.get(severity)
        )
        return ValidationFinding(
            domain=domain,
            area=area,
            verdict=verdict,  # type: ignore[arg-type]
            severity=severity,  # type: ignore[arg-type]
            description=description,
            recommendation=recommendation,
            regulatory_reference=citation_text(area),
            regulatory_reference_structured=reference_for(area, domain),
            remediation_deadline_days=deadline,
            finding_reference=finding_reference or f"{domain}:{area}",
        )

    def run(
        self,
        domain: RiskDomain,
        inputs: RiskCaseInputs,
        history: list[ValidationReport] | None = None,
    ) -> tuple[list[ValidationFinding], bool]:
        if domain == "credit_risk":
            findings = self._check_credit(inputs)  # type: ignore[arg-type]
        elif domain == "non_credit_risk":
            findings = self._check_non_credit(inputs)  # type: ignore[arg-type]
        else:
            findings = self._check_model_risk(inputs)  # type: ignore[arg-type]
        findings.extend(self._check_governance(domain, inputs))
        if history:
            findings.extend(self._check_trend(domain, inputs, history))
            findings.extend(self._check_prior_findings(domain, history))
        gate_passed = not any(f.severity == "critical" for f in findings)
        return findings, gate_passed

    # ---------- credit risk ----------

    def _check_credit(self, inputs: CreditModelValidationInputs) -> list[ValidationFinding]:
        t = self.thresholds.resolve(jurisdiction=inputs.jurisdiction)
        findings: list[ValidationFinding] = []
        if inputs.population_stability_index is not None and inputs.population_stability_index > t.credit_max_psi:
            findings.append(self._finding(
                domain="credit_risk",
                area="Calibration and back-testing",
                verdict="non_compliant",
                severity="critical",
                description=(
                    f"Population Stability Index {inputs.population_stability_index:.3f} "
                    f"exceeds the configured threshold {t.credit_max_psi:.3f} (jurisdiction: {inputs.jurisdiction})."
                ),
                recommendation="Investigate population drift and consider model recalibration.",
                finding_reference="credit_risk:psi",
            ))
        if inputs.gini_coefficient is not None and inputs.gini_coefficient < t.credit_min_gini:
            findings.append(self._finding(
                domain="credit_risk",
                area="Discriminatory power",
                verdict="non_compliant",
                severity="high",
                description=(
                    f"Gini coefficient {inputs.gini_coefficient:.3f} is below the "
                    f"configured minimum {t.credit_min_gini:.3f} (jurisdiction: {inputs.jurisdiction})."
                ),
                recommendation="Review model discriminatory power and candidate risk drivers.",
                finding_reference="credit_risk:gini",
            ))
        if inputs.backtesting_exceptions_count is not None and inputs.backtesting_observations_count:
            findings.extend(self._check_credit_backtesting(inputs, t))
        findings.extend(self._check_pd_calibration(inputs, t))
        findings.extend(self._check_lgd_ead_backtesting(inputs, t))
        findings.extend(self._check_downturn_lgd(inputs, t))
        findings.extend(self._check_moc(inputs, t))
        findings.extend(self._check_credit_data_quality(inputs, t))
        findings.extend(self._check_representativeness(inputs, t))
        findings.extend(self._check_model_change(inputs, t))
        if inputs.override_rate_pct is not None and inputs.override_rate_pct > t.credit_max_override_rate_pct:
            findings.append(self._finding(
                domain="credit_risk",
                area="Override analysis",
                verdict="partially_compliant",
                severity="medium",
                description=(
                    f"Override rate {inputs.override_rate_pct:.2f}% exceeds the "
                    f"configured threshold {t.credit_max_override_rate_pct:.2f}%."
                ),
                recommendation="Review override reason codes for systematic patterns.",
                finding_reference="credit_risk:override-rate",
            ))
        if (
            inputs.single_name_concentration_pct is not None
            and inputs.single_name_concentration_pct > t.credit_max_single_name_concentration_pct
        ):
            findings.append(self._finding(
                domain="credit_risk",
                area="Portfolio concentration risk",
                verdict="non_compliant",
                severity="high",
                description=(
                    f"Largest single-name exposure {inputs.single_name_concentration_pct:.2f}% exceeds the "
                    f"configured concentration limit {t.credit_max_single_name_concentration_pct:.2f}%."
                ),
                recommendation="Review concentration limits and diversification requirements for this segment.",
                finding_reference="credit_risk:single-name-concentration",
            ))
        if inputs.ifrs9_stage == "stage_2" and not inputs.sicr_trigger_flag and (
            inputs.days_past_due is None or inputs.days_past_due < 30
        ):
            findings.append(self._finding(
                domain="credit_risk",
                area="IFRS9 staging consistency",
                verdict="partially_compliant",
                severity="medium",
                description=(
                    "Exposure is classified as IFRS 9 Stage 2 but neither a Significant Increase "
                    "in Credit Risk (SICR) trigger nor material days-past-due arrears is evidenced "
                    f"in the case file (days_past_due={inputs.days_past_due!r})."
                ),
                recommendation="Confirm and document the specific SICR trigger supporting the Stage 2 classification.",
                finding_reference="credit_risk:ifrs9-staging",
            ))
        return findings

    def _check_credit_backtesting(
        self, inputs: CreditModelValidationInputs, t: ValidationThresholds,
    ) -> list[ValidationFinding]:
        exceptions = inputs.backtesting_exceptions_count
        observations = inputs.backtesting_observations_count
        rate = exceptions / observations
        if observations < _MIN_OBSERVATIONS_FOR_STATISTICAL_TEST:
            if rate > t.credit_max_backtesting_exception_rate:
                return [self._finding(
                    domain="credit_risk",
                    area="Calibration and back-testing",
                    verdict="non_compliant",
                    severity="high",
                    description=(
                        f"Backtesting exception rate {rate:.3f} ({exceptions}/{observations}) exceeds the "
                        f"configured threshold {t.credit_max_backtesting_exception_rate:.3f}. Sample size is "
                        f"below {_MIN_OBSERVATIONS_FOR_STATISTICAL_TEST} observations, so a flat-rate "
                        "comparison is used instead of a statistical test."
                    ),
                    recommendation="Investigate backtesting exceptions and assess calibration bias.",
                    finding_reference="credit_risk:backtesting-aggregate",
                )]
            return []
        result = kupiec_pof_test(exceptions, observations, t.credit_max_backtesting_exception_rate)
        if result["observed_rate"] > t.credit_max_backtesting_exception_rate and result["p_value"] < t.credit_backtesting_significance_level:
            return [self._finding(
                domain="credit_risk",
                area="Calibration and back-testing",
                verdict="non_compliant",
                severity="high",
                description=(
                    f"Backtesting exception rate {result['observed_rate']:.3f} ({exceptions}/{observations}) "
                    f"is statistically significantly above the tolerated rate {t.credit_max_backtesting_exception_rate:.3f} "
                    f"(Kupiec POF test p-value={result['p_value']:.4f} < {t.credit_backtesting_significance_level:.2f})."
                ),
                recommendation="Investigate backtesting exceptions and assess calibration bias.",
                finding_reference="credit_risk:backtesting-aggregate",
            )]
        return []

    def _check_pd_calibration(
        self, inputs: CreditModelValidationInputs, t: ValidationThresholds,
    ) -> list[ValidationFinding]:
        """Per-grade Jeffreys test plus a portfolio-level exact-binomial test
        (EBA/GL/2017/16 calibration to the long-run average default rate)."""
        grades = inputs.rating_grade_observations
        if not grades:
            return []
        sig = t.credit_pd_calibration_significance
        failing = [
            g.grade
            for g in grades
            for res in (jeffreys_test(g.observed_defaults, g.obligors, g.predicted_pd),)
            if res["observed_rate"] > g.predicted_pd and res["p_value"] < sig
        ]
        total_obligors = sum(g.obligors for g in grades)
        total_defaults = sum(g.observed_defaults for g in grades)
        wavg_pd = (
            sum(g.predicted_pd * g.obligors for g in grades) / total_obligors if total_obligors else 0.0
        )
        port = binomial_pd_test(total_defaults, total_obligors, wavg_pd) if total_obligors else None
        port_fail = bool(port and port["observed_rate"] > wavg_pd and port["p_value"] < sig)
        if not failing and not port_fail:
            return []
        all_grades_fail = len(failing) == len(grades)
        severity = "critical" if (port_fail and all_grades_fail) else "high"
        detail = f" Portfolio-level binomial p-value={port['p_value']:.4f}." if port_fail else ""
        grades_txt = ", ".join(failing) if failing else "none at grade level"
        return [self._finding(
            domain="credit_risk",
            area="Calibration and back-testing",
            verdict="non_compliant",
            severity=severity,
            description=(
                f"PD calibration back-testing rejects the assigned PDs at alpha={sig:.2f} "
                f"(Jeffreys test). Failing grades: {grades_txt}.{detail}"
            ),
            recommendation=(
                "Recalibrate the affected grades to the long-run average default rate and "
                "reassess the Margin of Conservatism."
            ),
            finding_reference="credit_risk:pd-calibration",
        )]

    def _check_lgd_ead_backtesting(
        self, inputs: CreditModelValidationInputs, t: ValidationThresholds,
    ) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        for label, predicted, observed, n, slug in (
            ("LGD", inputs.lgd_predicted_mean, inputs.lgd_observed_mean, inputs.lgd_observation_count, "lgd"),
            ("CCF/EAD", inputs.ccf_predicted_mean, inputs.ccf_observed_mean, inputs.ccf_observation_count, "ccf"),
        ):
            if predicted is None or observed is None:
                continue
            res = normal_mean_test(observed, predicted, n, None, tolerance=t.credit_lgd_backtesting_tolerance)
            if not res["underestimation"]:
                continue
            significant = (
                res["p_value"] is not None and res["p_value"] < t.credit_lgd_backtesting_significance
            ) or (res["p_value"] is None and not res["within_tolerance"])
            if not significant:
                continue
            how = (
                f" (one-sided t-test p-value={res['p_value']:.4f})"
                if res["p_value"] is not None
                else f" beyond the tolerance band of {res['tolerance']:.2f}"
            )
            findings.append(self._finding(
                domain="credit_risk",
                area="Calibration and back-testing",
                verdict="non_compliant",
                severity="high",
                description=(
                    f"Realised {label} mean {observed:.3f} exceeds the predicted mean {predicted:.3f}{how}; "
                    "the model under-estimates losses."
                ),
                recommendation=(
                    f"Recalibrate {label} estimates upward and review the downturn component and "
                    "Margin of Conservatism."
                ),
                finding_reference=f"credit_risk:{slug}-backtesting",
            ))
        return findings

    def _check_downturn_lgd(
        self, inputs: CreditModelValidationInputs, t: ValidationThresholds,
    ) -> list[ValidationFinding]:
        applied = inputs.downturn_lgd_applied
        addon = inputs.downturn_lgd_addon_pct
        is_lgd_model = inputs.model_type in ("LGD", "EAD")
        if applied is None and addon is None and not is_lgd_model:
            return []
        no_downturn = (
            applied is False
            or (applied is None and is_lgd_model and addon is None)
            or (addon is not None and addon <= 0)
        )
        if not no_downturn:
            return []
        return [self._finding(
            domain="credit_risk",
            area="Downturn LGD estimation",
            verdict="non_compliant",
            severity="high",
            description=(
                "No effective downturn adjustment is evidenced for LGD "
                f"(downturn_lgd_applied={applied!r}, add-on={addon!r}). EBA/GL/2019/03 requires a "
                "downturn LGD estimate reflecting an identified economic downturn period."
            ),
            recommendation=(
                "Document the downturn-period identification and apply the downturn LGD add-on or "
                "reference-value approach."
            ),
            finding_reference="credit_risk:downturn-lgd",
        )]

    def _check_moc(
        self, inputs: CreditModelValidationInputs, t: ValidationThresholds,
    ) -> list[ValidationFinding]:
        documented = inputs.moc_framework_documented
        if documented is None:
            return []
        if documented is False:
            return [self._finding(
                domain="credit_risk",
                area="Margin of Conservatism",
                verdict="non_compliant",
                severity="high",
                description=(
                    "No Margin of Conservatism (MoC) framework is documented. EBA/GL/2017/16 "
                    "section 4.4 requires an MoC covering category A (data/methodological "
                    "deficiencies) and category B (general estimation error)."
                ),
                recommendation="Establish and document an MoC framework with category A/B/C components.",
                finding_reference="credit_risk:moc",
            )]
        if (
            inputs.moc_total_pct is not None
            and inputs.moc_total_pct < t.credit_min_moc_pct
            and (inputs.data_deficiencies_count or 0) > 0
        ):
            return [self._finding(
                domain="credit_risk",
                area="Margin of Conservatism",
                verdict="partially_compliant",
                severity="medium",
                description=(
                    f"Documented MoC {inputs.moc_total_pct:.3f} is below the configured floor "
                    f"{t.credit_min_moc_pct:.3f} despite {inputs.data_deficiencies_count} recorded "
                    "data deficiency(ies)."
                ),
                recommendation="Increase the Margin of Conservatism to reflect the identified deficiencies.",
                finding_reference="credit_risk:moc",
            )]
        return []

    def _check_credit_data_quality(
        self, inputs: CreditModelValidationInputs, t: ValidationThresholds,
    ) -> list[ValidationFinding]:
        issues: list[str] = []
        if inputs.data_completeness_pct is not None and inputs.data_completeness_pct < t.credit_min_data_completeness_pct:
            issues.append(
                f"data completeness {inputs.data_completeness_pct:.1f}% < {t.credit_min_data_completeness_pct:.1f}%"
            )
        if inputs.data_accuracy_pct is not None and inputs.data_accuracy_pct < t.credit_min_data_accuracy_pct:
            issues.append(
                f"data accuracy {inputs.data_accuracy_pct:.1f}% < {t.credit_min_data_accuracy_pct:.1f}%"
            )
        if (
            inputs.historical_observation_period_years is not None
            and inputs.historical_observation_period_years < t.credit_min_observation_period_years
        ):
            issues.append(
                f"historical observation period {inputs.historical_observation_period_years:.1f}y "
                f"< {t.credit_min_observation_period_years:.1f}y"
            )
        if not issues:
            return []
        return [self._finding(
            domain="credit_risk",
            area="Data quality",
            verdict="non_compliant",
            severity="high",
            description="Data-quality shortfalls against EBA/GL/2017/16 and BCBS 239: " + "; ".join(issues) + ".",
            recommendation=(
                "Remediate the data-quality gaps and reassess the Margin of Conservatism for data deficiencies."
            ),
            finding_reference="credit_risk:data-quality",
        )]

    def _check_representativeness(
        self, inputs: CreditModelValidationInputs, t: ValidationThresholds,
    ) -> list[ValidationFinding]:
        flags: list[str] = []
        if inputs.representativeness_assessed is False:
            flags.append(
                "no representativeness assessment of the development sample vs the application "
                "portfolio is evidenced"
            )
        if (
            inputs.application_vs_development_psi is not None
            and inputs.application_vs_development_psi > t.credit_max_psi
        ):
            flags.append(
                f"application-vs-development PSI {inputs.application_vs_development_psi:.3f} "
                f"exceeds {t.credit_max_psi:.3f}"
            )
        if not flags:
            return []
        return [self._finding(
            domain="credit_risk",
            area="Data quality",
            verdict="partially_compliant",
            severity="medium",
            description="Representativeness concerns: " + "; ".join(flags) + ".",
            recommendation=(
                "Perform and document a representativeness analysis; quantify any resulting Margin of Conservatism."
            ),
            finding_reference="credit_risk:representativeness",
        )]

    def _check_model_change(
        self, inputs: CreditModelValidationInputs, t: ValidationThresholds,
    ) -> list[ValidationFinding]:
        if inputs.model_change_type == "material" and inputs.model_change_pre_approval_obtained is not True:
            return [self._finding(
                domain="credit_risk",
                area="Model change management",
                verdict="non_compliant",
                severity="critical",
                description=(
                    "A material change/extension to the IRB approach is reported without evidence of "
                    "prior competent-authority approval, contrary to Commission Delegated Regulation "
                    "(EU) No 529/2014."
                ),
                recommendation="Halt deployment of the material change until competent-authority approval is obtained.",
                finding_reference="credit_risk:model-change",
            )]
        return []

    # ---------- non-credit risk ----------

    def _check_non_credit(self, inputs: NonCreditRiskValidationInputs) -> list[ValidationFinding]:
        t = self.thresholds
        findings: list[ValidationFinding] = []
        if inputs.risk_type == "market" and inputs.market_metrics:
            findings.extend(self._check_market(inputs.market_metrics, t))
        elif inputs.risk_type == "operational" and inputs.operational_metrics:
            o = inputs.operational_metrics
            net_loss = o.gross_loss_amount - o.recovery_amount
            if net_loss > t.operational_material_loss_threshold:
                findings.append(self._finding(
                    domain="non_credit_risk",
                    area="Operational loss event review",
                    verdict="non_compliant",
                    severity="high",
                    description=(
                        f"Net operational loss {net_loss:,.2f} ({o.event_category}) exceeds "
                        f"the material loss threshold {t.operational_material_loss_threshold:,.2f}."
                    ),
                    recommendation="Perform root-cause analysis and assess control effectiveness.",
                    finding_reference="non_credit_risk:operational-material-loss",
                ))
        elif inputs.risk_type == "liquidity" and inputs.liquidity_metrics:
            liq = inputs.liquidity_metrics
            if liq.lcr_pct is not None and liq.lcr_pct < t.liquidity_min_lcr_pct:
                findings.append(self._finding(
                    domain="non_credit_risk",
                    area="Liquidity Coverage Ratio",
                    verdict="non_compliant",
                    severity="critical",
                    description=f"LCR {liq.lcr_pct:.1f}% is below the required minimum {t.liquidity_min_lcr_pct:.1f}%.",
                    recommendation="Escalate immediately; review high-quality liquid asset buffer.",
                    finding_reference="non_credit_risk:lcr",
                ))
            if liq.nsfr_pct is not None and liq.nsfr_pct < t.liquidity_min_nsfr_pct:
                findings.append(self._finding(
                    domain="non_credit_risk",
                    area="Net Stable Funding Ratio",
                    verdict="non_compliant",
                    severity="high",
                    description=f"NSFR {liq.nsfr_pct:.1f}% is below the required minimum {t.liquidity_min_nsfr_pct:.1f}%.",
                    recommendation="Review funding structure and stable funding sources.",
                    finding_reference="non_credit_risk:nsfr",
                ))
            if (
                liq.concentration_of_funding_pct is not None
                and liq.concentration_of_funding_pct > t.liquidity_max_funding_concentration_pct
            ):
                findings.append(self._finding(
                    domain="non_credit_risk",
                    area="Funding concentration",
                    verdict="non_compliant",
                    severity="high",
                    description=(
                        f"Funding concentration {liq.concentration_of_funding_pct:.1f}% exceeds the configured "
                        f"limit {t.liquidity_max_funding_concentration_pct:.1f}%."
                    ),
                    recommendation="Diversify funding sources to reduce reliance on the concentrated source(s).",
                    finding_reference="non_credit_risk:funding-concentration",
                ))
        return findings

    def _check_market(self, m, t: ValidationThresholds) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        findings.extend(self._market_backtesting_findings(
            m.var_backtesting_exceptions, m.var_backtesting_observations, m.var_confidence_level, t, "99%",
        ))
        if m.var_backtesting_exceptions_975 is not None and m.var_backtesting_observations_975:
            findings.extend(self._market_backtesting_findings(
                m.var_backtesting_exceptions_975, m.var_backtesting_observations_975,
                t.market_es_confidence_level, t, "97.5%",
            ))
        findings.extend(self._check_pla(m, t))
        findings.extend(self._check_expected_shortfall(m, t))
        return findings

    def _market_backtesting_findings(
        self, exceptions: int | None, observations: int | None, confidence: float,
        t: ValidationThresholds, level_label: str,
    ) -> list[ValidationFinding]:
        if exceptions is None or not observations or observations <= 0:
            return []
        expected_rate = 1.0 - confidence
        if observations < _MIN_OBSERVATIONS_FOR_STATISTICAL_TEST:
            if exceptions > t.market_max_var_exceptions_red_250d:
                zone, severity, verdict = "red", "critical", "non_compliant"
            elif exceptions > t.market_max_var_exceptions_250d:
                zone, severity, verdict = "yellow", "medium", "partially_compliant"
            else:
                return []
            return [self._finding(
                domain="non_credit_risk",
                area="VaR backtesting",
                verdict=verdict,
                severity=severity,
                description=(
                    f"{exceptions} VaR backtesting exceptions over {observations} observations at the "
                    f"{level_label} confidence level falls in the traffic-light {zone.upper()} zone. "
                    f"Sample size is below {_MIN_OBSERVATIONS_FOR_STATISTICAL_TEST} observations, so a "
                    "flat-count comparison is used instead of the Kupiec test."
                ),
                recommendation=_MARKET_REC_RED if zone == "red" else _MARKET_REC_MONITOR,
                finding_reference=f"non_credit_risk:var-backtesting-{level_label}",
            )]
        zone = traffic_light_zone(exceptions, observations, expected_rate)
        if zone == "green":
            return []
        result = kupiec_pof_test(exceptions, observations, expected_rate)
        severity = "critical" if zone == "red" else "medium"
        verdict = "non_compliant" if zone == "red" else "partially_compliant"
        plus = backtesting_plus_factor(exceptions, zone)
        plus_txt = (
            "the internal model approach is unavailable for this book without a supervisory add-on"
            if plus is None
            else f"implied multiplier plus-factor {plus:.2f}"
        )
        return [self._finding(
            domain="non_credit_risk",
            area="VaR backtesting",
            verdict=verdict,
            severity=severity,
            description=(
                f"{exceptions} VaR backtesting exceptions over {observations} observations at the "
                f"{level_label} confidence level falls in the Basel traffic-light {zone.upper()} zone "
                f"(Kupiec POF test p-value={result['p_value']:.4f}); {plus_txt}."
            ),
            recommendation=_MARKET_REC_RED if zone == "red" else _MARKET_REC_MONITOR,
            finding_reference=f"non_credit_risk:var-backtesting-{level_label}",
        )]

    def _check_pla(self, m, t: ValidationThresholds) -> list[ValidationFinding]:
        spearman, ks = m.pla_spearman_correlation, m.pla_ks_statistic
        if (spearman is None or ks is None) and m.hypothetical_pnl and m.risk_theoretical_pnl:
            try:
                computed = pla_statistics(m.hypothetical_pnl, m.risk_theoretical_pnl)
            except ValueError:
                return []
            spearman, ks = computed["spearman"], computed["ks"]
        zone = pla_test(
            spearman, ks,
            spearman_green=t.market_pla_spearman_green,
            spearman_amber=t.market_pla_spearman_amber,
            ks_green=t.market_pla_ks_green,
            ks_amber=t.market_pla_ks_amber,
        )
        if zone is None or zone == "green":
            return []
        if zone == "red":
            return [self._finding(
                domain="non_credit_risk",
                area="P&L attribution",
                verdict="non_compliant",
                severity="critical",
                description=(
                    f"The desk fails the FRTB P&L attribution test (Spearman correlation {spearman:.3f}, "
                    f"KS statistic {ks:.3f}) -- RED zone: the internal model approach is not available "
                    "for this desk's capital."
                ),
                recommendation=(
                    "Remediate the risk-factor coverage gap or move the desk to the standardised approach."
                ),
                finding_reference="non_credit_risk:pla",
            )]
        return [self._finding(
            domain="non_credit_risk",
            area="P&L attribution",
            verdict="partially_compliant",
            severity="medium",
            description=(
                f"The desk is in the AMBER zone of the FRTB P&L attribution test (Spearman correlation "
                f"{spearman:.3f}, KS statistic {ks:.3f}); a capital surcharge applies until it returns "
                "to the green zone."
            ),
            recommendation="Investigate the hypothetical vs risk-theoretical P&L divergence.",
            finding_reference="non_credit_risk:pla",
        )]

    def _check_expected_shortfall(self, m, t: ValidationThresholds) -> list[ValidationFinding]:
        frtb_submission = any(
            value is not None
            for value in (
                m.desk_id, m.expected_shortfall_975, m.stressed_es,
                m.var_backtesting_exceptions_975, m.pla_spearman_correlation,
                m.pla_ks_statistic, m.hypothetical_pnl, m.risk_theoretical_pnl,
            )
        )
        findings: list[ValidationFinding] = []
        if frtb_submission and t.market_require_expected_shortfall and m.expected_shortfall_975 is None:
            findings.append(self._finding(
                domain="non_credit_risk",
                area="Expected shortfall",
                verdict="partially_compliant",
                severity="medium",
                description=(
                    "Expected Shortfall at the 97.5% level is not reported. FRTB (BCBS d457) replaces "
                    "VaR with ES as the regulatory market-risk measure."
                ),
                recommendation="Report ES(97.5%) with the prescribed liquidity-horizon scaling.",
                finding_reference="non_credit_risk:expected-shortfall",
            ))
        if (
            m.stressed_es is not None
            and m.expected_shortfall_975 is not None
            and m.stressed_es <= m.expected_shortfall_975
        ):
            findings.append(self._finding(
                domain="non_credit_risk",
                area="Expected shortfall",
                verdict="partially_compliant",
                severity="medium",
                description=(
                    f"Stressed ES ({m.stressed_es:.4f}) is not greater than current ES "
                    f"({m.expected_shortfall_975:.4f}), which is implausible for a stressed calibration."
                ),
                recommendation="Review the stressed-period selection and the sES calculation.",
                finding_reference="non_credit_risk:stressed-es",
            ))
        return findings

    # ---------- model risk ----------

    def _check_model_risk(self, inputs: ModelRiskValidationInputs) -> list[ValidationFinding]:
        t = self.thresholds.resolve(tier=inputs.model_tier)
        findings: list[ValidationFinding] = []
        metrics = inputs.stability_metrics
        if metrics is None:
            return findings
        if metrics.psi is not None and metrics.psi > t.model_max_psi:
            findings.append(self._finding(
                domain="model_risk",
                area="Stability testing",
                verdict="non_compliant",
                severity="critical",
                description=(
                    f"Population Stability Index {metrics.psi:.3f} exceeds the "
                    f"configured threshold {t.model_max_psi:.3f} for model {inputs.model_id} (tier: {inputs.model_tier})."
                ),
                recommendation="Schedule expedited revalidation; investigate population drift.",
                finding_reference="model_risk:psi",
            ))
        if metrics.gini is not None and metrics.gini < t.model_min_gini:
            findings.append(self._finding(
                domain="model_risk",
                area="Outcomes analysis",
                verdict="non_compliant",
                severity="high",
                description=(
                    f"Gini coefficient {metrics.gini:.3f} is below the configured "
                    f"minimum {t.model_min_gini:.3f} for model {inputs.model_id} (tier: {inputs.model_tier})."
                ),
                recommendation="Review model performance degradation and benchmarking results.",
                finding_reference="model_risk:gini",
            ))
        return findings

    # ---------- governance & lifecycle (cross-domain) ----------

    def _check_governance(self, domain: RiskDomain, inputs: RiskCaseInputs) -> list[ValidationFinding]:
        findings: list[ValidationFinding] = []
        if getattr(inputs, "validation_function_independent", None) is False:
            findings.append(self._finding(
                domain=domain,
                area="Conceptual soundness",
                verdict="non_compliant",
                severity="high",
                description=(
                    "The case file does not attest that the validation function is independent of model "
                    "development, contrary to the ECB guide to internal models (general topics) and the "
                    "EBA validation-independence expectation."
                ),
                recommendation="Confirm and evidence the organisational independence of the validation function.",
                finding_reference=f"{domain}:validation-independence",
            ))
        findings.extend(self._check_revalidation_cadence(domain, inputs))
        return findings

    def _check_revalidation_cadence(
        self, domain: RiskDomain, inputs: RiskCaseInputs,
    ) -> list[ValidationFinding]:
        last = getattr(inputs, "last_validation_date", None)
        nxt = getattr(inputs, "next_scheduled_validation_date", None)
        if not last and not nxt:
            return []
        t = self.thresholds
        if domain == "model_risk":
            tier = getattr(inputs, "model_tier", "")
            max_days = t.model_tier_revalidation_days.get(tier, 365)
            warn_days = t.model_revalidation_warning_days
            area = "Ongoing monitoring review"
        else:
            max_days = t.credit_revalidation_days
            warn_days = t.credit_revalidation_warning_days
            area = "Ongoing monitoring"
        due: date | None = None
        last_date = _parse_iso_date(last)
        if last_date is not None:
            due = last_date + timedelta(days=max_days)
        if due is None:
            due = _parse_iso_date(nxt)
        if due is None:
            return []
        days_to_due = (due - date.today()).days
        if days_to_due < 0:
            return [self._finding(
                domain=domain,
                area=area,
                verdict="non_compliant",
                severity="high",
                description=(
                    f"The model is overdue for revalidation: the scheduled cadence ({max_days} days from "
                    f"the last validation {last or 'n/a'}) elapsed {abs(days_to_due)} day(s) ago "
                    f"(due {due.isoformat()})."
                ),
                recommendation="Schedule and perform the overdue revalidation without further delay.",
                finding_reference=f"{domain}:revalidation-cadence",
            )]
        if days_to_due <= warn_days:
            return [self._finding(
                domain=domain,
                area=area,
                verdict="partially_compliant",
                severity="low",
                description=(
                    f"Revalidation is due in {days_to_due} day(s) ({due.isoformat()}); the next "
                    "validation cycle should be planned to avoid a cadence breach."
                ),
                recommendation="Confirm the next validation is scheduled before the due date.",
                finding_reference=f"{domain}:revalidation-cadence",
            )]
        return []

    # ---------- trend / prior-cycle checks ----------

    def _check_trend(
        self, domain: RiskDomain, inputs: RiskCaseInputs, history: list[ValidationReport],
    ) -> list[ValidationFinding]:
        """Flag monotonic degradation across validation cycles even when the
        current value is still within the absolute threshold."""
        findings: list[ValidationFinding] = []
        if domain == "credit_risk" and isinstance(inputs, CreditModelValidationInputs):
            if inputs.gini_coefficient is not None:
                series = self._prior_values(history, ["gini_coefficient"]) + [inputs.gini_coefficient]
                if self._is_monotonic(series, increasing=False):
                    findings.append(self._trend_finding("credit_risk", "Discriminatory power", series, "declined"))
            if inputs.population_stability_index is not None:
                series = self._prior_values(history, ["population_stability_index"]) + [inputs.population_stability_index]
                if self._is_monotonic(series, increasing=True):
                    findings.append(self._trend_finding("credit_risk", "Calibration and back-testing", series, "increased"))
            if inputs.backtesting_exceptions_count is not None and inputs.backtesting_observations_count:
                current_rate = inputs.backtesting_exceptions_count / inputs.backtesting_observations_count
                series = self._prior_ratio_values(
                    history, "backtesting_exceptions_count", "backtesting_observations_count"
                ) + [current_rate]
                if self._is_monotonic(series, increasing=True):
                    findings.append(
                        self._trend_finding("credit_risk", "Calibration and back-testing", series, "increased")
                    )
        elif (
            domain == "model_risk"
            and isinstance(inputs, ModelRiskValidationInputs)
            and inputs.stability_metrics
        ):
            if inputs.stability_metrics.gini is not None:
                series = self._prior_values(history, ["stability_metrics", "gini"]) + [inputs.stability_metrics.gini]
                if self._is_monotonic(series, increasing=False):
                    findings.append(self._trend_finding("model_risk", "Outcomes analysis", series, "declined"))
            if inputs.stability_metrics.psi is not None:
                series = self._prior_values(history, ["stability_metrics", "psi"]) + [inputs.stability_metrics.psi]
                if self._is_monotonic(series, increasing=True):
                    findings.append(self._trend_finding("model_risk", "Stability testing", series, "increased"))
        return findings

    def _check_prior_findings(
        self, domain: RiskDomain, history: list[ValidationReport],
    ) -> list[ValidationFinding]:
        """Raise an escalated finding for each still-open prior-cycle finding
        that is past its remediation deadline (ECB expectation on follow-up of
        previous findings)."""
        if not history:
            return []
        prior = history[-1]
        prior_date = _parse_iso_date(getattr(prior, "generated_at", None))
        findings: list[ValidationFinding] = []
        for pf in prior.findings:
            if not pf.finding_reference or pf.finding_status != "open":
                continue
            if pf.severity not in ("medium", "high", "critical"):
                continue
            deadline = pf.remediation_deadline_days or self.thresholds.remediation_deadline_days.get(pf.severity)
            if prior_date is None or deadline is None:
                continue
            if (date.today() - prior_date).days <= deadline:
                continue
            escalated = _SEVERITY_ESCALATION.get(pf.severity, "high")
            findings.append(self._finding(
                domain=domain,
                area="Ongoing monitoring",
                verdict="non_compliant",
                severity=escalated,
                description=(
                    f"Finding '{pf.finding_reference}' ({pf.area}), raised in the prior validation cycle "
                    f"({prior.reporting_period or prior_date.isoformat()}), is still open and past its "
                    f"{deadline}-day remediation deadline."
                ),
                recommendation="Escalate the overdue remediation to the model risk committee.",
                finding_reference=f"{domain}:overdue:{pf.finding_reference}",
            ))
        return findings

    @staticmethod
    def _prior_values(history: list[ValidationReport], key_path: list[str]) -> list[float]:
        values: list[float] = []
        for report in history:
            node: Any = report.quantitative_results
            for key in key_path:
                if not isinstance(node, dict) or node.get(key) is None:
                    node = None
                    break
                node = node[key]
            if isinstance(node, (int, float)) and not isinstance(node, bool):
                values.append(float(node))
        return values

    @staticmethod
    def _prior_ratio_values(history: list[ValidationReport], num_key: str, den_key: str) -> list[float]:
        values: list[float] = []
        for report in history:
            q = report.quantitative_results
            num, den = q.get(num_key), q.get(den_key)
            if isinstance(num, (int, float)) and isinstance(den, (int, float)) and den:
                values.append(num / den)
        return values

    @staticmethod
    def _is_monotonic(series: list[float], *, increasing: bool, min_length: int = _TREND_MIN_CYCLES) -> bool:
        if len(series) < min_length:
            return False
        window = series[-min_length:]
        if increasing:
            return all(window[i] < window[i + 1] for i in range(len(window) - 1))
        return all(window[i] > window[i + 1] for i in range(len(window) - 1))

    def _trend_finding(
        self, domain: RiskDomain, area: str, series: list[float], direction: str,
    ) -> ValidationFinding:
        trail = " -> ".join(f"{v:.3f}" for v in series)
        return self._finding(
            domain=domain,
            area=area,
            verdict="partially_compliant",
            severity="medium",
            description=(
                f"{area} has {direction} for {len(series) - 1} consecutive validation cycles "
                f"({trail}), even though the current value remains within the absolute threshold."
            ),
            recommendation=(
                f"Investigate the cause of the {direction} trend before it breaches the absolute threshold."
            ),
            finding_reference=f"{domain}:trend:{area}:{direction}",
        )


def _overall_rating(domain: RiskDomain, findings: list[ValidationFinding]) -> str:
    severity = worst_severity(findings)
    if domain == "model_risk":
        mapping = {None: "low", "observation": "low", "low": "low", "medium": "medium", "high": "high", "critical": "unacceptable"}
    else:
        mapping = {
            None: "compliant",
            "observation": "compliant",
            "low": "compliant",
            "medium": "partially_compliant",
            "high": "non_compliant",
            "critical": "non_compliant",
        }
    return mapping[severity]


def _derive_recommendation(
    overall_rating: str, findings: list[ValidationFinding], gate_passed: bool,
) -> tuple[str, list[str]]:
    """Deterministically map the rating + gate result to a recommendation.
    The LLM never sets this. A draft support tool does not 'approve' a model,
    so the strongest positive outcome is 'not_a_recommendation'."""
    has_critical = any(f.severity == "critical" for f in findings)
    if overall_rating == "unacceptable" or has_critical or not gate_passed:
        recommendation = "reject"
    elif overall_rating in ("non_compliant", "partially_compliant", "high", "medium"):
        recommendation = "approve_with_conditions"
    else:
        recommendation = "not_a_recommendation"
    conditions: list[str] = []
    if recommendation == "approve_with_conditions":
        conditions = [
            f.recommendation
            for f in findings
            if f.severity in ("critical", "high") and f.recommendation
        ]
    return recommendation, conditions


class ReportComposerAgent:
    def __init__(self, llm: HostedLLM) -> None:
        self.llm = llm

    def run(
        self,
        domain: RiskDomain,
        entity_under_review: str,
        reporting_period: str,
        quantitative_results: dict[str, Any],
        findings: list[ValidationFinding],
        *,
        gate_passed: bool = True,
        follow_up: list[PriorFindingStatus] | None = None,
        preparer: str = "",
    ) -> ValidationReport:
        narrative = risk_roles.compose_report_narrative(self.llm, domain, entity_under_review, findings)
        overall_rating = _overall_rating(domain, findings)
        recommendation, conditions = _derive_recommendation(overall_rating, findings, gate_passed)
        return ValidationReport(
            domain=domain,
            title=f"{domain.replace('_', ' ').title()} Validation Report -- {entity_under_review}",
            scope=narrative.scope,
            methodology=narrative.methodology,
            entity_under_review=entity_under_review,
            reporting_period=reporting_period,
            findings=findings,
            quantitative_results=quantitative_results,
            overall_rating=overall_rating,
            overall_conclusion=narrative.overall_conclusion,
            recommendation=recommendation,  # type: ignore[arg-type]
            conditions=conditions,
            follow_up_on_prior_findings=follow_up or [],
            validation_sample=narrative.validation_sample,
            materiality_rationale=narrative.materiality_rationale,
            deviations_from_policy=narrative.deviations_from_policy,
            preparer=preparer,
        )


class RiskValidationOrchestrator:
    """Runs RAG -> specialist findings -> critique/cross-model checks ->
    deterministic gate (history-aware) -> report -> persisted sign-off token
    issuance. All state is persisted via ``storage`` so a pending sign-off
    survives a server restart -- see tools/risk_storage.py."""

    def __init__(
        self,
        llm: HostedLLM,
        rag_agent: RAGAgent,
        credit_agent: CreditRiskValidationAgent,
        noncredit_agent: NonCreditRiskValidationAgent,
        model_agent: ModelRiskValidationAgent,
        gate_agent: ValidationGateAgent,
        composer_agent: ReportComposerAgent,
        storage: RiskStorage,
        *,
        required_signoffs: int = 1,
        approval_ttl_s: int = 1800,
        require_distinct_signoffs: bool = True,
        tier_required_signoffs: dict[str, int] | None = None,
    ) -> None:
        self.llm = llm
        self.rag_agent = rag_agent
        self.credit_agent = credit_agent
        self.noncredit_agent = noncredit_agent
        self.model_agent = model_agent
        self.gate_agent = gate_agent
        self.composer_agent = composer_agent
        self.storage = storage
        self.required_signoffs = required_signoffs
        self.approval_ttl_s = approval_ttl_s
        self.require_distinct_signoffs = require_distinct_signoffs
        # Per-model-tier four-eyes overrides (e.g. Tier 1 -> 2 sign-offs).
        self.tier_required_signoffs = tier_required_signoffs or {}

    def _required_signoffs_for(self, inputs: RiskCaseInputs) -> int:
        if isinstance(inputs, ModelRiskValidationInputs):
            return self.tier_required_signoffs.get(inputs.model_tier, self.required_signoffs)
        return self.required_signoffs

    def _entity_and_period(self, domain: RiskDomain, inputs: RiskCaseInputs) -> tuple[str, str]:
        if isinstance(inputs, CreditModelValidationInputs):
            return f"{inputs.model_id} ({inputs.portfolio_segment})", inputs.last_recalibration_date or "n/a"
        if isinstance(inputs, NonCreditRiskValidationInputs):
            return inputs.business_unit, inputs.reporting_date
        return f"{inputs.model_name} ({inputs.model_id})", inputs.last_validation_date or "n/a"

    def _follow_up_on_prior_findings(
        self, current_findings: list[ValidationFinding], history: list[ValidationReport],
    ) -> list[PriorFindingStatus]:
        if not history:
            return []
        prior = history[-1]
        current_refs = {f.finding_reference for f in current_findings if f.finding_reference}
        overdue_refs = {
            f.finding_reference.split("overdue:", 1)[1]
            for f in current_findings
            if f.finding_reference and "overdue:" in f.finding_reference
        }
        statuses: list[PriorFindingStatus] = []
        for pf in prior.findings:
            if not pf.finding_reference:
                continue
            if pf.finding_reference in overdue_refs:
                status, note = "overdue", "Still open and past its remediation deadline."
            elif pf.finding_reference in current_refs:
                status, note = "open", "Recurs in the current validation cycle."
            else:
                status, note = "resolved", "Not reproduced by the current cycle's checks."
            statuses.append(PriorFindingStatus(
                finding_reference=pf.finding_reference, area=pf.area, status=status, note=note,
            ))
        return statuses

    def run(
        self,
        domain: RiskDomain,
        inputs: RiskCaseInputs,
        on_step: Callable[[AgentTrace], None] | None = None,
        run_id: str | None = None,
        *,
        preparer: str = "",
    ) -> RiskValidationRun:
        """``on_step``, if given, is called with each AgentTrace as it's
        appended -- used to stream live progress (see server.py). ``run_id``,
        if given, overrides the auto-generated run ID -- used by server.py's
        async run flow to pre-allocate an ID before the run starts so the
        client can immediately poll/stream against it. ``preparer`` names the
        identity that prepared the case file; a later sign-off by the same
        identity is rejected (preparer must differ from the validator)."""

        def _trace(agent: str, status: str, summary: str) -> None:
            entry = AgentTrace(agent=agent, status=status, summary=summary)
            state.trace.append(entry)
            if on_step is not None:
                on_step(entry)

        state = RiskValidationRun(domain=domain, inputs=inputs.model_dump(mode="json"), preparer=preparer)
        if run_id is not None:
            state.run_id = run_id
        entity, period = self._entity_and_period(domain, inputs)

        docs = self.rag_agent.run(_retrieval_query(domain, inputs))
        _trace("rag", "ok", f"Retrieved {len(docs)} chunks")

        if domain == "credit_risk":
            llm_findings = self.credit_agent.run(inputs, docs)  # type: ignore[arg-type]
            specialist_name = "credit_validator"
        elif domain == "non_credit_risk":
            llm_findings = self.noncredit_agent.run(inputs, docs)  # type: ignore[arg-type]
            specialist_name = "noncredit_validator"
        else:
            llm_findings = self.model_agent.run(inputs, docs)  # type: ignore[arg-type]
            specialist_name = "model_validator"
        _trace(specialist_name, "ok", f"Drafted {len(llm_findings)} qualitative findings")

        critique = risk_roles.critique_findings(self.llm, llm_findings)
        if not critique.accept and llm_findings:
            revised = risk_roles.revise_findings(self.llm, llm_findings, critique)
            for finding in revised:
                finding.domain = domain
            llm_findings = revised
        _trace(
            "critique",
            "ok" if critique.accept else "revised",
            "; ".join(critique.issues) or "No internal-consistency issues found",
        )

        case_context = inputs.model_dump_json()
        high_severity_checked = 0
        for finding in llm_findings:
            if finding.severity in ("critical", "high"):
                agrees = risk_roles.verify_high_severity_finding(self.llm, finding, case_context)
                finding.secondary_review_flag = not agrees
                high_severity_checked += 1
        if high_severity_checked:
            _trace(
                "cross_model_check", "ok",
                f"Cross-checked {high_severity_checked} critical/high finding(s) against the fallback-tier model",
            )

        history = self.storage.get_history(entity, domain)
        gate_findings, gate_passed = self.gate_agent.run(domain, inputs, history=history)
        _trace(
            "gate",
            "ok" if gate_passed else "escalated",
            "; ".join(f.description for f in gate_findings) or "No deterministic threshold breaches",
        )

        state.findings = llm_findings + gate_findings
        state.gate_passed = gate_passed
        follow_up = self._follow_up_on_prior_findings(gate_findings, history)

        state.report = self.composer_agent.run(
            domain, entity, period, inputs.model_dump(mode="json"), state.findings,
            gate_passed=gate_passed, follow_up=follow_up, preparer=preparer,
        )
        state.report.required_signoffs = self._required_signoffs_for(inputs)
        _trace(
            "composer", "ok",
            f"Overall rating: {state.report.overall_rating}; recommendation: {state.report.recommendation}",
        )

        token, expires = self.storage.issue_approval(state.run_id, state.report, ttl_s=self.approval_ttl_s)
        state.approval_token = token
        state.approval_expires_at_epoch = expires
        _trace("approval", "draft_ready", "Draft ready; human sign-off required before export")

        self.storage.save_run(state)
        self.storage.record_audit(state.run_id, "system", "run_created", f"{domain} run for {entity}")
        return state

    def execute(self, run_id: str, approval_token: str, signed_off_by: str, role: str = "") -> ValidationReport:
        run = self.storage.load_run(run_id)
        if run is None or run.report is None:
            raise ValueError("This run has no report to sign off")

        signer = signed_off_by.strip().lower()
        if run.preparer and signer == run.preparer.strip().lower():
            self.storage.record_audit(
                run_id, signed_off_by, "signoff_rejected", "preparer cannot sign off their own case file"
            )
            raise PermissionError("The preparer of the case file cannot sign off the validation report")
        existing = self.storage.get_signoffs(run_id)
        if self.require_distinct_signoffs and any(s.by.strip().lower() == signer for s in existing):
            self.storage.record_audit(run_id, signed_off_by, "signoff_rejected", "duplicate sign-off identity")
            raise PermissionError("This identity has already signed off this report")

        self.storage.consume_approval(run_id, run.report, approval_token)
        self.storage.add_signoff(run_id, signed_off_by, role=role)
        self.storage.record_audit(run_id, signed_off_by, "signoff_added", role)

        signoffs = self.storage.get_signoffs(run_id)
        run.report.signoffs = signoffs
        distinct_signoffs = len({s.by.strip().lower() for s in signoffs})
        if distinct_signoffs >= run.report.required_signoffs:
            run.report.signed_off_by = signoffs[-1].by
            run.report.signed_off_at = signoffs[-1].at
            self.storage.record_audit(run_id, signed_off_by, "finalized")

        self.storage.save_run(run)
        return run.report
