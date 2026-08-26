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
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any

from rag.hybrid import LiteHybridRAG
from tools.approval import ApprovalStore

from . import risk_roles
from .llm import HostedLLM
from .quant_team import RAGAgent  # generic; reused as-is
from .risk_schemas import (
    AgentTrace,
    CreditModelValidationInputs,
    ModelRiskValidationInputs,
    NonCreditRiskValidationInputs,
    RiskDomain,
    RiskValidationRun,
    ValidationFinding,
    ValidationReport,
    worst_severity,
)

RiskCaseInputs = CreditModelValidationInputs | NonCreditRiskValidationInputs | ModelRiskValidationInputs


@dataclass(frozen=True)
class ValidationThresholds:
    """Illustrative default thresholds -- see configs/config.yaml's
    ``risk_validation`` block. Must be replaced with the bank's actual
    validation-policy-owned limits before any real use."""

    credit_max_psi: float = 0.25
    credit_min_gini: float = 0.40
    credit_max_backtesting_exception_rate: float = 0.05
    credit_max_override_rate_pct: float = 0.10
    market_max_var_exceptions_250d: int = 4
    market_max_var_exceptions_red_250d: int = 9
    operational_material_loss_threshold: float = 100000.0
    liquidity_min_lcr_pct: float = 100.0
    liquidity_min_nsfr_pct: float = 100.0
    model_max_psi: float = 0.25
    model_min_gini: float = 0.40


def _retrieval_query(domain: RiskDomain, inputs: RiskCaseInputs) -> str:
    if isinstance(inputs, CreditModelValidationInputs):
        return (
            f"{inputs.model_type} model validation {inputs.exposure_class} "
            f"exposure class EBA GL 2017-11 credit risk model validation"
        )
    if isinstance(inputs, NonCreditRiskValidationInputs):
        return f"{inputs.risk_type} risk validation {inputs.business_unit}"
    return f"model risk validation {inputs.model_tier} ECB TRIM model risk management"


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


class ValidationGateAgent:
    """Deterministic, non-LLM threshold checks. The LLM never gets to decide
    whether a hard numeric threshold was breached; this class does."""

    def __init__(self, thresholds: ValidationThresholds | None = None) -> None:
        self.thresholds = thresholds or ValidationThresholds()

    def run(self, domain: RiskDomain, inputs: RiskCaseInputs) -> tuple[list[ValidationFinding], bool]:
        if domain == "credit_risk":
            findings = self._check_credit(inputs)  # type: ignore[arg-type]
        elif domain == "non_credit_risk":
            findings = self._check_non_credit(inputs)  # type: ignore[arg-type]
        else:
            findings = self._check_model_risk(inputs)  # type: ignore[arg-type]
        gate_passed = not any(f.severity == "critical" for f in findings)
        return findings, gate_passed

    def _check_credit(self, inputs: CreditModelValidationInputs) -> list[ValidationFinding]:
        t = self.thresholds
        findings: list[ValidationFinding] = []
        if inputs.population_stability_index is not None and inputs.population_stability_index > t.credit_max_psi:
            findings.append(ValidationFinding(
                domain="credit_risk",
                area="Calibration and back-testing",
                verdict="non_compliant",
                severity="critical",
                description=(
                    f"Population Stability Index {inputs.population_stability_index:.3f} "
                    f"exceeds the configured threshold {t.credit_max_psi:.3f}."
                ),
                recommendation="Investigate population drift and consider model recalibration.",
            ))
        if inputs.gini_coefficient is not None and inputs.gini_coefficient < t.credit_min_gini:
            findings.append(ValidationFinding(
                domain="credit_risk",
                area="Discriminatory power",
                verdict="non_compliant",
                severity="high",
                description=(
                    f"Gini coefficient {inputs.gini_coefficient:.3f} is below the "
                    f"configured minimum {t.credit_min_gini:.3f}."
                ),
                recommendation="Review model discriminatory power and candidate risk drivers.",
            ))
        if inputs.backtesting_exceptions_count is not None and inputs.backtesting_observations_count:
            rate = inputs.backtesting_exceptions_count / inputs.backtesting_observations_count
            if rate > t.credit_max_backtesting_exception_rate:
                findings.append(ValidationFinding(
                    domain="credit_risk",
                    area="Calibration and back-testing",
                    verdict="non_compliant",
                    severity="high",
                    description=(
                        f"Backtesting exception rate {rate:.3f} "
                        f"({inputs.backtesting_exceptions_count}/{inputs.backtesting_observations_count}) "
                        f"exceeds the configured threshold {t.credit_max_backtesting_exception_rate:.3f}."
                    ),
                    recommendation="Investigate backtesting exceptions and assess calibration bias.",
                ))
        if inputs.override_rate_pct is not None and inputs.override_rate_pct > t.credit_max_override_rate_pct:
            findings.append(ValidationFinding(
                domain="credit_risk",
                area="Override analysis",
                verdict="partially_compliant",
                severity="medium",
                description=(
                    f"Override rate {inputs.override_rate_pct:.2f}% exceeds the "
                    f"configured threshold {t.credit_max_override_rate_pct:.2f}%."
                ),
                recommendation="Review override reason codes for systematic patterns.",
            ))
        return findings

    def _check_non_credit(self, inputs: NonCreditRiskValidationInputs) -> list[ValidationFinding]:
        t = self.thresholds
        findings: list[ValidationFinding] = []
        if inputs.risk_type == "market" and inputs.market_metrics:
            m = inputs.market_metrics
            if m.var_backtesting_exceptions > t.market_max_var_exceptions_red_250d:
                findings.append(ValidationFinding(
                    domain="non_credit_risk",
                    area="VaR backtesting",
                    verdict="non_compliant",
                    severity="critical",
                    description=(
                        f"{m.var_backtesting_exceptions} VaR backtesting exceptions over "
                        f"{m.var_backtesting_observations} observations falls in the Basel "
                        f"traffic-light RED zone (> {t.market_max_var_exceptions_red_250d})."
                    ),
                    recommendation="Escalate to model risk committee; review VaR model assumptions.",
                ))
            elif m.var_backtesting_exceptions > t.market_max_var_exceptions_250d:
                findings.append(ValidationFinding(
                    domain="non_credit_risk",
                    area="VaR backtesting",
                    verdict="partially_compliant",
                    severity="medium",
                    description=(
                        f"{m.var_backtesting_exceptions} VaR backtesting exceptions over "
                        f"{m.var_backtesting_observations} observations falls in the Basel "
                        f"traffic-light AMBER zone (> {t.market_max_var_exceptions_250d})."
                    ),
                    recommendation="Monitor closely; document rationale for each exception.",
                ))
        elif inputs.risk_type == "operational" and inputs.operational_metrics:
            o = inputs.operational_metrics
            net_loss = o.gross_loss_amount - o.recovery_amount
            if net_loss > t.operational_material_loss_threshold:
                findings.append(ValidationFinding(
                    domain="non_credit_risk",
                    area="Operational loss event review",
                    verdict="non_compliant",
                    severity="high",
                    description=(
                        f"Net operational loss {net_loss:,.2f} ({o.event_category}) exceeds "
                        f"the material loss threshold {t.operational_material_loss_threshold:,.2f}."
                    ),
                    recommendation="Perform root-cause analysis and assess control effectiveness.",
                ))
        elif inputs.risk_type == "liquidity" and inputs.liquidity_metrics:
            liq = inputs.liquidity_metrics
            if liq.lcr_pct is not None and liq.lcr_pct < t.liquidity_min_lcr_pct:
                findings.append(ValidationFinding(
                    domain="non_credit_risk",
                    area="Liquidity Coverage Ratio",
                    verdict="non_compliant",
                    severity="critical",
                    description=f"LCR {liq.lcr_pct:.1f}% is below the required minimum {t.liquidity_min_lcr_pct:.1f}%.",
                    recommendation="Escalate immediately; review high-quality liquid asset buffer.",
                ))
            if liq.nsfr_pct is not None and liq.nsfr_pct < t.liquidity_min_nsfr_pct:
                findings.append(ValidationFinding(
                    domain="non_credit_risk",
                    area="Net Stable Funding Ratio",
                    verdict="non_compliant",
                    severity="high",
                    description=f"NSFR {liq.nsfr_pct:.1f}% is below the required minimum {t.liquidity_min_nsfr_pct:.1f}%.",
                    recommendation="Review funding structure and stable funding sources.",
                ))
        return findings

    def _check_model_risk(self, inputs: ModelRiskValidationInputs) -> list[ValidationFinding]:
        t = self.thresholds
        findings: list[ValidationFinding] = []
        metrics = inputs.stability_metrics
        if metrics is None:
            return findings
        if metrics.psi is not None and metrics.psi > t.model_max_psi:
            findings.append(ValidationFinding(
                domain="model_risk",
                area="Stability testing",
                verdict="non_compliant",
                severity="critical",
                description=(
                    f"Population Stability Index {metrics.psi:.3f} exceeds the "
                    f"configured threshold {t.model_max_psi:.3f} for model {inputs.model_id}."
                ),
                recommendation="Schedule expedited revalidation; investigate population drift.",
            ))
        if metrics.gini is not None and metrics.gini < t.model_min_gini:
            findings.append(ValidationFinding(
                domain="model_risk",
                area="Outcomes analysis",
                verdict="non_compliant",
                severity="high",
                description=(
                    f"Gini coefficient {metrics.gini:.3f} is below the configured "
                    f"minimum {t.model_min_gini:.3f} for model {inputs.model_id}."
                ),
                recommendation="Review model performance degradation and benchmarking results.",
            ))
        return findings


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
    ) -> ValidationReport:
        narrative = risk_roles.compose_report_narrative(self.llm, domain, entity_under_review, findings)
        return ValidationReport(
            domain=domain,
            title=f"{domain.replace('_', ' ').title()} Validation Report -- {entity_under_review}",
            scope=narrative.scope,
            methodology=narrative.methodology,
            entity_under_review=entity_under_review,
            reporting_period=reporting_period,
            findings=findings,
            quantitative_results=quantitative_results,
            overall_rating=_overall_rating(domain, findings),
            overall_conclusion=narrative.overall_conclusion,
        )


class RiskValidationOrchestrator:
    """Runs RAG -> specialist findings -> deterministic gate -> report ->
    sign-off token issuance."""

    def __init__(
        self,
        rag_agent: RAGAgent,
        credit_agent: CreditRiskValidationAgent,
        noncredit_agent: NonCreditRiskValidationAgent,
        model_agent: ModelRiskValidationAgent,
        gate_agent: ValidationGateAgent,
        composer_agent: ReportComposerAgent,
        approvals: ApprovalStore | None = None,
    ) -> None:
        self.rag_agent = rag_agent
        self.credit_agent = credit_agent
        self.noncredit_agent = noncredit_agent
        self.model_agent = model_agent
        self.gate_agent = gate_agent
        self.composer_agent = composer_agent
        self.approvals = approvals or ApprovalStore()

    def _entity_and_period(self, domain: RiskDomain, inputs: RiskCaseInputs) -> tuple[str, str]:
        if isinstance(inputs, CreditModelValidationInputs):
            return f"{inputs.model_id} ({inputs.portfolio_segment})", inputs.last_recalibration_date or "n/a"
        if isinstance(inputs, NonCreditRiskValidationInputs):
            return inputs.business_unit, inputs.reporting_date
        return f"{inputs.model_name} ({inputs.model_id})", inputs.last_validation_date or "n/a"

    def run(self, domain: RiskDomain, inputs: RiskCaseInputs) -> RiskValidationRun:
        state = RiskValidationRun(domain=domain, inputs=inputs.model_dump(mode="json"))

        docs = self.rag_agent.run(_retrieval_query(domain, inputs))
        state.trace.append(AgentTrace(agent="rag", status="ok", summary=f"Retrieved {len(docs)} chunks"))

        if domain == "credit_risk":
            llm_findings = self.credit_agent.run(inputs, docs)  # type: ignore[arg-type]
            specialist_name = "credit_validator"
        elif domain == "non_credit_risk":
            llm_findings = self.noncredit_agent.run(inputs, docs)  # type: ignore[arg-type]
            specialist_name = "noncredit_validator"
        else:
            llm_findings = self.model_agent.run(inputs, docs)  # type: ignore[arg-type]
            specialist_name = "model_validator"
        state.trace.append(AgentTrace(
            agent=specialist_name, status="ok", summary=f"Drafted {len(llm_findings)} qualitative findings",
        ))

        gate_findings, gate_passed = self.gate_agent.run(domain, inputs)
        state.trace.append(AgentTrace(
            agent="gate",
            status="ok" if gate_passed else "escalated",
            summary="; ".join(f.description for f in gate_findings) or "No deterministic threshold breaches",
        ))

        state.findings = llm_findings + gate_findings
        state.gate_passed = gate_passed

        entity, period = self._entity_and_period(domain, inputs)
        state.report = self.composer_agent.run(
            domain, entity, period, inputs.model_dump(mode="json"), state.findings,
        )
        state.trace.append(AgentTrace(
            agent="composer", status="ok", summary=f"Overall rating: {state.report.overall_rating}",
        ))

        token, expires = self.approvals.issue(state.report)
        state.approval_token = token
        state.approval_expires_at_epoch = expires
        state.trace.append(AgentTrace(
            agent="approval", status="draft_ready", summary="Draft ready; human sign-off required before export",
        ))
        return state

    def execute(self, run: RiskValidationRun, approval_token: str, signed_off_by: str) -> ValidationReport:
        if run.report is None:
            raise ValueError("This run has no report to sign off")
        self.approvals.consume(run.report, approval_token)
        run.report.signed_off_by = signed_off_by
        run.report.signed_off_at = dt.datetime.now(dt.timezone.utc).isoformat()
        return run.report
