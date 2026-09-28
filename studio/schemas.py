"""Typed contracts shared by the Modeler and Validator agents.

Everything an LLM produces is parsed into one of these models before it is
used, and everything the deterministic machinery produces (test results,
replication checks, findings from rules) is stored in them too. The JSON on
disk under ``output/studio/<project_id>/`` is a dump of :class:`Project`.
"""
from __future__ import annotations

import datetime as dt
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field

# --------------------------------------------------------------------------
# Enumerations
# --------------------------------------------------------------------------

ModelCategory = Literal[
    "credit_risk",        # PD, LGD, EAD/CCF, scorecards, IFRS 9 ECL
    "market_risk",        # VaR, ES, sensitivities, FRTB
    "operational_risk",
    "liquidity_risk",
    "counterparty_risk",
    "pricing",            # derivative / instrument valuation
    "quant_strategy",     # trading signals, alpha, backtested strategies
    "portfolio",          # allocation, optimisation, risk budgeting
    "ai_ml",              # general ML / AI models
    "other",
]

TestCategory = Literal[
    "unit",
    "data_quality",
    "statistical",
    "performance",
    "calibration",
    "discrimination",
    "stability",
    "sensitivity",
    "stress",
    "backtest",
    "benchmark",
    "robustness",
    "fairness",
    "reproducibility",
    "leakage",
    "other",
]

Severity = Literal["critical", "high", "medium", "low", "observation"]
SEVERITY_ORDER: dict[str, int] = {"critical": 4, "high": 3, "medium": 2, "low": 1, "observation": 0}

FindingStatus = Literal["open", "closed", "disputed", "accepted_limitation"]
Outcome = Literal["approved", "approved_with_conditions", "remediation_required", "rejected"]
ProjectMode = Literal["develop_and_validate", "validate_external"]
ProjectStatus = Literal[
    "queued", "running", "awaiting_signoff", "signed_off", "failed", "cancelled",
]
RequirementStatus = Literal["met", "partially_met", "not_met", "not_applicable", "not_assessed"]


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# --------------------------------------------------------------------------
# Knowledge / requirements
# --------------------------------------------------------------------------

class Citation(BaseModel):
    """Where a statement comes from: a library document and a locator."""

    source: str = Field(description="Document title or file name")
    locator: str = Field(default="", description="Page, section or article, e.g. 'p. 12' or 'Art. 10(2)'")
    chunk_id: str = Field(default="", description="Library chunk id the citation was taken from")

    def text(self) -> str:
        return f"{self.source}, {self.locator}" if self.locator else self.source


class Requirement(BaseModel):
    req_id: str = Field(description="Stable id, e.g. 'EBA-PDLGD-01' or 'ATT-03'")
    framework: str = Field(description="Framework or attached document the requirement comes from")
    text: str
    citation: Citation | None = None
    applies_to: list[str] = Field(default_factory=list, description="Model categories; empty means all")


class RequirementAssessment(BaseModel):
    req_id: str
    status: RequirementStatus
    evidence: str = Field(default="", description="Where in the document/code/results this is shown")
    finding_id: str | None = None


# --------------------------------------------------------------------------
# Modeler outputs
# --------------------------------------------------------------------------

class Equation(BaseModel):
    label: str
    latex: str = Field(description="LaTeX math without surrounding $ signs")
    explanation: str = ""


class AcceptanceCriterion(BaseModel):
    metric: str = Field(description="Key that pipeline writes into metrics.json")
    operator: Literal[">=", "<=", ">", "<", "between"]
    threshold: float
    upper: float | None = Field(default=None, description="Upper bound when operator is 'between'")
    primary: bool = Field(default=False, description="True for the single headline performance metric")
    higher_is_better: bool = True
    rationale: str = ""
    citation: Citation | None = None

    def passes(self, value: float) -> bool:
        if self.operator == ">=":
            return value >= self.threshold
        if self.operator == "<=":
            return value <= self.threshold
        if self.operator == ">":
            return value > self.threshold
        if self.operator == "<":
            return value < self.threshold
        upper = self.upper if self.upper is not None else float("inf")
        return self.threshold <= value <= upper

    def describe(self) -> str:
        if self.operator == "between":
            return f"{self.metric} in [{self.threshold:g}, {self.upper if self.upper is not None else 'inf'}]"
        return f"{self.metric} {self.operator} {self.threshold:g}"


class TestCase(BaseModel):
    __test__ = False  # not a pytest test class

    test_id: str = Field(description="'MT-01' style for modeler tests, 'VT-01' for validator tests")
    name: str
    category: TestCategory
    objective: str
    method: str = Field(description="How the test is carried out, including statistics used")
    acceptance: str = Field(description="Pass/fail rule in words")
    severity_if_fail: Severity = "medium"
    requirement_ids: list[str] = Field(default_factory=list)


class ModelSpec(BaseModel):
    title: str
    category: ModelCategory
    subcategory: str = Field(default="", description="e.g. 'retail PD scorecard', 'historical-simulation VaR'")
    objective: str
    intended_use: str
    target: str = Field(description="What is predicted/estimated/optimised, with units")
    unit_of_analysis: str = ""
    data_description: str
    data_treatment: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    methodology_summary: str
    equations: list[Equation] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    alternatives_considered: list[str] = Field(default_factory=list)
    acceptance_criteria: list[AcceptanceCriterion] = Field(default_factory=list)
    test_plan: list[TestCase] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    requirement_ids: list[str] = Field(default_factory=list, description="Requirements the design addresses")
    monitoring_plan: list[str] = Field(default_factory=list)
    regulatory_profile: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional fields for regulatory gates, e.g. model_type, exposure_class, portfolio_segment",
    )


class CodeFile(BaseModel):
    path: str = Field(description="Relative POSIX path inside the package, e.g. 'model/api.py'")
    content: str


class CodeBundle(BaseModel):
    files: list[CodeFile]
    notes: str = ""


class DocumentSection(BaseModel):
    heading: str
    body: str = Field(description="Plain paragraphs separated by blank lines; '- ' starts a bullet")


class DocumentNarrative(BaseModel):
    executive_summary: str
    sections: list[DocumentSection]


class FindingResponse(BaseModel):
    finding_id: str
    action: Literal["fixed", "disputed", "accepted_limitation", "not_addressed"]
    explanation: str
    changes: list[str] = Field(default_factory=list)


# --------------------------------------------------------------------------
# Execution results (deterministic)
# --------------------------------------------------------------------------

class TestResult(BaseModel):
    __test__ = False  # not a pytest test class

    test_id: str | None = None
    nodeid: str
    outcome: Literal["passed", "failed", "error", "skipped"]
    duration_s: float = 0.0
    message: str = ""


class ExecutionResult(BaseModel):
    command: str
    returncode: int
    timed_out: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""
    duration_s: float = 0.0

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class TestRun(BaseModel):
    __test__ = False  # not a pytest test class

    execution: ExecutionResult
    results: list[TestResult] = Field(default_factory=list)
    collection_error: str = ""

    def counts(self) -> dict[str, int]:
        out = {"passed": 0, "failed": 0, "error": 0, "skipped": 0}
        for r in self.results:
            out[r.outcome] += 1
        return out


class PipelineRun(BaseModel):
    execution: ExecutionResult
    metrics: dict[str, float] = Field(default_factory=dict)
    figures: list[str] = Field(default_factory=list)
    error: str = ""


# --------------------------------------------------------------------------
# Validator outputs
# --------------------------------------------------------------------------

class Finding(BaseModel):
    finding_id: str = Field(description="Stable across rounds, e.g. 'F-R04-MT-03' or 'F-Q-007'")
    title: str
    area: str = Field(description="Conceptual soundness, Data, Implementation, Testing, Performance, Documentation, Governance, Regulatory compliance")
    severity: Severity
    description: str
    evidence: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    recommendation: str = ""
    source: Literal["rule", "review", "independent_test", "challenger", "regulatory_gate"] = "review"
    status: FindingStatus = "open"
    raised_round: int = 1
    closed_round: int | None = None
    modeler_response: FindingResponse | None = None


class ReplicationCheck(BaseModel):
    reproduced: bool
    pipeline_ok: bool
    compared: dict[str, dict[str, float]] = Field(default_factory=dict, description="metric -> {reported, replicated, abs_diff}")
    mismatches: list[str] = Field(default_factory=list)
    note: str = ""


class ChallengerComparison(BaseModel):
    description: str = ""
    champion_metrics: dict[str, float] = Field(default_factory=dict)
    challenger_metrics: dict[str, float] = Field(default_factory=dict)
    primary_metric: str | None = None
    challenger_better: bool | None = None
    pipeline_ok: bool = False
    error: str = ""


class ReviewResult(BaseModel):
    findings: list[Finding] = Field(default_factory=list)
    requirement_assessments: list[RequirementAssessment] = Field(default_factory=list)
    previous_finding_updates: list[dict[str, str]] = Field(
        default_factory=list, description="[{finding_id, status, note}] for earlier review findings"
    )


# --------------------------------------------------------------------------
# Rounds / project
# --------------------------------------------------------------------------

class ModelerRound(BaseModel):
    spec: ModelSpec | None = None
    files: list[str] = Field(default_factory=list)
    pipeline: PipelineRun | None = None
    tests: TestRun | None = None
    repair_attempts: int = 0
    responses: list[FindingResponse] = Field(default_factory=list)
    document_docx: str | None = None
    document_pdf: str | None = None
    narrative: DocumentNarrative | None = None


class ValidatorRound(BaseModel):
    requirements: list[Requirement] = Field(default_factory=list)
    replication: ReplicationCheck | None = None
    modeler_tests_rerun: TestRun | None = None
    independent_plan: list[TestCase] = Field(default_factory=list)
    independent_tests: TestRun | None = None
    challenger: ChallengerComparison | None = None
    review: ReviewResult | None = None
    findings: list[Finding] = Field(default_factory=list)
    outcome: Outcome | None = None
    conclusion: str = ""
    report_docx: str | None = None
    report_pdf: str | None = None


class Round(BaseModel):
    number: int
    modeler: ModelerRound = Field(default_factory=ModelerRound)
    validator: ValidatorRound = Field(default_factory=ValidatorRound)
    started_at: str = Field(default_factory=_now)
    finished_at: str | None = None


class TraceEvent(BaseModel):
    at: str = Field(default_factory=_now)
    agent: Literal["modeler", "validator", "system"]
    step: str
    status: Literal["started", "ok", "warning", "failed"]
    message: str = ""


class Signoff(BaseModel):
    name: str
    role: str = ""
    decision: Literal["accept", "reject"] = "accept"
    comment: str = ""
    at: str = Field(default_factory=_now)


class ProjectInput(BaseModel):
    mode: ProjectMode
    title: str
    brief: str = Field(description="What to model, or what the external model is for")
    frameworks: list[str] = Field(default_factory=list)
    max_rounds: int = Field(default=3, ge=1, le=5)
    challenger: Literal["auto", "on", "off"] = "auto"
    data_files: list[str] = Field(default_factory=list)
    attachment_files: list[str] = Field(default_factory=list)
    package_files: list[str] = Field(default_factory=list, description="External model code/docs (validate_external)")


class Project(BaseModel):
    project_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    input: ProjectInput
    status: ProjectStatus = "queued"
    created_at: str = Field(default_factory=_now)
    updated_at: str = Field(default_factory=_now)
    rounds: list[Round] = Field(default_factory=list)
    trace: list[TraceEvent] = Field(default_factory=list)
    attachment_doc_ids: list[str] = Field(default_factory=list)
    data_profile: dict[str, Any] = Field(default_factory=dict)
    final_outcome: Outcome | None = None
    signoffs: list[Signoff] = Field(default_factory=list)
    error: str = ""
    disclaimer: str = (
        "Generated by AI agents for review by qualified model developers and validators. "
        "It is not an approval of the model and not a regulatory submission."
    )

    def current_round(self) -> Round | None:
        return self.rounds[-1] if self.rounds else None

    def open_findings(self) -> list[Finding]:
        rnd = self.current_round()
        if rnd is None:
            return []
        return [f for f in rnd.validator.findings if f.status == "open"]
