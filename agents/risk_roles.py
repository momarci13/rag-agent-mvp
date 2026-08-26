"""Role prompts and typed helpers for the bank risk-validation agent team.

Mirrors agents/roles.py's pattern: a system prompt per role plus a thin
typed function that calls ``llm.chat_json(..., schema_hint=..., role=...)``
and validates the result with Pydantic, retrying once on validation failure.

DRAFT / SUPPORT TOOL ONLY -- see agents/risk_schemas.py's module docstring.
The LLM only ever drafts qualitative finding text and report narrative; it
never decides whether a quantitative threshold has been breached (that is
agents/risk_validation_team.py::ValidationGateAgent's job, in plain Python).
"""
from __future__ import annotations

import json

from pydantic import BaseModel, Field, ValidationError

from .llm import HostedLLM
from .risk_schemas import (
    CREDIT_VALIDATION_AREAS,
    CreditModelValidationInputs,
    ModelRiskValidationInputs,
    NonCreditRiskValidationInputs,
    ValidationFinding,
    ValidationReport,
)

RISK_SYSTEM_PROMPTS: dict[str, str] = {
    "credit_validator": """You are a Credit Risk Model Validation analyst in a
bank's independent Risk Validation Department. You review a credit risk
model's case file (PD/LGD/EAD/rating scorecard/IFRS9 ECL) against the
following validation checklist areas, in the style of EBA GL 2017-11:
""" + "\n".join(f"  - {area}" for area in CREDIT_VALIDATION_AREAS) + """

For each area with enough information in the case file to form a judgement,
produce one finding: a verdict (compliant / partially_compliant /
non_compliant / not_applicable), a severity, a description grounded ONLY in
the supplied case file and RAG evidence, and (if not fully compliant) a
concrete recommendation. Do not invent metrics that are not in the case
file. Cite RAG evidence source IDs in `evidence` when you rely on them. You
are drafting for human validator review -- never claim the model is
approved or state a final regulatory conclusion. Return JSON matching the
supplied schema.""",

    "noncredit_validator": """You are a Market/Operational/Liquidity Risk
Validation analyst in a bank's independent Risk Validation Department. You
review the supplied non-credit risk case file (a single risk_type: market,
operational, or liquidity) and its quantitative metrics. Produce findings
covering: whether the reported metrics are within a normal/expected range
for the type of risk, data quality/completeness of the case file, and any
governance or control observations evident from the case file. Do not
invent metrics that are not in the case file; the deterministic threshold
check is performed separately by other code, not by you. Cite RAG evidence
source IDs in `evidence` when you rely on them. Return JSON matching the
supplied schema.""",

    "model_validator": """You are a Model Risk Validation analyst in a bank's
independent Model Risk Management function (ECB TRIM / EBA GL 2017-11
style). You review the supplied model's case file (tier, owner, validation
activities performed, stability metrics) and produce findings covering
conceptual soundness, data quality, outcomes analysis / benchmarking
results, and stability (PSI/Gini/KS) as reported in the case file. Do not
invent metrics that are not in the case file. Cite RAG evidence source IDs
in `evidence` when you rely on them. Return JSON matching the supplied
schema.""",

    "report_writer": """You are a Risk Validation report writer. Given a
domain, the case file summary, and the assembled findings (already reviewed
and finalized -- do not add, remove, or reinterpret them), write a concise
scope statement, a one-paragraph methodology description, and an overall
conclusion paragraph that accurately reflects the findings' severities and
verdicts. Do not soften or omit any critical/high severity finding. This
text is a DRAFT for human validator review, not a final regulatory
conclusion. Return JSON matching the supplied schema.""",
}


class _FindingsDraft(BaseModel):
    """Wrapper so chat_json (which requires a JSON object, not an array) can
    return a list of findings in one call."""

    findings: list[ValidationFinding] = Field(default_factory=list)


class _ReportNarrative(BaseModel):
    scope: str
    methodology: str
    overall_conclusion: str


def _build_messages(role: str, user_msg: str, *, context_docs: list[dict] | None = None) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": RISK_SYSTEM_PROMPTS[role]}]
    if context_docs:
        ctx = "\n\n".join(
            f"[source:{d.get('id', '?')}] {d.get('text', '')}" for d in context_docs
        )
        msgs.append({
            "role": "system",
            "content": f"RAG evidence (regulatory reference material):\n{ctx}",
        })
    msgs.append({"role": "user", "content": user_msg})
    return msgs


def _chat_json_with_retry(llm: HostedLLM, msgs: list[dict], schema: type[BaseModel], *, role: str) -> BaseModel:
    schema_hint = json.dumps(schema.model_json_schema())
    raw = llm.chat_json(msgs, schema_hint=schema_hint, role=role)
    try:
        return schema.model_validate(raw)
    except ValidationError as exc:
        retry_msgs = msgs + [
            {"role": "assistant", "content": json.dumps(raw)},
            {"role": "user", "content": f"Your JSON failed validation: {exc}. Fix and resend."},
        ]
        raw2 = llm.chat_json(retry_msgs, schema_hint=schema_hint, role=role)
        return schema.model_validate(raw2)


def draft_credit_findings(
    llm: HostedLLM, inputs: CreditModelValidationInputs, docs: list[dict],
) -> list[ValidationFinding]:
    user_msg = f"Credit risk model case file:\n{inputs.model_dump_json(indent=2)}"
    msgs = _build_messages("credit_validator", user_msg, context_docs=docs)
    draft = _chat_json_with_retry(llm, msgs, _FindingsDraft, role="credit_validator")
    findings = draft.findings
    for finding in findings:
        finding.domain = "credit_risk"
    return findings


def draft_noncredit_findings(
    llm: HostedLLM, inputs: NonCreditRiskValidationInputs, docs: list[dict],
) -> list[ValidationFinding]:
    user_msg = f"Non-credit risk case file:\n{inputs.model_dump_json(indent=2)}"
    msgs = _build_messages("noncredit_validator", user_msg, context_docs=docs)
    draft = _chat_json_with_retry(llm, msgs, _FindingsDraft, role="noncredit_validator")
    findings = draft.findings
    for finding in findings:
        finding.domain = "non_credit_risk"
    return findings


def draft_model_risk_findings(
    llm: HostedLLM, inputs: ModelRiskValidationInputs, docs: list[dict],
) -> list[ValidationFinding]:
    user_msg = f"Model risk validation case file:\n{inputs.model_dump_json(indent=2)}"
    msgs = _build_messages("model_validator", user_msg, context_docs=docs)
    draft = _chat_json_with_retry(llm, msgs, _FindingsDraft, role="model_validator")
    findings = draft.findings
    for finding in findings:
        finding.domain = "model_risk"
    return findings


def compose_report_narrative(
    llm: HostedLLM, domain: str, entity_under_review: str, findings: list[ValidationFinding],
) -> _ReportNarrative:
    findings_json = json.dumps([f.model_dump(mode="json") for f in findings], indent=2)
    user_msg = (
        f"Domain: {domain}\nEntity under review: {entity_under_review}\n"
        f"Findings:\n{findings_json}"
    )
    msgs = _build_messages("report_writer", user_msg)
    return _chat_json_with_retry(llm, msgs, _ReportNarrative, role="report_writer")
