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
import re

from pydantic import BaseModel, Field, ValidationError

from .llm import HostedLLM
from .risk_schemas import (
    CREDIT_VALIDATION_AREAS,
    CreditModelValidationInputs,
    ModelRiskValidationInputs,
    NonCreditRiskValidationInputs,
    ValidationFinding,
)
from .schemas import Critique

# Below this token-overlap ratio between a finding's description and its
# cited evidence text, the finding is flagged evidence_grounded=False
# (informational -- a human validator should double-check it, not an
# automatic rejection). Overridable per call; see configs/config.yaml's
# risk_validation.evidence_overlap_min_ratio.
_DEFAULT_EVIDENCE_OVERLAP_MIN_RATIO = 0.15

RISK_SYSTEM_PROMPTS: dict[str, str] = {
    "credit_validator": """You are a Credit Risk Model Validation analyst in a
bank's independent Risk Validation Department. You review a credit risk
model's case file (PD/LGD/EAD/rating scorecard/IFRS9 ECL) against the
following validation checklist areas, framed by EBA/GL/2017/16 (PD/LGD
estimation and the treatment of defaulted exposures), EBA/GL/2019/03
(downturn LGD estimation), Commission Delegated Regulation (EU) 529/2014
(materiality of IRB model changes), IFRS 9 / EBA/GL/2017/06 (ECL), BCBS 239
(risk data), and the ECB guide to internal models (credit risk chapter):
""" + "\n".join(f"  - {area}" for area in CREDIT_VALIDATION_AREAS) + """

For each area with enough information in the case file to form a judgement,
produce one finding: a verdict (compliant / partially_compliant /
non_compliant / not_applicable), a severity, a description grounded ONLY in
the supplied case file and RAG evidence, and (if not fully compliant) a
concrete recommendation. Pay particular attention to: calibration to the
long-run average default rate per rating grade; the Margin of Conservatism
framework (categories A/B/C); the downturn LGD component; representativeness
of the development sample vs the application portfolio; and whether any
model change is classified and pre-approved. Do not invent metrics that are
not in the case file, and do not decide whether a numeric threshold is
breached -- that is done separately in code. Cite RAG evidence source IDs in
`evidence` when you rely on them. You are drafting for human validator
review -- never claim the model is approved or state a final regulatory
conclusion. Return JSON matching the supplied schema.""",

    "noncredit_validator": """You are a Market/Operational/Liquidity Risk
Validation analyst in a bank's independent Risk Validation Department. You
review the supplied non-credit risk case file (a single risk_type: market,
operational, or liquidity) and its quantitative metrics. Frame your review
by risk type:
  - market: FRTB / BCBS d457 -- Expected Shortfall at 97.5%, VaR
    back-testing at the 99% and 97.5% levels, the P&L attribution test
    (Spearman correlation and KS statistic, green/amber/red), non-modellable
    risk factors, and the stressed measures;
  - liquidity: the LCR and NSFR, funding concentration, and the ECB Guide to
    the ILAAP;
  - operational: the Basel event-type taxonomy and control effectiveness.
Produce findings covering whether the reported metrics look reasonable for
the risk type, data quality/completeness of the case file, and any
governance or control observations. Do not invent metrics that are not in
the case file; the deterministic threshold check is performed separately by
other code, not by you. Cite RAG evidence source IDs in `evidence` when you
rely on them. Return JSON matching the supplied schema.""",

    "model_validator": """You are a Model Risk Validation analyst in a bank's
independent Model Risk Management function, working to the ECB guide to
internal models (EGIM) -- the living guidance that superseded the TRIM
exercise -- and EBA/GL/2017/16. Review the supplied model's case file
(tier, owner, validation activities performed, stability metrics, validation
dates) against the EGIM/TRIM assessment areas: conceptual soundness review,
data quality assessment, outcomes analysis, benchmarking against challenger
models, sensitivity analysis, stability testing (PSI/Gini/KS),
implementation testing, and ongoing monitoring review. Also comment on the
general-topics expectations: independence of the internal validation
function, model tiering and the revalidation cadence, roll-out / permanent
partial use, and use test. Do not invent metrics that are not in the case
file. Cite RAG evidence source IDs in `evidence` when you rely on them.
Return JSON matching the supplied schema.""",

    "report_writer": """You are a Risk Validation report writer. Given a
domain, the case file summary, and the assembled findings (already reviewed
and finalized -- do not add, remove, or reinterpret them), write: a concise
scope statement; a one-paragraph methodology description; an overall
conclusion paragraph that accurately reflects the findings' severities and
verdicts; a `validation_sample` description (reference dates, volumes,
exclusions, as evidenced in the case file); a `materiality_rationale` (why
this tier / scope applies); and any `deviations_from_policy` you can
identify. Do NOT state an approve/reject recommendation -- that is set
deterministically in code from the rating and the gate result. Do not
soften or omit any critical/high severity finding. This text is a DRAFT for
human validator review, not a final regulatory conclusion. Return JSON
matching the supplied schema.""",

    "critic": """You are an independent internal-consistency reviewer for a
bank Risk Validation Department's draft findings. You do not have access to
the original case file -- only the findings themselves. Check for:
  - A verdict that contradicts its severity (e.g. verdict="compliant" paired
    with severity="critical" or "high").
  - Two findings for the same area that contradict each other.
  - A finding whose description doesn't support its stated verdict.
  - Vague, non-actionable descriptions that don't identify a specific issue.
Set accept=true only if you find no such issues. List concrete issues and
suggested_revisions when accept=false. Do not invent new findings or comment
on anything outside internal consistency of what's shown. Return JSON
matching the supplied schema.""",

    "second_opinion": """You are a second, independent Risk Validation
reviewer giving a fast sanity check on a single finding already drafted by
another reviewer, using only the case context and the finding text provided.
Decide whether you agree with the finding's verdict and severity given that
context. This is a lightweight cross-check, not a full re-review. Return
JSON matching the supplied schema.""",
}


class _FindingsDraft(BaseModel):
    """Wrapper so chat_json (which requires a JSON object, not an array) can
    return a list of findings in one call."""

    findings: list[ValidationFinding] = Field(default_factory=list)


class _ReportNarrative(BaseModel):
    scope: str
    methodology: str
    overall_conclusion: str
    # Optional supervisory-report sections. Defaulted so a minimal LLM
    # response ({scope, methodology, overall_conclusion}) still validates.
    validation_sample: str = ""
    materiality_rationale: str = ""
    deviations_from_policy: list[str] = Field(default_factory=list)


class _SecondOpinion(BaseModel):
    agrees: bool
    rationale: str = ""


def _token_overlap_ratio(a: str, b: str) -> float:
    tokens_a = set(re.findall(r"[a-z0-9]+", a.lower()))
    tokens_b = set(re.findall(r"[a-z0-9]+", b.lower()))
    if not tokens_a or not tokens_b:
        return 0.0
    return len(tokens_a & tokens_b) / len(tokens_a)


def _apply_grounding_checks(
    findings: list[ValidationFinding],
    docs: list[dict],
    *,
    min_overlap_ratio: float = _DEFAULT_EVIDENCE_OVERLAP_MIN_RATIO,
) -> list[ValidationFinding]:
    """Strip any evidence ID the RAG step didn't actually retrieve (exact
    pattern reused from agents/quant_team.py::QuantResearchAgent.run()'s
    available_ids set-intersection filter), then flag low plausibility
    overlap between a finding's description and its remaining evidence text.
    Informational only -- never removes or blocks a finding."""

    available = {str(d.get("id", "")): str(d.get("text", "")) for d in docs}
    for finding in findings:
        finding.evidence = [source_id for source_id in finding.evidence if source_id in available]
        if not finding.evidence:
            finding.evidence_grounded = None
            continue
        combined_text = " ".join(available[source_id] for source_id in finding.evidence)
        finding.evidence_grounded = _token_overlap_ratio(finding.description, combined_text) >= min_overlap_ratio
    return findings


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
    llm: HostedLLM,
    inputs: CreditModelValidationInputs,
    docs: list[dict],
    *,
    evidence_overlap_min_ratio: float = _DEFAULT_EVIDENCE_OVERLAP_MIN_RATIO,
) -> list[ValidationFinding]:
    user_msg = f"Credit risk model case file:\n{inputs.model_dump_json(indent=2)}"
    msgs = _build_messages("credit_validator", user_msg, context_docs=docs)
    draft = _chat_json_with_retry(llm, msgs, _FindingsDraft, role="credit_validator")
    findings = draft.findings
    for finding in findings:
        finding.domain = "credit_risk"
    return _apply_grounding_checks(findings, docs, min_overlap_ratio=evidence_overlap_min_ratio)


def draft_noncredit_findings(
    llm: HostedLLM,
    inputs: NonCreditRiskValidationInputs,
    docs: list[dict],
    *,
    evidence_overlap_min_ratio: float = _DEFAULT_EVIDENCE_OVERLAP_MIN_RATIO,
) -> list[ValidationFinding]:
    user_msg = f"Non-credit risk case file:\n{inputs.model_dump_json(indent=2)}"
    msgs = _build_messages("noncredit_validator", user_msg, context_docs=docs)
    draft = _chat_json_with_retry(llm, msgs, _FindingsDraft, role="noncredit_validator")
    findings = draft.findings
    for finding in findings:
        finding.domain = "non_credit_risk"
    return _apply_grounding_checks(findings, docs, min_overlap_ratio=evidence_overlap_min_ratio)


def draft_model_risk_findings(
    llm: HostedLLM,
    inputs: ModelRiskValidationInputs,
    docs: list[dict],
    *,
    evidence_overlap_min_ratio: float = _DEFAULT_EVIDENCE_OVERLAP_MIN_RATIO,
) -> list[ValidationFinding]:
    user_msg = f"Model risk validation case file:\n{inputs.model_dump_json(indent=2)}"
    msgs = _build_messages("model_validator", user_msg, context_docs=docs)
    draft = _chat_json_with_retry(llm, msgs, _FindingsDraft, role="model_validator")
    findings = draft.findings
    for finding in findings:
        finding.domain = "model_risk"
    return _apply_grounding_checks(findings, docs, min_overlap_ratio=evidence_overlap_min_ratio)


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


def critique_findings(llm: HostedLLM, findings: list[ValidationFinding]) -> Critique:
    """Internal-consistency pass over already-drafted findings (severity/
    verdict mismatches, contradictions). Reuses the shared Critique schema
    from agents/schemas.py -- no new schema needed."""
    findings_json = json.dumps([f.model_dump(mode="json") for f in findings], indent=2)
    msgs = _build_messages("critic", f"Findings to review:\n{findings_json}")
    return _chat_json_with_retry(llm, msgs, Critique, role="risk")


def revise_findings(
    llm: HostedLLM, findings: list[ValidationFinding], critique: Critique,
) -> list[ValidationFinding]:
    """One bounded re-draft using the critic's suggested_revisions as extra
    guidance, keeping the same domain-agnostic shape as the original draft."""
    findings_json = json.dumps([f.model_dump(mode="json") for f in findings], indent=2)
    revisions = "\n".join(f"- {item}" for item in critique.suggested_revisions) or "\n".join(critique.issues)
    user_msg = (
        f"Original findings:\n{findings_json}\n\n"
        f"A reviewer flagged these issues and revisions:\n{revisions}\n\n"
        "Return the corrected full set of findings (same schema), fixing only "
        "the flagged issues -- do not remove findings that weren't flagged."
    )
    msgs = _build_messages("critic", user_msg)
    draft = _chat_json_with_retry(llm, msgs, _FindingsDraft, role="risk")
    return draft.findings


def verify_high_severity_finding(llm: HostedLLM, finding: ValidationFinding, case_context: str) -> bool:
    """Re-ask the fallback-tier model (if configured) to independently assess
    a single critical/high finding from the same case data. Returns True
    (agreement) when there's no fallback tier to cross-check against, or when
    the cross-check itself fails -- this is a best-effort flag, never a
    pipeline blocker."""
    models_by_priority = sorted(llm.cfg.models or [], key=lambda m: m.priority)
    if len(models_by_priority) < 2:
        return True
    fallback_model = models_by_priority[1].name
    schema_hint = json.dumps(_SecondOpinion.model_json_schema())
    msgs = [
        {"role": "system", "content": RISK_SYSTEM_PROMPTS["second_opinion"]},
        {"role": "user", "content": f"Case context:\n{case_context}\n\nFinding under review:\n{finding.model_dump_json(indent=2)}"},
        {"role": "system", "content": f"Respond with JSON only. Schema hint:\n{schema_hint}"},
    ]
    try:
        text = llm.chat(msgs, model=fallback_model, json_mode=True, temperature=0.0)
        opinion = _SecondOpinion.model_validate(json.loads(text))
    except Exception:
        return True
    return opinion.agrees
