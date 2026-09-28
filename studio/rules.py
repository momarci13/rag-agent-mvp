"""Deterministic validation rules and the outcome rating.

These findings come from executed results only; the LLM cannot create,
soften or remove them. IDs are stable across rounds so a finding raised in
round 1 is recognised (and closed) in round 2.
"""
from __future__ import annotations

import math
import re
from typing import Any

from pydantic import ValidationError

from .schemas import (
    SEVERITY_ORDER,
    ChallengerComparison,
    Citation,
    Finding,
    ModelSpec,
    Outcome,
    ReplicationCheck,
    Severity,
    TestCase,
    TestRun,
)


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-")[:40] or "x"


def _f(fid: str, title: str, area: str, severity: Severity, description: str, *,
       evidence: list[str] | None = None, recommendation: str = "", source: str = "rule",
       citations: list[Citation] | None = None) -> Finding:
    return Finding(
        finding_id=fid, title=title, area=area, severity=severity, description=description,
        evidence=evidence or [], recommendation=recommendation, source=source,  # type: ignore[arg-type]
        citations=citations or [],
    )


def compare_metrics(reported: dict[str, float], replicated: dict[str, float], rel_tol: float, abs_tol: float
                    ) -> tuple[dict[str, dict[str, float]], list[str]]:
    compared: dict[str, dict[str, float]] = {}
    mismatches: list[str] = []
    for key, rep in sorted(reported.items()):
        new = replicated.get(key)
        if new is None:
            mismatches.append(key)
            compared[key] = {"reported": rep, "replicated": float("nan"), "abs_diff": float("nan")}
            continue
        diff = abs(new - rep)
        compared[key] = {"reported": rep, "replicated": new, "abs_diff": diff}
        if not math.isclose(rep, new, rel_tol=rel_tol, abs_tol=abs_tol):
            mismatches.append(key)
    return compared, mismatches


def replication_findings(rep: ReplicationCheck) -> list[Finding]:
    out: list[Finding] = []
    if not rep.pipeline_ok:
        out.append(_f(
            "F-R01", "Model pipeline does not run from a clean copy", "Implementation", "critical",
            "The delivered pipeline failed when the validator re-ran it from a fresh copy of the package. "
            "No result in the documentation can be relied on until it runs.",
            evidence=[rep.note[:1500]] if rep.note else [],
            recommendation="Fix the pipeline so `python pipeline.py --data-dir <data> --out-dir <dir>` completes and writes metrics.json.",
        ))
        return out
    for metric in rep.mismatches:
        d = rep.compared.get(metric, {})
        out.append(_f(
            f"F-R02-{_slug(metric)}", f"Reported {metric} could not be reproduced", "Implementation", "high",
            f"Re-running the pipeline gave {d.get('replicated')} for {metric}, against {d.get('reported')} reported by the developer.",
            recommendation="Remove non-determinism (seeds, unordered operations, time-dependent code) or explain and bound it.",
        ))
    return out


def test_suite_findings(spec: ModelSpec | None, run: TestRun | None, *, external: bool) -> list[Finding]:
    out: list[Finding] = []
    if run is None:
        return out
    if run.collection_error:
        out.append(_f(
            "F-R08", "Developer test suite cannot be run", "Testing", "high",
            "pytest could not collect or run the delivered test suite.",
            evidence=[run.collection_error[:1500]],
            recommendation="Make the test suite runnable with `python -m pytest tests`.",
        ))
        return out
    plan: dict[str, TestCase] = {t.test_id: t for t in (spec.test_plan if spec else [])}
    by_id: dict[str, list] = {}
    for r in run.results:
        if r.test_id:
            by_id.setdefault(r.test_id, []).append(r)
    if not external:
        for tid, tc in plan.items():
            if tid not in by_id:
                out.append(_f(
                    f"F-R03-{tid}", f"Planned test {tid} is not implemented", "Testing", "medium",
                    f"The test plan lists {tid} ({tc.name}) but no test function named test_{tid.replace('-', '')}... was run.",
                    recommendation=f"Implement {tid} as documented or remove it from the plan with a justification.",
                ))
    for r in run.results:
        if r.outcome not in ("failed", "error"):
            continue
        tc = plan.get(r.test_id or "")
        key = r.test_id or _slug(r.nodeid.split("::")[-1])
        if r.outcome == "error":
            severity: Severity = "high"
            title = f"Developer test {key} errors"
        else:
            severity = tc.severity_if_fail if tc else "medium"
            title = f"Developer test {key} fails" + (f": {tc.name}" if tc else "")
        out.append(_f(
            f"F-R04-{key}", title, "Testing", severity,
            (f"{tc.objective} Acceptance rule: {tc.acceptance}. " if tc else "") + f"Outcome: {r.outcome}.",
            evidence=[r.nodeid, r.message[:1200]],
            recommendation="Address the cause of the failure; do not relax the acceptance rule without justification.",
        ))
    # Deduplicate by id, keeping the most severe.
    dedup: dict[str, Finding] = {}
    for f in out:
        if f.finding_id not in dedup or SEVERITY_ORDER[f.severity] > SEVERITY_ORDER[dedup[f.finding_id].severity]:
            dedup[f.finding_id] = f
    return list(dedup.values())


def acceptance_findings(spec: ModelSpec | None, metrics: dict[str, float], pipeline_ok: bool) -> list[Finding]:
    out: list[Finding] = []
    if spec is None or not pipeline_ok:
        return out
    for ac in spec.acceptance_criteria:
        value = metrics.get(ac.metric)
        if value is None:
            out.append(_f(
                f"F-R05-{_slug(ac.metric)}", f"Acceptance metric {ac.metric} is not produced", "Performance", "medium",
                f"The acceptance criterion {ac.describe()} cannot be checked because the pipeline does not output {ac.metric}.",
                recommendation=f"Write {ac.metric} to metrics.json.",
            ))
        elif not ac.passes(value):
            out.append(_f(
                f"F-R05-{_slug(ac.metric)}", f"Acceptance criterion not met: {ac.describe()}", "Performance",
                "high" if ac.primary else "medium",
                f"Replicated value {value:.6g} does not satisfy {ac.describe()}." + (f" Rationale for the threshold: {ac.rationale}" if ac.rationale else ""),
                citations=[ac.citation] if ac.citation else [],
                recommendation="Improve the model or document why the criterion cannot be met and what compensates for it.",
            ))
    return out


def documentation_findings(spec: ModelSpec | None) -> list[Finding]:
    if spec is None:
        return []
    checks = [
        ("assumptions", spec.assumptions, "Key assumptions are not documented"),
        ("limitations", spec.limitations, "Limitations are not documented"),
        ("monitoring", spec.monitoring_plan, "No ongoing monitoring plan"),
        ("equations", spec.equations, "The methodology has no mathematical specification"),
        ("acceptance", spec.acceptance_criteria, "No quantitative acceptance criteria were set"),
    ]
    return [
        _f(f"F-R09-{key}", title, "Documentation", "medium",
           f"{title}. Independent review and ongoing use depend on this element.",
           recommendation="Add this element to the modelling document.")
        for key, items, title in checks if not items
    ]


def independent_findings(plan: list[TestCase], run: TestRun | None) -> list[Finding]:
    out: list[Finding] = []
    if run is None:
        return out
    if run.collection_error:
        out.append(_f(
            "F-R06-suite", "Independent test suite could not be executed", "Testing", "low",
            "The validator's own tests could not be run; the checks they cover remain unperformed.",
            evidence=[run.collection_error[:1000]], source="independent_test",
            recommendation="Validator to re-run the independent tests manually.",
        ))
        return out
    plan_by_id = {t.test_id: t for t in plan}
    for r in run.results:
        if r.outcome not in ("failed", "error") or not r.test_id:
            continue
        tc = plan_by_id.get(r.test_id)
        if r.outcome == "error":
            out.append(_f(
                f"F-R06-{r.test_id}", f"Independent test {r.test_id} could not be completed", "Testing", "low",
                "The test raised an error (not an assertion failure); the check is inconclusive.",
                evidence=[r.message[:1000]], source="independent_test",
            ))
            continue
        out.append(_f(
            f"F-R06-{r.test_id}", f"Independent test {r.test_id} fails" + (f": {tc.name}" if tc else ""),
            "Performance" if tc and tc.category in ("performance", "calibration", "discrimination", "backtest", "benchmark") else "Testing",
            tc.severity_if_fail if tc else "medium",
            (f"{tc.objective} Method: {tc.method} Acceptance: {tc.acceptance}. " if tc else "") + "The model did not pass this check.",
            evidence=[r.message[:1200]], source="independent_test",
            recommendation="Investigate and remediate, or justify why the behaviour is acceptable for the intended use.",
        ))
    dedup: dict[str, Finding] = {}
    for f in out:
        if f.finding_id not in dedup or SEVERITY_ORDER[f.severity] > SEVERITY_ORDER[dedup[f.finding_id].severity]:
            dedup[f.finding_id] = f
    return list(dedup.values())


def challenger_findings(ch: ChallengerComparison | None) -> list[Finding]:
    if ch is None or not ch.challenger_better:
        return []
    m = ch.primary_metric or "primary metric"
    return [_f(
        "F-R07", "Independent challenger outperforms the model", "Performance", "medium",
        f"The validator's challenger achieved {ch.challenger_metrics.get(m, float('nan')):.6g} on {m} "
        f"against {ch.champion_metrics.get(m, float('nan')):.6g} for the model under review.",
        source="challenger",
        recommendation="Explain why the more complex or different approach is preferred, or adopt the stronger specification.",
    )]


def compare_challenger(champion: dict[str, float], challenger: dict[str, float], spec: ModelSpec | None,
                       material_gap: float) -> tuple[str | None, bool | None]:
    primary = next((ac for ac in (spec.acceptance_criteria if spec else []) if ac.primary), None)
    if primary is None:
        return None, None
    a, b = champion.get(primary.metric), challenger.get(primary.metric)
    if a is None or b is None:
        return primary.metric, None
    gap = material_gap * max(abs(a), 1e-12)
    better = (b > a + gap) if primary.higher_is_better else (b < a - gap)
    return primary.metric, bool(better)


def regulatory_gate_findings(spec: ModelSpec | None, metrics: dict[str, float], thresholds: Any
                             ) -> tuple[list[Finding], str]:
    """Run the deterministic CRR/EBA/FRTB gates when the category and inputs allow it.
    Returns (findings, note)."""
    if spec is None or spec.category not in ("credit_risk", "market_risk"):
        return [], ""
    from agents.risk_schemas import CreditModelValidationInputs, NonCreditRiskValidationInputs

    from .regulatory_gates import ValidationGateAgent

    profile = dict(spec.regulatory_profile or {})
    try:
        if spec.category == "credit_risk":
            fields = CreditModelValidationInputs.model_fields
            payload = {k: v for k, v in profile.items() if k in fields}
            payload.setdefault("model_id", _slug(spec.title))
            for k, v in metrics.items():
                if k in fields and k not in payload:
                    payload[k] = int(v) if k.endswith("_count") else v
            inputs = CreditModelValidationInputs.model_validate(payload)
            domain = "credit_risk"
        else:
            market_fields = {
                "var_confidence_level", "var_horizon_days", "var_backtesting_exceptions",
                "var_backtesting_observations", "stressed_var", "expected_shortfall_975",
                "pla_spearman_correlation", "pla_ks_statistic",
            }
            mm = {k: (int(v) if k in ("var_backtesting_exceptions", "var_backtesting_observations", "var_horizon_days") else v)
                  for k, v in metrics.items() if k in market_fields}
            inputs = NonCreditRiskValidationInputs.model_validate({
                "risk_type": "market",
                "business_unit": profile.get("business_unit", spec.title),
                "reporting_date": profile.get("reporting_date", ""),
                "jurisdiction": profile.get("jurisdiction", "EU"),
                "market_metrics": mm,
            })
            domain = "non_credit_risk"
    except ValidationError as exc:
        return [], f"Regulatory gates skipped: inputs incomplete ({str(exc).splitlines()[0]})"

    gate_findings, _passed = ValidationGateAgent(thresholds).run(domain, inputs)  # type: ignore[arg-type]
    out = []
    for g in gate_findings:
        sev = g.severity if g.severity in SEVERITY_ORDER else "medium"
        out.append(_f(
            f"F-G-{_slug(g.finding_reference or g.area)}", g.area, "Regulatory compliance", sev,  # type: ignore[arg-type]
            g.description, recommendation=g.recommendation, source="regulatory_gate",
            citations=[Citation(source=g.regulatory_reference)] if g.regulatory_reference else [],
        ))
    return out, f"Regulatory gates run for {domain}: {len(out)} findings"


def outcome_for(findings: list[Finding]) -> Outcome:
    open_sev = {f.severity for f in findings if f.status == "open"}
    if "critical" in open_sev:
        return "rejected"
    if "high" in open_sev:
        return "remediation_required"
    if "medium" in open_sev:
        return "approved_with_conditions"
    return "approved"


def needs_remediation(outcome: Outcome) -> bool:
    return outcome in ("rejected", "remediation_required")
