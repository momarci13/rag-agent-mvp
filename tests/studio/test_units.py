from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from studio.execution import flatten_metrics, normalise_test_id, run_pytest
from studio.frameworks import checklist_for
from studio.library import _markdown_segments
from studio.rules import (
    acceptance_findings,
    compare_challenger,
    compare_metrics,
    outcome_for,
    regulatory_gate_findings,
    test_suite_findings as suite_findings,
)
from studio.regulatory_gates import ValidationThresholds
from studio.schemas import AcceptanceCriterion, Finding, ModelSpec, TestCase, TestResult, TestRun, ExecutionResult
from studio.storage import ProjectStore, resolve_inside, safe_filename
from studio.validator import _safe_extract


def _spec(**kw) -> ModelSpec:
    base = dict(title="PD", category="credit_risk", objective="o", intended_use="u", target="t",
                data_description="d", methodology_summary="m",
                acceptance_criteria=[AcceptanceCriterion(metric="auc", operator=">=", threshold=0.7, primary=True)],
                test_plan=[TestCase(test_id="MT-01", name="a", category="unit", objective="o", method="m", acceptance="x", severity_if_fail="high"),
                           TestCase(test_id="MT-02", name="b", category="unit", objective="o", method="m", acceptance="x")])
    base.update(kw)
    return ModelSpec(**base)


def _finding(sev: str, status: str = "open") -> Finding:
    return Finding(finding_id=f"F-{sev}", title="t", area="a", severity=sev, description="d", status=status)


def test_outcome_rules():
    assert outcome_for([]) == "approved"
    assert outcome_for([_finding("low")]) == "approved"
    assert outcome_for([_finding("medium")]) == "approved_with_conditions"
    assert outcome_for([_finding("high"), _finding("medium")]) == "remediation_required"
    assert outcome_for([_finding("critical")]) == "rejected"
    assert outcome_for([_finding("critical", "closed")]) == "approved"


def test_acceptance_and_primary_severity():
    spec = _spec()
    assert acceptance_findings(spec, {"auc": 0.8}, True) == []
    [f] = acceptance_findings(spec, {"auc": 0.6}, True)
    assert f.severity == "high" and f.finding_id == "F-R05-auc"
    [f] = acceptance_findings(spec, {}, True)
    assert "not produced" in f.title


def test_suite_findings_cover_missing_failed_and_errors():
    run = TestRun(execution=ExecutionResult(command="pytest", returncode=1), results=[
        TestResult(test_id="MT-01", nodeid="t::test_MT01", outcome="failed", message="boom"),
        TestResult(test_id=None, nodeid="t::test_extra", outcome="error", message="err"),
    ])
    ids = {f.finding_id: f for f in suite_findings(_spec(), run, external=False)}
    assert ids["F-R04-MT-01"].severity == "high"
    assert ids["F-R03-MT-02"].severity == "medium"
    assert ids["F-R04-test-extra"].severity == "high"
    assert "F-R03-MT-02" not in {f.finding_id for f in suite_findings(_spec(), run, external=True)}


def test_metric_comparison_and_challenger():
    compared, mism = compare_metrics({"a": 1.0, "b": 2.0, "c": 3.0}, {"a": 1.0, "b": 2.1}, 1e-6, 1e-9)
    assert mism == ["b", "c"] and compared["a"]["abs_diff"] == 0
    assert compare_challenger({"auc": 0.70}, {"auc": 0.75}, _spec(), 0.02) == ("auc", True)
    assert compare_challenger({"auc": 0.70}, {"auc": 0.71}, _spec(), 0.02) == ("auc", False)


def test_regulatory_gate_mapping():
    spec = _spec(regulatory_profile={"model_type": "PD", "exposure_class": "retail", "portfolio_segment": "mortgages",
                                     "estimation_approach": "internal_ratings_based"})
    findings, note = regulatory_gate_findings(spec, {"gini_coefficient": 0.2}, ValidationThresholds())
    assert any(f.finding_id == "F-G-credit-risk-gini" and f.source == "regulatory_gate" for f in findings)
    findings, note = regulatory_gate_findings(_spec(), {"gini_coefficient": 0.2}, ValidationThresholds())
    assert findings == [] and "skipped" in note


def test_checklists():
    ids = {r.req_id for r in checklist_for(["eu_banking"], "market_risk")}
    assert "GEN-01" in ids and "EUB-08" in ids and "EUB-02" not in ids
    assert any(r.req_id == "AIA-03" for r in checklist_for(["eu_ai_act"], "ai_ml"))


def test_flatten_and_test_ids():
    assert flatten_metrics({"a": 1, "b": {"c": 2.5}, "s": "x", "n": float("nan"), "t": True}) == {"a": 1.0, "b.c": 2.5, "t": 1.0}
    assert normalise_test_id("test_MT03_calibration") == "MT-03"
    assert normalise_test_id("test_vt_12_x") == "VT-12"
    assert normalise_test_id("test_other") is None


def test_path_safety(tmp_path: Path):
    assert safe_filename("../../etc/passwd") == "passwd"
    with pytest.raises(ValueError):
        resolve_inside(tmp_path, "../outside.txt")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("../evil.py", "x")
    (tmp_path / "evil.zip").write_bytes(buf.getvalue())
    with pytest.raises(ValueError):
        _safe_extract(tmp_path / "evil.zip", tmp_path / "out")
    with pytest.raises(ValueError):
        ProjectStore(tmp_path).project_dir("../../x")


def test_markdown_segments_keep_headings():
    segs = _markdown_segments("# A\ntext a\n## B\ntext b\n")
    assert segs == [("section 'A'", "text a"), ("section 'B'", "text b")]


def test_sandbox_hides_secrets_and_times_out(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_env.py").write_text(
        "import os, time\n"
        "def test_MT01_no_key():\n    assert 'OPENAI_API_KEY' not in os.environ\n"
        "def test_MT02_slow():\n    time.sleep(30)\n"
    )
    run = run_pytest(Path("tests"), cwd=tmp_path, pythonpath=[tmp_path], junit_path=tmp_path / "j.xml",
                     timeout_s=8, mem_mb=2048)
    assert run.execution.timed_out
    ok = run_pytest(Path("tests/test_env.py::test_MT01_no_key"), cwd=tmp_path, pythonpath=[tmp_path],
                    junit_path=tmp_path / "j2.xml", timeout_s=60, mem_mb=2048)
    assert [r.outcome for r in ok.results] == ["passed"]
