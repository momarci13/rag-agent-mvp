"""Standalone tests for tools/risk_storage.py::RiskStorage."""

import time

import pytest

from agents.risk_schemas import RiskValidationRun, ValidationReport
from tools.risk_storage import RiskStorage


@pytest.fixture
def storage(tmp_path):
    return RiskStorage(tmp_path / "risk_validation.db")


def _run_with_report(entity: str, domain: str = "credit_risk", rating: str = "compliant") -> RiskValidationRun:
    report = ValidationReport(
        domain=domain, title="t", scope="s", methodology="m",
        entity_under_review=entity, reporting_period="p", overall_rating=rating,
    )
    return RiskValidationRun(domain=domain, report=report)


def test_save_and_load_run_round_trips(storage):
    run = _run_with_report("PD-RETAIL-01")
    storage.save_run(run)
    loaded = storage.load_run(run.run_id)
    assert loaded is not None
    assert loaded.run_id == run.run_id
    assert loaded.report.entity_under_review == "PD-RETAIL-01"


def test_load_run_returns_none_for_unknown_id(storage):
    assert storage.load_run("does-not-exist") is None


def test_save_run_upserts_on_conflict(storage):
    run = _run_with_report("PD-RETAIL-01")
    storage.save_run(run)
    run.gate_passed = True
    storage.save_run(run)
    loaded = storage.load_run(run.run_id)
    assert loaded.gate_passed is True


def test_get_history_orders_chronologically_and_respects_limit(storage):
    for rating in ["compliant", "partially_compliant", "non_compliant", "compliant"]:
        run = _run_with_report("PD-RETAIL-01", rating=rating)
        storage.save_run(run)
        time.sleep(0.01)  # ensure created_at ordering is unambiguous
    history = storage.get_history("PD-RETAIL-01", "credit_risk", limit=3)
    assert len(history) == 3
    # most recent last (chronological order)
    assert [r.overall_rating for r in history] == ["partially_compliant", "non_compliant", "compliant"]


def test_get_history_filters_by_entity_and_domain(storage):
    storage.save_run(_run_with_report("PD-RETAIL-01", domain="credit_risk"))
    storage.save_run(_run_with_report("PD-RETAIL-01", domain="model_risk"))
    storage.save_run(_run_with_report("OTHER-MODEL", domain="credit_risk"))
    history = storage.get_history("PD-RETAIL-01", "credit_risk")
    assert len(history) == 1


def test_list_runs_filters_by_domain(storage):
    storage.save_run(_run_with_report("A", domain="credit_risk"))
    storage.save_run(_run_with_report("B", domain="model_risk"))
    runs = storage.list_runs(domain="model_risk")
    assert len(runs) == 1
    assert runs[0].domain == "model_risk"


def test_approval_issue_and_consume(storage):
    run = _run_with_report("PD-RETAIL-01")
    storage.save_run(run)
    token, expires_at = storage.issue_approval(run.run_id, run.report)
    assert expires_at > time.time()
    storage.consume_approval(run.run_id, run.report, token)  # should not raise


def test_approval_rejects_unknown_token(storage):
    run = _run_with_report("PD-RETAIL-01")
    storage.save_run(run)
    with pytest.raises(PermissionError, match="Unknown"):
        storage.consume_approval(run.run_id, run.report, "not-a-real-token")


def test_approval_rejects_reuse(storage):
    run = _run_with_report("PD-RETAIL-01")
    storage.save_run(run)
    token, _ = storage.issue_approval(run.run_id, run.report)
    storage.consume_approval(run.run_id, run.report, token)
    with pytest.raises(PermissionError, match="already used"):
        storage.consume_approval(run.run_id, run.report, token)


def test_approval_rejects_mismatched_report(storage):
    run = _run_with_report("PD-RETAIL-01")
    storage.save_run(run)
    token, _ = storage.issue_approval(run.run_id, run.report)
    mutated = run.report.model_copy(update={"overall_conclusion": "changed after issuance"})
    with pytest.raises(PermissionError):
        storage.consume_approval(run.run_id, mutated, token)


def test_approval_rejects_expired_token(storage):
    run = _run_with_report("PD-RETAIL-01")
    storage.save_run(run)
    token, _ = storage.issue_approval(run.run_id, run.report, ttl_s=0)
    time.sleep(0.05)
    with pytest.raises(PermissionError, match="expired"):
        storage.consume_approval(run.run_id, run.report, token)


def test_signoffs_are_recorded_and_retrievable(storage):
    run = _run_with_report("PD-RETAIL-01")
    storage.save_run(run)
    storage.add_signoff(run.run_id, "jane", role="preparer")
    storage.add_signoff(run.run_id, "john", role="reviewer")
    signoffs = storage.get_signoffs(run.run_id)
    assert [s.by for s in signoffs] == ["jane", "john"]
    assert signoffs[1].role == "reviewer"


def test_audit_log_records_entries_in_order(storage):
    run = _run_with_report("PD-RETAIL-01")
    storage.save_run(run)
    storage.record_audit(run.run_id, "system", "run_created", "credit_risk run")
    storage.record_audit(run.run_id, "jane", "approval_consumed")
    log = storage.get_audit_log(run.run_id)
    assert [entry["action"] for entry in log] == ["run_created", "approval_consumed"]
    assert log[0]["detail"] == "credit_risk run"
