"""Concurrency tests for tools/risk_storage.py::RiskStorage.

RiskStorage opens a fresh sqlite3 connection per call and serializes writes
with a process-local lock on top of SQLite's own file locking. These tests
spin up N threads hammering the same store concurrently and assert no writes
are lost and no exceptions escape (e.g. "database is locked" errors).
"""

import threading

from agents.risk_schemas import RiskValidationRun, ValidationReport
from tools.risk_storage import RiskStorage

N_THREADS = 16
N_RUNS_PER_THREAD = 5


def _make_run(entity: str, domain: str = "credit_risk") -> RiskValidationRun:
    report = ValidationReport(
        domain=domain, title="t", scope="s", methodology="m",
        entity_under_review=entity, reporting_period="p", overall_rating="compliant",
    )
    return RiskValidationRun(domain=domain, report=report)


def test_concurrent_save_run_does_not_lose_writes(tmp_path):
    storage = RiskStorage(tmp_path / "risk_validation.db")
    errors: list[Exception] = []
    run_ids: list[str] = []
    lock = threading.Lock()

    def worker(thread_idx: int) -> None:
        try:
            for i in range(N_RUNS_PER_THREAD):
                run = _make_run(f"entity-{thread_idx}")
                storage.save_run(run)
                with lock:
                    run_ids.append(run.run_id)
        except Exception as exc:  # pragma: no cover - failure path
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Concurrent writes raised: {errors}"
    assert len(run_ids) == N_THREADS * N_RUNS_PER_THREAD
    assert len(set(run_ids)) == len(run_ids)  # no ID collisions
    for run_id in run_ids:
        assert storage.load_run(run_id) is not None


def test_concurrent_audit_log_writes_preserve_all_entries(tmp_path):
    storage = RiskStorage(tmp_path / "risk_validation.db")
    run = _make_run("entity-shared")
    storage.save_run(run)
    errors: list[Exception] = []

    def worker(thread_idx: int) -> None:
        try:
            for i in range(N_RUNS_PER_THREAD):
                storage.record_audit(run.run_id, f"actor-{thread_idx}", "test_action", detail=f"{thread_idx}-{i}")
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Concurrent audit writes raised: {errors}"
    log = storage.get_audit_log(run.run_id)
    assert len(log) == N_THREADS * N_RUNS_PER_THREAD


def test_concurrent_approval_issue_and_consume_only_one_winner(tmp_path):
    """Only one of several concurrent consume() calls with distinct tokens
    should succeed per issued token; each token is single-use regardless of
    concurrent contention."""
    storage = RiskStorage(tmp_path / "risk_validation.db")
    run = _make_run("entity-approval")
    storage.save_run(run)
    token, _ = storage.issue_approval(run.run_id, run.report)

    successes: list[bool] = []
    errors: list[Exception] = []
    lock = threading.Lock()

    def worker() -> None:
        try:
            storage.consume_approval(run.run_id, run.report, token)
            with lock:
                successes.append(True)
        except PermissionError:
            pass
        except Exception as exc:  # pragma: no cover - failure path
            with lock:
                errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    assert not errors, f"Concurrent approval consumption raised unexpected errors: {errors}"
    assert len(successes) == 1  # exactly one thread won the race
