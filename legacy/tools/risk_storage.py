"""SQLite-backed persistence for the risk-validation agent team.

Deliberately standalone from tools/approval.py::ApprovalStore, which stays
in-memory and keeps serving the unrelated IBKR trading flow unchanged. This
module gives the risk-validation pipeline runs, approvals, sign-offs, and an
audit log that all survive a server restart -- the biggest "prototype-grade"
gap flagged after the v1 build.

Uses only the Python standard library (``sqlite3``) -- no new dependency.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import secrets
import sqlite3
import threading
import time
from pathlib import Path

from agents.risk_schemas import RiskValidationRun, SignoffRecord, ValidationReport

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    domain TEXT NOT NULL,
    entity_under_review TEXT NOT NULL DEFAULT '',
    gate_passed INTEGER NOT NULL DEFAULT 0,
    run_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_runs_entity_domain ON runs (entity_under_review, domain, created_at);

CREATE TABLE IF NOT EXISTS approvals (
    token_hash TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    report_fingerprint TEXT NOT NULL,
    expires_at REAL NOT NULL,
    consumed INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS signoffs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    by TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT '',
    at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    actor TEXT NOT NULL DEFAULT '',
    action TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    at TEXT NOT NULL
);
"""


def _now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _fingerprint(report: ValidationReport) -> str:
    payload = report.model_dump_json(exclude_none=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class RiskStorage:
    """SQLite-backed store for risk-validation runs, approvals, sign-offs,
    and an audit log. One connection is opened per call (SQLite handles
    cross-thread access fine that way); a process-local lock serializes
    writes on top of SQLite's own file locking."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._lock, self._connect() as conn:
            conn.executescript(_SCHEMA_SQL)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), timeout=30)
        conn.row_factory = sqlite3.Row
        return conn

    # ---------- runs ----------

    def save_run(self, run: RiskValidationRun) -> None:
        entity = run.report.entity_under_review if run.report else ""
        now = _now_iso()
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO runs (run_id, domain, entity_under_review, gate_passed, run_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    entity_under_review = excluded.entity_under_review,
                    gate_passed = excluded.gate_passed,
                    run_json = excluded.run_json,
                    updated_at = excluded.updated_at
                """,
                (run.run_id, run.domain, entity, int(run.gate_passed), run.model_dump_json(), now, now),
            )

    def load_run(self, run_id: str) -> RiskValidationRun | None:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT run_json FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            return None
        return RiskValidationRun.model_validate_json(row["run_json"])

    def list_runs(self, *, domain: str | None = None, since: str | None = None) -> list[RiskValidationRun]:
        query = "SELECT run_json FROM runs WHERE 1=1"
        params: list[str] = []
        if domain is not None:
            query += " AND domain = ?"
            params.append(domain)
        if since is not None:
            query += " AND created_at >= ?"
            params.append(since)
        query += " ORDER BY created_at DESC"
        with self._lock, self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [RiskValidationRun.model_validate_json(row["run_json"]) for row in rows]

    def get_history(self, entity_under_review: str, domain: str, limit: int = 5) -> list[ValidationReport]:
        """Prior validation cycles for the same entity, most recent last
        (chronological order, convenient for trend checks)."""
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_json FROM runs
                WHERE entity_under_review = ? AND domain = ?
                ORDER BY created_at DESC LIMIT ?
                """,
                (entity_under_review, domain, limit),
            ).fetchall()
        reports = [
            run.report
            for row in rows
            if (run := RiskValidationRun.model_validate_json(row["run_json"])).report is not None
        ]
        return list(reversed(reports))

    # ---------- approvals (persisted equivalent of tools/approval.py::ApprovalStore) ----------

    def issue_approval(self, run_id: str, report: ValidationReport, ttl_s: int = 1800) -> tuple[str, float]:
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        expires_at = time.time() + ttl_s
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO approvals (token_hash, run_id, report_fingerprint, expires_at, consumed) VALUES (?, ?, ?, ?, 0)",
                (token_hash, run_id, _fingerprint(report), expires_at),
            )
        return token, expires_at

    def consume_approval(self, run_id: str, report: ValidationReport, token: str) -> None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT * FROM approvals WHERE token_hash = ?", (token_hash,)).fetchone()
            if row is None:
                raise PermissionError("Unknown approval token")
            if row["run_id"] != run_id:
                raise PermissionError("Approval token does not match this run")
            if row["consumed"]:
                raise PermissionError("Approval token was already used")
            if time.time() > row["expires_at"]:
                raise PermissionError("Approval token expired")
            if row["report_fingerprint"] != _fingerprint(report):
                raise PermissionError("Approval token does not match this report")
            conn.execute("UPDATE approvals SET consumed = 1 WHERE token_hash = ?", (token_hash,))

    # ---------- sign-offs ----------

    def add_signoff(self, run_id: str, by: str, role: str = "") -> SignoffRecord:
        record = SignoffRecord(by=by, role=role)
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO signoffs (run_id, by, role, at) VALUES (?, ?, ?, ?)",
                (run_id, record.by, record.role, record.at),
            )
        return record

    def get_signoffs(self, run_id: str) -> list[SignoffRecord]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT by, role, at FROM signoffs WHERE run_id = ? ORDER BY id ASC", (run_id,)
            ).fetchall()
        return [SignoffRecord(by=row["by"], role=row["role"], at=row["at"]) for row in rows]

    # ---------- audit log ----------

    def record_audit(self, run_id: str | None, actor: str, action: str, detail: str = "") -> None:
        with self._lock, self._connect() as conn:
            conn.execute(
                "INSERT INTO audit_log (run_id, actor, action, detail, at) VALUES (?, ?, ?, ?, ?)",
                (run_id, actor, action, detail, _now_iso()),
            )

    def get_audit_log(self, run_id: str) -> list[dict]:
        with self._lock, self._connect() as conn:
            rows = conn.execute(
                "SELECT actor, action, detail, at FROM audit_log WHERE run_id = ? ORDER BY id ASC", (run_id,)
            ).fetchall()
        return [dict(row) for row in rows]
