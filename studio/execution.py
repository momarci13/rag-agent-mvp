"""Runs generated model code and test suites in a subprocess.

This is a stability boundary, not a security boundary: code runs as the
current user. What it does guarantee:

* a minimal environment with no API keys or other secrets (the OpenAI key is
  never visible to generated code),
* wall-clock timeouts that kill the whole process tree,
* a memory cap on POSIX systems,
* results are read from files the code writes (metrics.json, JUnit XML), not
  from anything the LLM claims.
"""
from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from .schemas import ExecutionResult, PipelineRun, TestResult, TestRun

REPO_ROOT = Path(__file__).resolve().parents[1]
_TEST_ID = re.compile(r"test_?(MT|VT)[_-]?(\d{1,3})", re.IGNORECASE)
_PASS_THROUGH_ENV = (
    "PATH", "SYSTEMROOT", "SystemRoot", "WINDIR", "COMSPEC", "PATHEXT", "TEMP", "TMP", "TMPDIR",
    "HOME", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "LANG", "LC_ALL", "NUMBER_OF_PROCESSORS",
)
PYTEST_INI = "[pytest]\naddopts = -p no:cacheprovider\nfilterwarnings =\n    ignore::DeprecationWarning\n"


def _tail(text: str | bytes | None, limit: int = 6000) -> str:
    if text is None:
        return ""
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="replace")
    return text[-limit:]


def sandbox_env(pythonpath: list[Path], extra: dict[str, str] | None = None) -> dict[str, str]:
    env = {k: os.environ[k] for k in _PASS_THROUGH_ENV if k in os.environ}
    paths = [str(p) for p in pythonpath] + [str(REPO_ROOT)]
    env.update({
        "PYTHONPATH": os.pathsep.join(paths),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONIOENCODING": "utf-8",
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", "2"),
    })
    if extra:
        env.update(extra)
    return env


def run_command(
    args: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_s: int,
    mem_mb: int = 4096,
) -> ExecutionResult:
    cwd.mkdir(parents=True, exist_ok=True)
    kwargs: dict[str, Any] = {}
    if os.name == "posix":
        def _limit() -> None:  # pragma: no cover - runs in the child
            import resource

            os.setsid()
            limit = mem_mb * 1024 * 1024
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))

        kwargs["preexec_fn"] = _limit
    else:  # pragma: no cover - Windows
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP  # type: ignore[attr-defined]

    started = time.monotonic()
    proc = subprocess.Popen(
        args, cwd=str(cwd), env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs
    )
    timed_out = False
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        timed_out = True
        _kill_tree(proc)
        out, err = proc.communicate()
    return ExecutionResult(
        command=" ".join(Path(a).name if i == 0 else a for i, a in enumerate(args)),
        returncode=proc.returncode if not timed_out else -1,
        timed_out=timed_out,
        stdout_tail=_tail(out),
        stderr_tail=_tail(err) + (f"\nTIMEOUT after {timeout_s}s" if timed_out else ""),
        duration_s=round(time.monotonic() - started, 2),
    )


def _kill_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            os.killpg(proc.pid, signal.SIGKILL)
        else:  # pragma: no cover
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
    except (ProcessLookupError, PermissionError, OSError):
        proc.kill()


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

def flatten_metrics(obj: Any, prefix: str = "") -> dict[str, float]:
    """Keep finite numbers only; nested dicts become dotted keys."""
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten_metrics(value, name))
    elif isinstance(obj, bool):
        out[prefix] = float(obj)
    elif isinstance(obj, (int, float)):
        if math.isfinite(float(obj)):
            out[prefix] = float(obj)
    return out


def run_pipeline(
    package_dir: Path,
    *,
    data_dir: Path,
    out_dir: Path,
    timeout_s: int,
    mem_mb: int,
    entrypoint: str = "pipeline.py",
) -> PipelineRun:
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.json"
    if metrics_path.exists():
        metrics_path.unlink()
    if not (package_dir / entrypoint).exists():
        return PipelineRun(
            execution=ExecutionResult(command=entrypoint, returncode=127),
            error=f"{entrypoint} not found in the package",
        )
    env = sandbox_env([package_dir], {"STUDIO_DATA_DIR": str(data_dir), "STUDIO_OUT_DIR": str(out_dir)})
    execution = run_command(
        [sys.executable, entrypoint, "--data-dir", str(data_dir), "--out-dir", str(out_dir)],
        cwd=package_dir, env=env, timeout_s=timeout_s, mem_mb=mem_mb,
    )
    run = PipelineRun(execution=execution)
    if metrics_path.exists():
        try:
            run.metrics = flatten_metrics(json.loads(metrics_path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            run.error = f"metrics.json is not valid JSON: {exc}"
    elif execution.ok:
        run.error = "Pipeline finished but did not write metrics.json"
    if not execution.ok and not run.error:
        run.error = execution.stderr_tail[-2000:] or "Pipeline failed"
    figures_dir = out_dir / "figures"
    if figures_dir.exists():
        run.figures = sorted(str(p) for p in figures_dir.glob("*.png"))
    return run


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def normalise_test_id(name: str) -> str | None:
    m = _TEST_ID.search(name)
    if not m:
        return None
    return f"{m.group(1).upper()}-{int(m.group(2)):02d}"


def parse_junit(path: Path) -> list[TestResult]:
    if not path.exists():
        return []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return []
    results: list[TestResult] = []
    for case in root.iter("testcase"):
        classname = case.get("classname", "")
        name = case.get("name", "")
        nodeid = f"{classname}::{name}" if classname else name
        outcome = "passed"
        message = ""
        for child in case:
            if child.tag in ("failure", "error", "skipped"):
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[child.tag]
                message = (child.get("message") or "") + ("\n" + child.text if child.text else "")
                break
        results.append(TestResult(
            test_id=normalise_test_id(name),
            nodeid=nodeid,
            outcome=outcome,  # type: ignore[arg-type]
            duration_s=float(case.get("time") or 0.0),
            message=message.strip()[:3000],
        ))
    return results


def run_pytest(
    tests_path: Path,
    *,
    cwd: Path,
    pythonpath: list[Path],
    junit_path: Path,
    timeout_s: int,
    mem_mb: int,
    extra_env: dict[str, str] | None = None,
) -> TestRun:
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    if junit_path.exists():
        junit_path.unlink()
    ini = cwd / "pytest.ini"
    if not ini.exists():
        ini.write_text(PYTEST_INI, encoding="utf-8")
    env = sandbox_env(pythonpath, extra_env)
    execution = run_command(
        [
            sys.executable, "-m", "pytest", str(tests_path), "-q", "-rA",
            "-c", str(ini), "--rootdir", str(cwd),
            f"--junitxml={junit_path}", "-o", "junit_family=xunit2",
        ],
        cwd=cwd, env=env, timeout_s=timeout_s, mem_mb=mem_mb,
    )
    run = TestRun(execution=execution, results=parse_junit(junit_path))
    # pytest exit codes: 0 all passed, 1 some failed, 5 no tests collected.
    if execution.returncode not in (0, 1) and not any(r.outcome in ("passed", "failed") for r in run.results):
        run.collection_error = (execution.stdout_tail + "\n" + execution.stderr_tail).strip()[-4000:] or "pytest failed to run"
    return run


def write_files(root: Path, files: list[tuple[str, str]]) -> list[str]:
    """Write (relative_path, content) pairs under ``root``; refuses path escapes."""
    from .storage import resolve_inside

    written: list[str] = []
    for rel, content in files:
        rel = rel.replace("\\", "/").lstrip("/")
        target = resolve_inside(root, rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(rel)
    return written
