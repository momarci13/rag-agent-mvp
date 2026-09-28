"""Safe-ish code sandbox for Python, C++, and C.

This is NOT a security boundary — the LLM is trusted code running on
your machine. It IS a stability boundary: timeouts + memory limits keep
a runaway loop from freezing your laptop.

On Linux we use resource.setrlimit. On Windows we rely on subprocess
timeout + optional psutil memory polling.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


def run_py(
    code: str,
    *,
    timeout_s: int = 60,
    mem_mb: int = 2048,
    workdir: str | None = None,
    extra_env: dict | None = None,
) -> dict:
    """Execute `code` in a subprocess and capture stdout/stderr.

    Returns {stdout, stderr, returncode, timed_out}.
    """
    cwd = Path(workdir) if workdir else Path.cwd()
    cwd.mkdir(parents=True, exist_ok=True)

    # Save code to temp file for better tracebacks
    with tempfile.NamedTemporaryFile("w", suffix=".py", dir=str(cwd), delete=False) as f:
        f.write(code)
        script_path = f.name

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    repo_root = Path(__file__).resolve().parents[1]
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(repo_root) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    preexec = None
    if os.name == "posix":
        def _limit():
            import resource
            resource.setrlimit(
                resource.RLIMIT_AS,
                (mem_mb * 1024 * 1024, mem_mb * 1024 * 1024),
            )
        preexec = _limit

    timed_out = False
    try:
        r = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(cwd),
            env=env,
            preexec_fn=preexec,
        )
        result = {
            "stdout": r.stdout,
            "stderr": r.stderr,
            "returncode": r.returncode,
            "timed_out": False,
        }
    except subprocess.TimeoutExpired as e:
        timed_out = True
        result = {
            "stdout": (e.stdout or b"").decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or ""),
            "stderr": f"TIMEOUT after {timeout_s}s\n"
                      + ((e.stderr or b"").decode(errors="ignore") if isinstance(e.stderr, bytes) else (e.stderr or "")),
            "returncode": -1,
            "timed_out": True,
        }
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass
    return result


def run_cpp(
    code: str,
    *,
    timeout_s: int = 60,
    mem_mb: int = 2048,
    workdir: str | None = None,
    extra_env: dict | None = None,
) -> dict:
    """Compile and run C++ code. Returns {stdout, stderr, returncode, timed_out}.

    Requires g++ on PATH (e.g. MSYS2/ucrt64 on Windows, gcc package on Linux).
    Compiles with -O3 -std=c++17 -march=native -Wall.
    """
    compiler = shutil.which("g++")
    if not compiler:
        return {
            "stdout": "",
            "stderr": "[Compilation failed]\ng++ not found on PATH. Install MSYS2 (Windows) or gcc package (Linux).",
            "returncode": 1,
            "timed_out": False,
        }

    cwd = Path(workdir) if workdir else Path.cwd()
    cwd.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".cpp", dir=str(cwd), delete=False) as f:
        f.write(code)
        src_path = f.name

    exe_path = str(Path(src_path).with_suffix(".exe" if os.name == "nt" else ".out"))

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    try:
        compile_r = subprocess.run(
            [compiler, "-O3", "-std=c++17", "-march=native", "-Wall",
             src_path, "-o", exe_path, "-lm"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(cwd),
            env=env,
        )
    except subprocess.TimeoutExpired:
        try:
            os.unlink(src_path)
        except OSError:
            pass
        return {"stdout": "", "stderr": "[Compilation failed]\nCompiler timed out.", "returncode": 1, "timed_out": False}

    try:
        os.unlink(src_path)
    except OSError:
        pass

    if compile_r.returncode != 0:
        return {
            "stdout": "",
            "stderr": f"[Compilation failed]\n{compile_r.stderr}",
            "returncode": 1,
            "timed_out": False,
        }

    try:
        r = subprocess.run(
            [exe_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(cwd),
            env=env,
        )
        result = {"stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode, "timed_out": False}
    except subprocess.TimeoutExpired as e:
        result = {
            "stdout": (e.stdout or b"").decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or ""),
            "stderr": f"TIMEOUT after {timeout_s}s\n"
                      + ((e.stderr or b"").decode(errors="ignore") if isinstance(e.stderr, bytes) else (e.stderr or "")),
            "returncode": -1,
            "timed_out": True,
        }
    finally:
        try:
            os.unlink(exe_path)
        except OSError:
            pass
    return result


def run_c(
    code: str,
    *,
    timeout_s: int = 60,
    mem_mb: int = 2048,
    workdir: str | None = None,
    extra_env: dict | None = None,
) -> dict:
    """Compile and run C code. Returns {stdout, stderr, returncode, timed_out}.

    Requires gcc on PATH. Compiles with -O3 -std=c11 -march=native -Wall.
    """
    compiler = shutil.which("gcc")
    if not compiler:
        return {
            "stdout": "",
            "stderr": "[Compilation failed]\ngcc not found on PATH. Install MSYS2 (Windows) or gcc package (Linux).",
            "returncode": 1,
            "timed_out": False,
        }

    cwd = Path(workdir) if workdir else Path.cwd()
    cwd.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile("w", suffix=".c", dir=str(cwd), delete=False) as f:
        f.write(code)
        src_path = f.name

    exe_path = str(Path(src_path).with_suffix(".exe" if os.name == "nt" else ".out"))

    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)

    try:
        compile_r = subprocess.run(
            [compiler, "-O3", "-std=c11", "-march=native", "-Wall",
             src_path, "-o", exe_path, "-lm"],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(cwd),
            env=env,
        )
    except subprocess.TimeoutExpired:
        try:
            os.unlink(src_path)
        except OSError:
            pass
        return {"stdout": "", "stderr": "[Compilation failed]\nCompiler timed out.", "returncode": 1, "timed_out": False}

    try:
        os.unlink(src_path)
    except OSError:
        pass

    if compile_r.returncode != 0:
        return {
            "stdout": "",
            "stderr": f"[Compilation failed]\n{compile_r.stderr}",
            "returncode": 1,
            "timed_out": False,
        }

    try:
        r = subprocess.run(
            [exe_path],
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=str(cwd),
            env=env,
        )
        result = {"stdout": r.stdout, "stderr": r.stderr, "returncode": r.returncode, "timed_out": False}
    except subprocess.TimeoutExpired as e:
        result = {
            "stdout": (e.stdout or b"").decode(errors="ignore") if isinstance(e.stdout, bytes) else (e.stdout or ""),
            "stderr": f"TIMEOUT after {timeout_s}s\n"
                      + ((e.stderr or b"").decode(errors="ignore") if isinstance(e.stderr, bytes) else (e.stderr or "")),
            "returncode": -1,
            "timed_out": True,
        }
    finally:
        try:
            os.unlink(exe_path)
        except OSError:
            pass
    return result


# Alias used by task_conversation
run_code_sync = run_py
