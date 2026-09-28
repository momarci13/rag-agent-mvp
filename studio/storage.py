"""On-disk layout and persistence for studio projects.

    <output_dir>/<project_id>/
        project.json
        data/                      uploaded datasets (read-only inputs)
        attachments/               concept papers, regulations
        external/                  external model package (validate_external)
        rounds/<n>/modeler/package/   model code, tests, results
        rounds/<n>/modeler/           modelling document
        rounds/<n>/validator/         replication copy, independent tests,
                                      challenger, validation report
"""
from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
import threading
from pathlib import Path

from .schemas import Project

_SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")
_ID = re.compile(r"^[a-f0-9]{12}$")


def safe_filename(name: str) -> str:
    base = Path(name.replace("\\", "/")).name
    cleaned = _SAFE_NAME.sub("_", base).strip("._") or "file"
    return cleaned[:120]


def resolve_inside(root: Path, relative: str) -> Path:
    """Resolve ``relative`` under ``root`` and refuse anything that escapes it."""
    root = root.resolve()
    target = (root / relative).resolve()
    if target != root and root not in target.parents:
        raise ValueError(f"Path escapes project directory: {relative}")
    return target


class ProjectStore:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    # ---------- paths ----------
    def project_dir(self, project_id: str) -> Path:
        if not _ID.match(project_id):
            raise ValueError("Invalid project id")
        return self.root / project_id

    def data_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "data"

    def attachments_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "attachments"

    def external_dir(self, project_id: str) -> Path:
        return self.project_dir(project_id) / "external"

    def round_dir(self, project_id: str, n: int) -> Path:
        return self.project_dir(project_id) / "rounds" / str(n)

    def modeler_dir(self, project_id: str, n: int) -> Path:
        return self.round_dir(project_id, n) / "modeler"

    def package_dir(self, project_id: str, n: int) -> Path:
        return self.modeler_dir(project_id, n) / "package"

    def validator_dir(self, project_id: str, n: int) -> Path:
        return self.round_dir(project_id, n) / "validator"

    def relpath(self, project_id: str, path: Path) -> str:
        return Path(path).resolve().relative_to(self.project_dir(project_id).resolve()).as_posix()

    # ---------- persistence ----------
    def create(self, project: Project) -> Path:
        pdir = self.project_dir(project.project_id)
        for sub in ("data", "attachments", "external", "rounds"):
            (pdir / sub).mkdir(parents=True, exist_ok=True)
        self.save(project)
        return pdir

    def save(self, project: Project) -> None:
        import datetime as dt

        project.updated_at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        pdir = self.project_dir(project.project_id)
        pdir.mkdir(parents=True, exist_ok=True)
        payload = project.model_dump_json(indent=2)
        with self._lock:
            fd, tmp = tempfile.mkstemp(dir=str(pdir), prefix=".project-", suffix=".json")
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(payload)
            os.replace(tmp, pdir / "project.json")

    def load(self, project_id: str) -> Project:
        path = self.project_dir(project_id) / "project.json"
        if not path.exists():
            raise FileNotFoundError(project_id)
        return Project.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[Project]:
        out: list[Project] = []
        for child in self.root.iterdir() if self.root.exists() else []:
            if child.is_dir() and _ID.match(child.name) and (child / "project.json").exists():
                try:
                    out.append(self.load(child.name))
                except (ValueError, json.JSONDecodeError):
                    continue
        out.sort(key=lambda p: p.created_at, reverse=True)
        return out

    # ---------- files ----------
    def save_upload(self, dest_dir: Path, filename: str, content: bytes) -> Path:
        dest_dir.mkdir(parents=True, exist_ok=True)
        name = safe_filename(filename)
        target = dest_dir / name
        stem, suffix = target.stem, target.suffix
        i = 1
        while target.exists():
            target = dest_dir / f"{stem}_{i}{suffix}"
            i += 1
        target.write_bytes(content)
        return target

    @staticmethod
    def copy_tree(src: Path, dst: Path, *, exclude: tuple[str, ...] = ("results", "__pycache__", ".pytest_cache")) -> None:
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst, ignore=shutil.ignore_patterns(*exclude))
