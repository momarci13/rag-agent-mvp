"""Runs a project: Modeler and Validator rounds until the model passes or the
round budget is spent, then waits for a human sign-off."""
from __future__ import annotations

import datetime as dt
import logging
import traceback
from pathlib import Path
from typing import Any

from .config import StudioSettings
from .context import AgentContext, ProgressFn
from .dataprep import DATA_EXTENSIONS, profile_directory
from .library import LIBRARY_EXTENSIONS, Library
from .llm_io import JsonLLM
from .modeler import ModelerAgent
from .rules import needs_remediation
from .schemas import Project, ProjectInput, Round, Signoff, TraceEvent
from .storage import ProjectStore
from .validator import Subject, ValidatorAgent

log = logging.getLogger(__name__)
PACKAGE_EXTENSIONS = {".py", ".zip", ".pdf", ".docx", ".md", ".txt", ".json", ".yaml", ".yml", ".cfg", ".toml", ".ini"}


class InputError(ValueError):
    """The request cannot start a project (missing or unsupported files)."""


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


class Studio:
    def __init__(self, cfg: dict[str, Any], llm: JsonLLM, library: Library | None,
                 store: ProjectStore, settings: StudioSettings) -> None:
        self.cfg = cfg
        self.llm = llm
        self.library = library
        self.store = store
        self.settings = settings

    # ------------------------------------------------------------------
    def create_project(
        self,
        inp: ProjectInput,
        *,
        data: list[tuple[str, bytes]],
        regulations: list[tuple[str, bytes]] = (),  # type: ignore[assignment]
        concept_papers: list[tuple[str, bytes]] = (),  # type: ignore[assignment]
        package: list[tuple[str, bytes]] = (),  # type: ignore[assignment]
    ) -> Project:
        for name, _ in data:
            if Path(name).suffix.lower() not in DATA_EXTENSIONS:
                raise InputError(f"{name}: data must be one of {sorted(DATA_EXTENSIONS)}")
        for name, _ in list(regulations) + list(concept_papers):
            if Path(name).suffix.lower() not in LIBRARY_EXTENSIONS:
                raise InputError(f"{name}: documents must be one of {sorted(LIBRARY_EXTENSIONS)}")
        for name, _ in package:
            if Path(name).suffix.lower() not in PACKAGE_EXTENSIONS:
                raise InputError(f"{name}: package files must be one of {sorted(PACKAGE_EXTENSIONS)}")
        if not data:
            raise InputError("Upload at least one data file.")
        if inp.mode == "validate_external" and not package:
            raise InputError("Upload the model package (code and documentation) to validate.")
        if inp.mode == "validate_external":
            inp.max_rounds = 1

        project = Project(input=inp)
        self.store.create(project)
        pid = project.project_id
        inp.data_files = [self.store.save_upload(self.store.data_dir(pid), n, b).name for n, b in data]
        att_dir = self.store.attachments_dir(pid)
        inp.attachment_files = (
            [f"regulation/{self.store.save_upload(att_dir / 'regulation', n, b).name}" for n, b in regulations]
            + [f"concept_paper/{self.store.save_upload(att_dir / 'concept_paper', n, b).name}" for n, b in concept_papers]
        )
        inp.package_files = [self.store.save_upload(self.store.external_dir(pid), n, b).name for n, b in package]
        self.store.save(project)
        return project

    # ------------------------------------------------------------------
    def run(self, project_id: str, progress: ProgressFn | None = None) -> Project:
        project = self.store.load(project_id)
        ctx = AgentContext(
            llm=self.llm, library=self.library, settings=self.settings, store=self.store,
            project=project, progress=progress or (lambda _e: None), cfg=self.cfg,
        )
        project.status = "running"
        project.error = ""
        self.store.save(project)
        try:
            self._ingest_attachments(ctx)
            self._profile_data(ctx)
            if project.input.mode == "validate_external":
                self._run_external(ctx)
            else:
                self._run_develop(ctx)
            last = project.current_round()
            project.final_outcome = last.validator.outcome if last else None
            project.status = "awaiting_signoff"
            ctx.emit("system", "complete", "ok", f"Final outcome {project.final_outcome}; awaiting human sign-off")
        except Exception as exc:  # noqa: BLE001 - any failure ends the run visibly
            log.exception("Project %s failed", project_id)
            project.status = "failed"
            project.error = f"{type(exc).__name__}: {exc}"
            ctx.emit("system", "failed", "failed", project.error + "\n" + traceback.format_exc()[-1500:])
        self.store.save(project)
        return project

    def _ingest_attachments(self, ctx: AgentContext) -> None:
        project = ctx.project
        if not project.input.attachment_files or self.library is None:
            return
        ctx.emit("system", "library", "started", f"Adding {len(project.input.attachment_files)} documents to the shared library")
        att_dir = self.store.attachments_dir(project.project_id)
        ids: list[str] = []
        for rel in project.input.attachment_files:
            kind, _, _name = rel.partition("/")
            doc = self.library.add_document(att_dir / rel, kind=kind, project_id=project.project_id)
            ids.append(doc.doc_id)
        project.attachment_doc_ids = sorted(set(ids))
        ctx.emit("system", "library", "ok", f"{len(ids)} documents available with page and section citations")

    def _profile_data(self, ctx: AgentContext) -> None:
        ctx.project.data_profile = profile_directory(ctx.data_dir, self.settings.data_sample_rows)
        tables = ctx.project.data_profile.get("tables", {})
        errors = ctx.project.data_profile.get("errors", {})
        if not tables:
            raise InputError(f"No readable data tables. {errors}")
        ctx.emit("system", "data", "ok" if not errors else "warning",
                 f"{len(tables)} tables profiled" + (f"; unreadable: {', '.join(errors)}" if errors else ""))

    def _run_develop(self, ctx: AgentContext) -> None:
        project = ctx.project
        modeler, validator = ModelerAgent(ctx), ValidatorAgent(ctx)
        previous: Round | None = None
        for n in range(1, project.input.max_rounds + 1):
            rnd = Round(number=n)
            project.rounds.append(rnd)
            ctx.emit("system", "round", "started", f"Round {n} of at most {project.input.max_rounds}")
            open_findings = [f for f in previous.validator.findings if f.status == "open"] if previous else []
            modeler.run_round(rnd, previous, open_findings)
            validator.run_round(rnd, previous)
            rnd.finished_at = _now()
            self.store.save(project)
            if not needs_remediation(rnd.validator.outcome):  # type: ignore[arg-type]
                break
            if n < project.input.max_rounds:
                ctx.emit("system", "round", "ok", f"Outcome {rnd.validator.outcome}: returning findings to the Modeler")
            previous = rnd
        else:
            ctx.emit("system", "round", "warning", "Round budget spent with blocking findings still open; handing over to you")

    def _run_external(self, ctx: AgentContext) -> None:
        rnd = Round(number=1)
        ctx.project.rounds.append(rnd)
        ValidatorAgent(ctx).run_round(rnd, None)
        rnd.finished_at = _now()

    # ------------------------------------------------------------------
    def signoff(self, project_id: str, signoff: Signoff) -> Project:
        project = self.store.load(project_id)
        if project.status not in ("awaiting_signoff", "signed_off"):
            raise InputError(f"Project is {project.status}; sign-off is possible once the agents have finished.")
        if any(s.name.strip().lower() == signoff.name.strip().lower() for s in project.signoffs):
            raise InputError(f"{signoff.name} has already signed off this project.")
        project.signoffs.append(signoff)
        project.status = "signed_off"
        rnd = project.current_round()
        if rnd is not None:
            ctx = AgentContext(llm=self.llm, library=self.library, settings=self.settings,
                               store=self.store, project=project, cfg=self.cfg)
            validator = ValidatorAgent(ctx)
            title = rnd.modeler.spec.title if rnd.modeler.spec else project.input.title
            validator.report(rnd, Subject(Path("."), rnd.modeler.spec, {}, "", "", "", False, title))
        project.trace.append(TraceEvent(
            agent="system", step="signoff", status="ok",
            message=f"{signoff.name} ({signoff.role or 'no role given'}) recorded decision: {signoff.decision}",
        ))
        self.store.save(project)
        return project
