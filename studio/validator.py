"""The Validator agent: independent validation of a model package.

Works on the Modeler's package, or on an external package (document + code +
data) for which it first writes a thin adapter to the standard API.

Steps each round:
 1. requirement checklist (built-in frameworks + requirements extracted from
    attached regulations and concept papers, with page/section citations)
 2. replication from a clean copy; metric-by-metric comparison
 3. re-run of the developer's tests; plan coverage; acceptance criteria
 4. deterministic regulatory gates (credit / market) where inputs exist
 5. document and conceptual-soundness review against the checklist (LLM)
 6. independent tests written and run by the Validator
 7. challenger model built without seeing the developer's code
 8. findings merged with earlier rounds, rule-based outcome, report
"""
from __future__ import annotations

import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, Field

from .context import CODE_CONTRACT, AgentContext, passages_block
from .documents import build_validation_report, to_pdf
from .execution import run_pipeline, run_pytest, write_files
from .frameworks import checklist_for
from .library import document_text
from .llm_io import AgentOutputError, ask, clip, to_json
from .modeler import _read_files
from .regulatory_gates import thresholds_from_config
from .rules import (
    acceptance_findings,
    challenger_findings,
    compare_challenger,
    compare_metrics,
    documentation_findings,
    independent_findings,
    outcome_for,
    regulatory_gate_findings,
    replication_findings,
    test_suite_findings,
)
from .schemas import (
    ChallengerComparison,
    Citation,
    CodeBundle,
    CodeFile,
    Finding,
    ModelSpec,
    ReplicationCheck,
    Requirement,
    ReviewResult,
    Round,
    TestCase,
)

AGENT = "validator"

SYSTEM = """\
You are the Validator: an independent model validator in a regulated financial institution
(second line of defence). You validate risk, pricing, trading, portfolio and AI/ML models
against regulation, attached concept papers and sound practice. You are sceptical and
specific: every finding names the evidence (document section, code file and function, test
id, metric value) and, where a requirement comes from a document, cites the library passage
by its [chunk id]. You do not invent regulatory text; if something is not in the passages or
the checklist, say it is a sound-practice expectation. Severity scale: critical = model must
not be used; high = must be fixed before use; medium = fix within an agreed period, may be a
condition of use; low = improvement; observation = no action required."""

EXTERNAL_API = "studio_adapter.api"
EXTERNAL_ENTRY = "studio_pipeline.py"


class ExtractedRequirements(BaseModel):
    requirements: list[Requirement]


class IndependentSuite(BaseModel):
    plan: list[TestCase] = Field(description="Validator test plan with ids VT-01, VT-02, ...")
    files: list[CodeFile] = Field(description="tests/conftest.py and tests/test_*.py")


class ChallengerBuild(BaseModel):
    description: str = Field(description="Method, why it is a meaningful challenger, and how it differs")
    files: list[CodeFile]


class ExternalUnderstanding(BaseModel):
    spec: ModelSpec = Field(description="The model as documented by its developer")
    documented_metrics: dict[str, float] = Field(default_factory=dict, description="Metrics stated in the document, keyed as the adapter outputs them")
    adapter_files: list[CodeFile] = Field(description="studio_adapter/__init__.py, studio_adapter/api.py and studio_pipeline.py")


class Conclusion(BaseModel):
    conclusion: str


@dataclass
class Subject:
    package_dir: Path
    spec: ModelSpec | None
    reported_metrics: dict[str, float]
    document: str
    api_module: str
    entrypoint: str
    external: bool
    title: str


class ValidatorAgent:
    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    # ------------------------------------------------------------------
    # Subjects
    # ------------------------------------------------------------------
    def subject_from_modeler(self, rnd: Round) -> Subject:
        ctx = self.ctx
        pkg = ctx.store.package_dir(ctx.project.project_id, rnd.number)
        doc_text = ""
        if rnd.modeler.document_docx:
            try:
                doc_text = document_text(ctx.store.project_dir(ctx.project.project_id) / rnd.modeler.document_docx)
            except Exception:  # noqa: BLE001 - fall back to the spec
                doc_text = ""
        spec = rnd.modeler.spec
        return Subject(
            package_dir=pkg, spec=spec,
            reported_metrics=dict(rnd.modeler.pipeline.metrics) if rnd.modeler.pipeline else {},
            document=doc_text or to_json(spec, 30000), api_module="model.api", entrypoint="pipeline.py",
            external=False, title=spec.title if spec else ctx.project.input.title,
        )

    def prepare_external(self, rnd: Round) -> Subject:
        """Unpack the uploaded package, read its documentation and write an adapter."""
        ctx = self.ctx
        pid = ctx.project.project_id
        ext = ctx.store.external_dir(pid)
        pkg = ctx.store.validator_dir(pid, rnd.number) / "subject_package"
        if pkg.exists():
            shutil.rmtree(pkg)
        pkg.mkdir(parents=True)
        docs: list[Path] = []
        for path in sorted(ext.iterdir()):
            if path.suffix.lower() == ".zip":
                _safe_extract(path, pkg)
            elif path.suffix.lower() in (".pdf", ".docx", ".md", ".txt"):
                docs.append(path)
            elif path.is_file():
                shutil.copy2(path, pkg / path.name)
        # A zip with a single top-level folder: work inside it.
        entries = [p for p in pkg.iterdir() if not p.name.startswith(".")]
        if len(entries) == 1 and entries[0].is_dir():
            inner = entries[0]
            for child in inner.iterdir():
                shutil.move(str(child), pkg / child.name)
            inner.rmdir()
        for doc in list(pkg.rglob("*")):
            if doc.suffix.lower() in (".pdf", ".docx") and doc.is_file():
                docs.append(doc)
        doc_text = "\n\n".join(f"===== {d.name} =====\n{document_text(d)}" for d in docs)
        code_files = sorted(p.relative_to(pkg).as_posix() for p in pkg.rglob("*.py"))

        ctx.emit(AGENT, "external-intake", "started", f"Reading {len(docs)} documents and {len(code_files)} code files")
        user = f"""BRIEF: {ctx.project.input.brief}

DATA PROFILE (files in data_dir):
{to_json(ctx.project.data_profile, 10000)}

MODEL DOCUMENTATION PROVIDED BY THE DEVELOPER:
{clip(doc_text, 45000)}

DEVELOPER CODE:
{clip(_read_files(pkg, code_files), 45000)}

{CODE_CONTRACT.replace("model/api.py", "studio_adapter/api.py").replace("model/__init__.py", "studio_adapter/__init__.py").replace("pipeline.py", "studio_pipeline.py").replace("model.api", "studio_adapter.api")}

TASK:
1. spec: describe the model exactly as the developer documented it (do not improve it).
   test_plan: the developer's documented tests, ids MT-01... in document order. If the
   document states no acceptance criteria, leave acceptance_criteria empty.
2. documented_metrics: every performance figure the document reports, keyed by the metric
   name your adapter will output.
3. adapter_files: a thin adapter that exposes the standard API by calling the developer's
   code (import it, do not rewrite the model), plus studio_pipeline.py. Only write the
   adapter; the developer's files stay untouched. Output the metrics the document reports."""
        understanding = ask(ctx.llm, role="validator", system=SYSTEM, user=user, model=ExternalUnderstanding)
        write_files(pkg, [(f.path, f.content) for f in understanding.adapter_files])
        init = pkg / "studio_adapter" / "__init__.py"
        if not init.exists():
            init.parent.mkdir(parents=True, exist_ok=True)
            init.write_text("", encoding="utf-8")
        rnd.modeler.spec = understanding.spec
        rnd.modeler.files = sorted(p.relative_to(pkg).as_posix() for p in pkg.rglob("*") if p.is_file())
        ctx.emit(AGENT, "external-intake", "ok", f"Adapter written for {understanding.spec.title}")
        return Subject(
            package_dir=pkg, spec=understanding.spec, reported_metrics=understanding.documented_metrics,
            document=doc_text, api_module=EXTERNAL_API, entrypoint=EXTERNAL_ENTRY, external=True,
            title=understanding.spec.title,
        )

    # ------------------------------------------------------------------
    # 1. Requirements
    # ------------------------------------------------------------------
    def requirements(self, subject: Subject, previous: Round | None) -> list[Requirement]:
        ctx = self.ctx
        if previous is not None and previous.validator.requirements:
            return previous.validator.requirements
        category = subject.spec.category if subject.spec else "other"
        reqs = checklist_for(ctx.project.input.frameworks, category)
        if ctx.library is None or not ctx.project.attachment_doc_ids:
            return reqs
        ctx.emit(AGENT, "requirements", "started", "Extracting requirements from attached documents")
        queries = [
            f"requirements for {subject.title} {category} model",
            "must shall requirement validation model development documentation",
            "data requirements observation period sample representativeness",
            "performance testing calibration discrimination backtesting thresholds",
            "governance monitoring use test human oversight",
        ]
        passages, seen = [], set()
        for q in queries:
            for p in ctx.library.retrieve(q, k=8, doc_ids=ctx.project.attachment_doc_ids):
                if p.chunk_id not in seen:
                    seen.add(p.chunk_id)
                    passages.append(p)
        if not passages:
            return reqs
        user = f"""MODEL UNDER VALIDATION: {subject.title} ({category})
BRIEF: {ctx.project.input.brief}

PASSAGES FROM THE ATTACHED REGULATIONS AND CONCEPT PAPERS:
{passages_block(passages, 40000)}

TASK: List the concrete, checkable requirements these passages impose on this model's
development, testing, documentation or use. ids ATT-01, ATT-02, ...; framework = the document
title; citation.chunk_id = the [chunk id] of the passage it comes from (copy it exactly);
citation.source and locator may be left empty, they are filled in from the library. Skip
anything not applicable. At most 25 requirements."""
        try:
            extracted = ask(ctx.llm, role="validator", system=SYSTEM, user=user, model=ExtractedRequirements)
        except AgentOutputError as exc:
            ctx.emit(AGENT, "requirements", "warning", str(exc)[:300])
            return reqs
        kept = 0
        for r in extracted.requirements:
            r.citation = self._verified_citation(r.citation)
            if r.citation is None:
                continue  # a requirement we cannot trace to a passage is dropped
            reqs.append(r)
            kept += 1
        ctx.emit(AGENT, "requirements", "ok", f"{len(reqs)} requirements ({kept} from attachments)")
        return reqs

    def _verified_citation(self, c: Citation | None) -> Citation | None:
        if c is None or not c.chunk_id or self.ctx.library is None:
            return None
        return self.ctx.library.chunk_citation(c.chunk_id)

    # ------------------------------------------------------------------
    # 2-4. Replication, developer tests, gates
    # ------------------------------------------------------------------
    def replicate(self, rnd: Round, subject: Subject) -> tuple[Path, list[Finding]]:
        ctx = self.ctx
        s = ctx.settings
        vdir = ctx.store.validator_dir(ctx.project.project_id, rnd.number)
        rep_pkg = vdir / "replication" / "package"
        ctx.store.copy_tree(subject.package_dir, rep_pkg)
        ctx.emit(AGENT, "replication", "started", "Re-running the pipeline from a clean copy")
        run = run_pipeline(rep_pkg, data_dir=ctx.data_dir, out_dir=vdir / "replication" / "results",
                           timeout_s=s.pipeline_timeout_s, mem_mb=s.sandbox_mem_mb, entrypoint=subject.entrypoint)
        if subject.external:
            rel, abs_ = 0.01, 1e-3  # documents round their figures
        else:
            rel, abs_ = s.replication_rel_tol, s.replication_abs_tol
        compared, mismatches = compare_metrics(subject.reported_metrics, run.metrics, rel, abs_)
        pipeline_ok = run.execution.ok and not run.error
        rnd.validator.replication = ReplicationCheck(
            reproduced=pipeline_ok and not mismatches, pipeline_ok=pipeline_ok, compared=compared,
            mismatches=mismatches if pipeline_ok else [],
            note=(run.error or run.execution.stderr_tail[-1500:]) if not pipeline_ok
            else f"{len(compared)} reported metrics compared; tolerance rel {rel:g}, abs {abs_:g}.",
        )
        self._replicated_metrics = run.metrics
        findings = replication_findings(rnd.validator.replication)
        ctx.emit(AGENT, "replication", "ok" if rnd.validator.replication.reproduced else "warning",
                 "Reproduced" if rnd.validator.replication.reproduced else
                 ("Pipeline failed" if not pipeline_ok else f"Mismatches: {', '.join(mismatches)}"))

        tests_dir = rep_pkg / "tests"
        if tests_dir.exists():
            ctx.emit(AGENT, "developer-tests", "started", "Re-running the developer's test suite")
            rnd.validator.modeler_tests_rerun = run_pytest(
                Path("tests"), cwd=rep_pkg, pythonpath=[rep_pkg], junit_path=vdir / "replication" / "junit.xml",
                timeout_s=s.test_timeout_s, mem_mb=s.sandbox_mem_mb,
                extra_env={"STUDIO_DATA_DIR": str(ctx.data_dir), "STUDIO_OUT_DIR": str(vdir / "replication" / "results")},
            )
            c = rnd.validator.modeler_tests_rerun.counts()
            ctx.emit(AGENT, "developer-tests", "ok", f"{c['passed']} passed, {c['failed']} failed, {c['error']} errors")
        findings += test_suite_findings(subject.spec, rnd.validator.modeler_tests_rerun, external=subject.external)
        findings += acceptance_findings(subject.spec, run.metrics, pipeline_ok)
        findings += documentation_findings(subject.spec)
        gate, note = regulatory_gate_findings(subject.spec, run.metrics, thresholds_from_config(ctx.cfg))
        if note:
            ctx.emit(AGENT, "regulatory-gates", "ok" if gate or "run" in note else "warning", note)
        findings += gate
        return rep_pkg, findings

    # ------------------------------------------------------------------
    # 5. Review
    # ------------------------------------------------------------------
    def review(self, rnd: Round, subject: Subject, rep_pkg: Path, rule_findings: list[Finding],
               previous_review: list[Finding]) -> ReviewResult:
        ctx = self.ctx
        ctx.emit(AGENT, "review", "started", "Reviewing documentation, methodology and code against the checklist")
        passages = ctx.evidence(f"{subject.title} {subject.spec.methodology_summary if subject.spec else ''}", k=10)
        responses = {r.finding_id: r.model_dump() for r in rnd.modeler.responses}
        user = f"""MODEL: {subject.title}
BRIEF: {ctx.project.input.brief}

REQUIREMENT CHECKLIST (assess every item):
{to_json([r.model_dump(exclude_none=True) for r in rnd.validator.requirements], 20000)}

DEVELOPER DOCUMENTATION:
{clip(subject.document, 40000)}

CODE:
{clip(_read_files(rep_pkg, [p.relative_to(rep_pkg).as_posix() for p in sorted(rep_pkg.rglob('*.py'))]), 40000)}

EXECUTED EVIDENCE (authoritative):
replication: {to_json(rnd.validator.replication, 6000)}
replicated metrics: {to_json(getattr(self, '_replicated_metrics', {}), 6000)}
developer tests: {to_json(rnd.validator.modeler_tests_rerun.counts() if rnd.validator.modeler_tests_rerun else 'none', 500)}
data profile: {to_json(ctx.project.data_profile, 8000)}

FINDINGS ALREADY RAISED BY DETERMINISTIC RULES (do not duplicate them):
{to_json([{"id": f.finding_id, "title": f.title} for f in rule_findings], 6000)}

YOUR OPEN FINDINGS FROM THE PREVIOUS ROUND, WITH THE DEVELOPER'S RESPONSES:
{to_json([{**f.model_dump(include={"finding_id", "title", "severity", "description", "recommendation"}), "response": responses.get(f.finding_id)} for f in previous_review], 15000) if previous_review else "none"}

LIBRARY PASSAGES:
{passages_block(passages, 20000)}

TASK:
- requirement_assessments: one per checklist item; status met, partially_met, not_met or
  not_applicable, with evidence pointing to the document section, code or result.
- findings: new issues only, on conceptual soundness, data, implementation, documentation,
  governance and regulatory compliance. finding_id: leave as "new". source: "review".
  Cite passages with citations[].chunk_id when a requirement comes from a document.
  Every not_met or partially_met requirement needs a finding (set its id in the assessment
  only if you raise one).
- previous_finding_updates: for each previous finding, {{"finding_id", "status": "closed" or
  "open", "note"}}. Close only when the evidence shows the issue is resolved; a dispute is
  closed only if the developer's argument is correct."""
        result = ask(ctx.llm, role="validator", system=SYSTEM, user=user, model=ReviewResult)
        for f in result.findings:
            f.source = "review"
            f.citations = [c for c in (self._verified_citation(c) for c in f.citations) if c is not None]
        ctx.emit(AGENT, "review", "ok", f"{len(result.findings)} new review findings; "
                 f"{sum(1 for a in result.requirement_assessments if a.status in ('not_met', 'partially_met'))} requirements not fully met")
        return result

    # ------------------------------------------------------------------
    # 6. Independent tests
    # ------------------------------------------------------------------
    def independent_tests(self, rnd: Round, subject: Subject, rep_pkg: Path, previous: Round | None) -> list[Finding]:
        ctx = self.ctx
        s = ctx.settings
        pid = ctx.project.project_id
        idir = ctx.store.validator_dir(pid, rnd.number) / "independent"
        prev_dir = ctx.store.validator_dir(pid, previous.number) / "independent" if previous else None
        if previous is not None and prev_dir is not None and (prev_dir / "tests").exists() and previous.validator.independent_plan:
            ctx.store.copy_tree(prev_dir, idir, exclude=("results", "__pycache__", ".pytest_cache", "pytest.ini"))
            rnd.validator.independent_plan = previous.validator.independent_plan
            ctx.emit(AGENT, "independent-tests", "started", "Re-running the independent test suite on the revised model")
        else:
            ctx.emit(AGENT, "independent-tests", "started", "Designing independent tests")
            user = f"""MODEL: {subject.title}
SPECIFICATION:
{to_json(subject.spec, 20000)}

The model is importable as `{subject.api_module}` with load_data(data_dir), fit(data, seed),
predict(model, df), evaluate(model, data) and run(data_dir, out_dir, seed). Code:
{clip(_read_files(rep_pkg, [p.relative_to(rep_pkg).as_posix() for p in sorted(rep_pkg.rglob('*.py')) if 'tests' not in p.parts]), 30000)}

DATA PROFILE: {to_json(ctx.project.data_profile, 8000)}
DEVELOPER TEST PLAN (do not simply repeat it): {to_json([t.model_dump(include={"test_id", "name"}) for t in (subject.spec.test_plan if subject.spec else [])], 4000)}
REQUIREMENTS: {to_json([r.model_dump(include={"req_id", "text"}) for r in rnd.validator.requirements], 8000)}

TASK: Design and implement 6 to 12 independent validation tests (ids VT-01...) that challenge
the model rather than confirm it: outcome analysis on held-out or out-of-time data computed
by you, calibration or backtesting with formal statistics, stability across subsamples and
time, sensitivity and stress of key inputs and parameters, a naive benchmark, data leakage,
edge cases and invalid inputs, reproducibility. Set severity_if_fail for each.
Files: tests/conftest.py (session fixtures data_dir from os.environ["STUDIO_DATA_DIR"], data,
model, using `{subject.api_module}`), and tests/test_*.py with function names starting
test_VT01_..., test_VT02_... Assertions must print observed values and thresholds.
Same library and runtime rules as the developer: numpy, pandas, scipy, scikit-learn,
statsmodels, matplotlib, standard library, tools.statistical_tests; no network; under 10 minutes."""
            suite = ask(ctx.llm, role="validator", system=SYSTEM, user=user, model=IndependentSuite)
            if idir.exists():
                shutil.rmtree(idir)
            write_files(idir, [(f.path, f.content) for f in suite.files])
            rnd.validator.independent_plan = suite.plan

        def _run():
            return run_pytest(
                Path("tests"), cwd=idir, pythonpath=[rep_pkg], junit_path=idir / "results" / "junit.xml",
                timeout_s=s.test_timeout_s, mem_mb=s.sandbox_mem_mb,
                extra_env={"STUDIO_DATA_DIR": str(ctx.data_dir), "STUDIO_OUT_DIR": str(idir / "results")},
            )

        run = _run()
        for attempt in range(1, s.max_repair_attempts + 1):
            errors = [r for r in run.results if r.outcome == "error"]
            if not run.collection_error and not errors:
                break
            ctx.emit(AGENT, "independent-tests", "started", f"Fixing errors in the validator's own tests (attempt {attempt})")
            problems = run.collection_error or "\n\n".join(f"{r.nodeid}:\n{clip(r.message, 1500)}" for r in errors[:8])
            user = f"""Your independent tests failed to execute (errors, not assertion failures).
TEST FILES:
{clip(_read_files(idir, [p.relative_to(idir).as_posix() for p in sorted(idir.rglob('*.py'))]), 30000)}
MODEL API ({subject.api_module}):
{clip(_read_files(rep_pkg, [p.relative_to(rep_pkg).as_posix() for p in sorted(rep_pkg.rglob('*.py')) if 'tests' not in p.parts]), 20000)}
ERRORS:
{problems}
Return only the test files you change, with full contents. Do not change what a test checks
or its threshold; fix how it runs."""
            fix = ask(ctx.llm, role="validator", system=SYSTEM, user=user, model=CodeBundle)
            write_files(idir, [(f.path, f.content) for f in fix.files if f.path.replace("\\", "/").startswith("tests/")])
            run = _run()
        rnd.validator.independent_tests = run
        c = run.counts()
        ctx.emit(AGENT, "independent-tests", "ok", f"{c['passed']} passed, {c['failed']} failed, {c['error']} errors")
        return independent_findings(rnd.validator.independent_plan, run)

    # ------------------------------------------------------------------
    # 7. Challenger
    # ------------------------------------------------------------------
    def wants_challenger(self) -> bool:
        mode = self.ctx.project.input.challenger
        return mode == "on" or (mode == "auto" and bool(self.ctx.project.attachment_doc_ids))

    def challenger(self, rnd: Round, subject: Subject, previous: Round | None) -> list[Finding]:
        ctx = self.ctx
        s = ctx.settings
        pid = ctx.project.project_id
        champion = getattr(self, "_replicated_metrics", {})
        prev_ch = previous.validator.challenger if previous else None
        if prev_ch is not None and prev_ch.pipeline_ok:
            ch = prev_ch.model_copy(deep=True)
            ch.champion_metrics = champion
        else:
            cdir = ctx.store.validator_dir(pid, rnd.number) / "challenger"
            ctx.emit(AGENT, "challenger", "started", "Building an independent challenger model")
            passages = ctx.evidence(f"{ctx.project.input.brief} methodology estimation approach", k=12)
            spec = subject.spec
            targets = to_json([ac.model_dump(include={"metric", "operator", "threshold", "primary", "higher_is_better"})
                               for ac in (spec.acceptance_criteria if spec else [])], 4000)
            user = f"""BRIEF: {ctx.project.input.brief}
TARGET: {spec.target if spec else 'see brief'}
DATA PROFILE: {to_json(ctx.project.data_profile, 10000)}
METRICS TO REPORT (same names, same held-out evaluation logic so results are comparable): {targets}
CONCEPT PAPERS AND REGULATIONS:
{passages_block(passages, 25000)}

{CODE_CONTRACT}

TASK: Build an independent challenger model from the brief and documents. You have NOT seen
the developer's code and must not try to reproduce it; choose a credible alternative
(different model family, specification or estimation approach) that the documents allow.
Evaluate on a held-out sample defined the same way the brief or documents require.
Return description and files: model/__init__.py, model/api.py, pipeline.py. No tests needed."""
            try:
                build = ask(ctx.llm, role="validator", system=SYSTEM, user=user, model=ChallengerBuild)
            except AgentOutputError as exc:
                rnd.validator.challenger = ChallengerComparison(error=str(exc)[:500])
                ctx.emit(AGENT, "challenger", "warning", "Challenger could not be generated")
                return []
            if cdir.exists():
                shutil.rmtree(cdir)
            write_files(cdir, [(f.path, f.content) for f in build.files])
            (cdir / "model").mkdir(exist_ok=True)
            (cdir / "model" / "__init__.py").touch()
            run = run_pipeline(cdir, data_dir=ctx.data_dir, out_dir=cdir / "results",
                               timeout_s=s.pipeline_timeout_s, mem_mb=s.sandbox_mem_mb)
            for attempt in range(1, s.max_repair_attempts + 1):
                if run.execution.ok and not run.error:
                    break
                ctx.emit(AGENT, "challenger", "started", f"Fixing the challenger (attempt {attempt})")
                fix = ask(ctx.llm, role="validator", system=SYSTEM, model=CodeBundle, user=f"""The challenger failed:
{clip(run.error or run.execution.stderr_tail, 5000)}
FILES:
{clip(_read_files(cdir, [p.relative_to(cdir).as_posix() for p in sorted(cdir.rglob('*.py'))]), 30000)}
{CODE_CONTRACT}
Return only changed files with full contents.""")
                write_files(cdir, [(f.path, f.content) for f in fix.files])
                run = run_pipeline(cdir, data_dir=ctx.data_dir, out_dir=cdir / "results",
                                   timeout_s=s.pipeline_timeout_s, mem_mb=s.sandbox_mem_mb)
            ch = ChallengerComparison(
                description=build.description, champion_metrics=champion, challenger_metrics=run.metrics,
                pipeline_ok=run.execution.ok and not run.error, error=run.error,
            )
        ch.primary_metric, ch.challenger_better = compare_challenger(
            ch.champion_metrics, ch.challenger_metrics, subject.spec, s.challenger_material_gap)
        rnd.validator.challenger = ch
        ctx.emit(AGENT, "challenger", "ok" if ch.pipeline_ok else "warning",
                 f"Primary metric {ch.primary_metric}: champion {_fmt(ch.champion_metrics.get(ch.primary_metric or ''))}, "
                 f"challenger {_fmt(ch.challenger_metrics.get(ch.primary_metric or ''))}" if ch.pipeline_ok else f"Challenger failed: {ch.error[:200]}")
        return challenger_findings(ch)

    # ------------------------------------------------------------------
    # 8. Merge, outcome, report
    # ------------------------------------------------------------------
    def merge(self, rnd: Round, previous: Round | None, rule_like: list[Finding], review: ReviewResult) -> list[Finding]:
        n = rnd.number
        prev_findings = previous.validator.findings if previous else []
        responses = {r.finding_id: r for r in rnd.modeler.responses}
        merged: dict[str, Finding] = {}

        # Carry over findings closed earlier.
        for f in prev_findings:
            if f.status != "open":
                merged[f.finding_id] = f.model_copy(deep=True)

        current = {f.finding_id: f for f in rule_like}
        for f in prev_findings:
            if f.status != "open" or f.source == "review":
                continue
            if f.finding_id in current:
                cur = current.pop(f.finding_id)
                cur.raised_round = f.raised_round
                merged[cur.finding_id] = cur
            else:
                closed = f.model_copy(deep=True)
                closed.status, closed.closed_round = "closed", n
                merged[closed.finding_id] = closed
        for fid, f in current.items():
            f.raised_round = n
            merged[fid] = f

        updates = {u.get("finding_id"): u for u in review.previous_finding_updates}
        for f in prev_findings:
            if f.status != "open" or f.source != "review":
                continue
            upd = updates.get(f.finding_id, {})
            g = f.model_copy(deep=True)
            if upd.get("status") == "closed":
                g.status, g.closed_round = "closed", n
            if upd.get("note"):
                g.evidence = g.evidence + [f"Round {n} review: {upd['note']}"]
            merged[g.finding_id] = g

        next_q = 1 + max([int(fid.split("-")[-1]) for fid in merged if fid.startswith("F-Q-") and fid.split("-")[-1].isdigit()] or [0])
        for f in review.findings:
            f.finding_id = f"F-Q-{next_q:03d}"
            next_q += 1
            f.raised_round = n
            merged[f.finding_id] = f

        for fid, f in merged.items():
            if fid in responses:
                f.modeler_response = responses[fid]
        return list(merged.values())

    def conclude(self, rnd: Round, subject: Subject) -> None:
        ctx = self.ctx
        v = rnd.validator
        open_f = [f for f in v.findings if f.status == "open"]
        user = f"""MODEL: {subject.title}
RULE-BASED OUTCOME (fixed; you cannot change it): {v.outcome}
ROUND: {rnd.number} of at most {ctx.project.input.max_rounds}
REPLICATION: {to_json(v.replication, 3000)}
OPEN FINDINGS: {to_json([f.model_dump(include={"finding_id", "severity", "title"}) for f in open_f], 8000)}
CLOSED THIS ROUND: {to_json([f.finding_id for f in v.findings if f.closed_round == rnd.number], 2000)}
INDEPENDENT TESTS: {to_json(v.independent_tests.counts() if v.independent_tests else 'none', 300)}
CHALLENGER: {to_json(v.challenger, 3000) if v.challenger else 'none'}

TASK: Write the conclusion of the validation report in 2 to 4 short paragraphs: overall
view, the findings that drive the outcome, conditions of use if any, and what must happen
next. Be factual; do not soften the outcome."""
        try:
            v.conclusion = ask(ctx.llm, role="validator", system=SYSTEM, user=user, model=Conclusion).conclusion
        except AgentOutputError:
            v.conclusion = f"Outcome: {v.outcome}. {len(open_f)} findings remain open."

    def report(self, rnd: Round, subject: Subject) -> None:
        ctx = self.ctx
        pid = ctx.project.project_id
        titles = []
        if ctx.library is not None:
            titles = [d.title for d in (ctx.library.get(i) for i in ctx.project.attachment_doc_ids) if d]
        path = build_validation_report(
            ctx.project, rnd, ctx.store.validator_dir(pid, rnd.number) / f"validation_report_round{rnd.number}.docx",
            subject_title=subject.title, attached_titles=titles,
        )
        rnd.validator.report_docx = ctx.store.relpath(pid, path)
        pdf = to_pdf(path, ctx.settings.pdf_via)
        rnd.validator.report_pdf = ctx.store.relpath(pid, pdf) if pdf else None

    # ------------------------------------------------------------------
    def run_round(self, rnd: Round, previous: Round | None) -> None:
        ctx = self.ctx
        if ctx.project.input.mode == "validate_external":
            subject = self.prepare_external(rnd)
        else:
            subject = self.subject_from_modeler(rnd)
        ctx.store.save(ctx.project)

        rnd.validator.requirements = self.requirements(subject, previous)
        ctx.store.save(ctx.project)
        rep_pkg, rule_findings = self.replicate(rnd, subject)
        ctx.store.save(ctx.project)

        rule_findings += self.independent_tests(rnd, subject, rep_pkg, previous)
        ctx.store.save(ctx.project)
        if self.wants_challenger():
            rule_findings += self.challenger(rnd, subject, previous)
            ctx.store.save(ctx.project)

        prev_review = [f for f in (previous.validator.findings if previous else []) if f.status == "open" and f.source == "review"]
        review = self.review(rnd, subject, rep_pkg, rule_findings, prev_review)
        rnd.validator.review = review
        rnd.validator.findings = self.merge(rnd, previous, rule_findings, review)
        rnd.validator.outcome = outcome_for(rnd.validator.findings)
        ctx.store.save(ctx.project)

        self.conclude(rnd, subject)
        self.report(rnd, subject)
        ctx.emit(AGENT, "report", "ok", f"Round {rnd.number} outcome: {rnd.validator.outcome}")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _safe_extract(zip_path: Path, dest: Path) -> None:
    """Extract a zip, refusing absolute paths, '..' and oversized archives."""
    from .storage import resolve_inside

    with zipfile.ZipFile(zip_path) as zf:
        total = sum(i.file_size for i in zf.infolist())
        if total > 500 * 1024 * 1024:
            raise ValueError("Archive expands to more than 500 MB")
        for info in zf.infolist():
            if info.is_dir():
                continue
            target = resolve_inside(dest, info.filename)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(info) as src, open(target, "wb") as out:
                shutil.copyfileobj(src, out)
