"""The Modeler agent: designs, implements, tests and documents a model.

Round 1:  spec -> code -> run pipeline + tests -> repair code errors ->
          modelling document (DOCX + PDF)
Round n>1: respond to each open validation finding, revise spec and code,
          then the same run/repair/document steps.

What the LLM decides: the model design, the code and the narrative.
What it cannot decide: whether tests pass, what the metrics are, what goes
into the results tables. Those come from executing the code.
"""
from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from .context import CODE_CONTRACT, CREDIT_GATE_METRICS, AgentContext, passages_block
from .documents import build_modelling_document, to_pdf
from .execution import run_pipeline, run_pytest, write_files
from .frameworks import checklist_for
from .llm_io import ask, clip, to_json
from .schemas import (
    CodeBundle,
    DocumentNarrative,
    Finding,
    FindingResponse,
    ModelSpec,
    Requirement,
    Round,
)

AGENT = "modeler"

SYSTEM = """\
You are the Modeler: a senior quantitative model developer who builds risk models (credit,
market, operational, liquidity, counterparty), pricing models, trading strategies, portfolio
construction models and AI/ML models for a regulated financial institution. An independent
validator will re-run and challenge everything you deliver, so:
- choose methods you can justify from the brief, the data and the cited documents;
- state assumptions and limitations plainly; never overstate performance;
- design tests that could genuinely fail, with acceptance thresholds set before seeing results;
- cite library passages by their [chunk id] when a design choice relies on them.
Mathematics is written in LaTeX that matplotlib mathtext can render: use \\mathrm{} for words,
no \\text{}, no align environments, one equation per item."""


class RemediationPlan(BaseModel):
    responses: list[FindingResponse] = Field(description="Exactly one response per open finding")
    spec: ModelSpec = Field(description="The full revised specification")


class ModelerAgent:
    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx

    # ------------------------------------------------------------------
    def requirements(self, category: str) -> list[Requirement]:
        return checklist_for(self.ctx.project.input.frameworks, category)

    def _brief_block(self) -> str:
        p = self.ctx.project
        profile = to_json(p.data_profile, 14000)
        return (
            f"PROJECT TITLE: {p.input.title}\n\nBRIEF FROM THE USER:\n{p.input.brief}\n\n"
            f"DATA PROFILE (files in data_dir; only this summary is shown to you):\n{profile}"
        )

    # ------------------------------------------------------------------
    def design(self) -> ModelSpec:
        ctx = self.ctx
        ctx.emit(AGENT, "design", "started", "Designing the model specification and test plan")
        passages = ctx.evidence(f"{ctx.project.input.title}. {ctx.project.input.brief}", k=10)
        generic = checklist_for(ctx.project.input.frameworks, "other")
        user = f"""{self._brief_block()}

REQUIREMENT CHECKLIST THE VALIDATOR WILL APPLY (ids you can reference in requirement_ids;
category-specific items are added once the category is known):
{to_json([r.model_dump(exclude_none=True) for r in generic], 8000)}

LIBRARY PASSAGES (attached concept papers and regulations come first):
{passages_block(passages)}

TASK: Write the model specification.
- Pick the category and a method appropriate to the data and the brief. Explain why.
- equations: the full estimation and prediction maths, with every symbol explained.
- acceptance_criteria: metric keys in snake_case exactly as the code will write them; mark one
  primary metric; give the rationale and cite a passage where a threshold comes from a document.
- test_plan: 8 to 15 tests with ids MT-01, MT-02, ... covering data quality, out-of-sample
  performance against each acceptance criterion, calibration or discrimination where relevant,
  stability over time or subsamples, sensitivity to key parameters, a stress scenario, a
  benchmark against a simple baseline, reproducibility with a fixed seed, and leakage checks.
- monitoring_plan: metrics, thresholds and frequency.
{CREDIT_GATE_METRICS}"""
        spec = ask(ctx.llm, role="modeler", system=SYSTEM, user=user, model=ModelSpec)
        ctx.emit(AGENT, "design", "ok", f"{spec.title}: {spec.category}, {len(spec.test_plan)} planned tests")
        return spec

    def remediation_plan(self, previous: Round, open_findings: list[Finding]) -> RemediationPlan:
        ctx = self.ctx
        ctx.emit(AGENT, "remediation", "started", f"Responding to {len(open_findings)} open findings")
        pkg = ctx.store.package_dir(ctx.project.project_id, previous.number)
        user = f"""{self._brief_block()}

YOUR PREVIOUS SPECIFICATION:
{to_json(previous.modeler.spec, 20000)}

YOUR PREVIOUS CODE:
{clip(_read_files(pkg, previous.modeler.files), 40000)}

PREVIOUS RESULTS: metrics={to_json(previous.modeler.pipeline.metrics if previous.modeler.pipeline else {}, 4000)}

OPEN VALIDATION FINDINGS:
{to_json([f.model_dump(exclude={"modeler_response"}) for f in open_findings], 20000)}

TASK: Respond to every finding (fixed, disputed with evidence, or accepted_limitation with a
compensating control) and return the full revised specification. Fix substantive issues in
the design rather than arguing. Do not loosen acceptance criteria unless the finding shows the
threshold itself was wrong; keep test ids stable and add new tests for new checks."""
        plan = ask(ctx.llm, role="modeler", system=SYSTEM, user=user, model=RemediationPlan)
        ctx.emit(AGENT, "remediation", "ok", ", ".join(f"{r.finding_id}: {r.action}" for r in plan.responses)[:500])
        return plan

    # ------------------------------------------------------------------
    def implement(self, spec: ModelSpec, package_dir: Path, *, previous_files: str = "", change_notes: str = "") -> list[str]:
        ctx = self.ctx
        ctx.emit(AGENT, "implement", "started", "Writing model code, pipeline and test suite")
        user = f"""{self._brief_block()}

SPECIFICATION TO IMPLEMENT:
{to_json(spec, 30000)}

{CODE_CONTRACT}
{CREDIT_GATE_METRICS if spec.category in ("credit_risk", "market_risk") else ""}
{"PREVIOUS VERSION OF THE CODE:" + chr(10) + previous_files if previous_files else ""}
{"CHANGES REQUIRED IN THIS VERSION:" + chr(10) + change_notes if change_notes else ""}

TASK: Return the complete package as files (full contents, no placeholders). Implement every
test in the test plan."""
        bundle = ask(ctx.llm, role="modeler", system=SYSTEM, user=user, model=CodeBundle)
        written = write_files(package_dir, [(f.path, f.content) for f in bundle.files])
        _ensure_init(package_dir, written)
        ctx.emit(AGENT, "implement", "ok", f"{len(written)} files written")
        files = set(written)
        if (package_dir / "model" / "__init__.py").exists():
            files.add("model/__init__.py")
        return sorted(files)

    def execute(self, rnd: Round, package_dir: Path) -> None:
        ctx = self.ctx
        s = ctx.settings
        ctx.emit(AGENT, "execute", "started", "Running pipeline and test suite")
        results = package_dir / "results"
        rnd.modeler.pipeline = run_pipeline(
            package_dir, data_dir=ctx.data_dir, out_dir=results,
            timeout_s=s.pipeline_timeout_s, mem_mb=s.sandbox_mem_mb,
        )
        rnd.modeler.tests = run_pytest(
            Path("tests"), cwd=package_dir, pythonpath=[package_dir],
            junit_path=results / "junit.xml", timeout_s=s.test_timeout_s, mem_mb=s.sandbox_mem_mb,
            extra_env={"STUDIO_DATA_DIR": str(ctx.data_dir), "STUDIO_OUT_DIR": str(results)},
        )
        c = rnd.modeler.tests.counts()
        status = "ok" if rnd.modeler.pipeline.execution.ok and not rnd.modeler.tests.collection_error else "warning"
        ctx.emit(AGENT, "execute", status, (
            f"pipeline {'ok' if rnd.modeler.pipeline.execution.ok else 'failed'}, "
            f"{len(rnd.modeler.pipeline.metrics)} metrics; tests {c['passed']} passed, "
            f"{c['failed']} failed, {c['error']} errors"
        ))

    def _defects(self, rnd: Round) -> str:
        """Code defects that justify a repair: crashes, collection errors, test errors,
        missing planned tests. Assertion failures are results, not defects."""
        m = rnd.modeler
        problems: list[str] = []
        if m.pipeline and (not m.pipeline.execution.ok or m.pipeline.error):
            problems.append("PIPELINE FAILED:\n" + clip(m.pipeline.error or m.pipeline.execution.stderr_tail, 5000))
        if m.pipeline and m.spec:
            missing = [ac.metric for ac in m.spec.acceptance_criteria if ac.metric not in m.pipeline.metrics]
            if m.pipeline.execution.ok and missing:
                problems.append(f"metrics.json is missing acceptance metrics: {missing}")
        if m.tests:
            if m.tests.collection_error:
                problems.append("PYTEST COULD NOT COLLECT OR RUN:\n" + clip(m.tests.collection_error, 5000))
            errors = [r for r in m.tests.results if r.outcome == "error"]
            for r in errors[:8]:
                problems.append(f"TEST ERROR {r.nodeid}:\n{clip(r.message, 1500)}")
            if m.spec:
                present = {r.test_id for r in m.tests.results if r.test_id}
                absent = [t.test_id for t in m.spec.test_plan if t.test_id not in present]
                if absent and not m.tests.collection_error:
                    problems.append(f"Planned tests not implemented (no test function named test_<ID>...): {absent}")
        return "\n\n".join(problems)

    def repair(self, rnd: Round, package_dir: Path) -> None:
        ctx = self.ctx
        for attempt in range(1, ctx.settings.max_repair_attempts + 1):
            defects = self._defects(rnd)
            if not defects:
                return
            ctx.emit(AGENT, "repair", "started", f"Fixing code defects (attempt {attempt})")
            user = f"""SPECIFICATION:
{to_json(rnd.modeler.spec, 20000)}

{CODE_CONTRACT}

CURRENT PACKAGE:
{clip(_read_files(package_dir, rnd.modeler.files), 50000)}

DEFECTS FOUND WHEN RUNNING IT:
{defects}

TASK: Return only the files you change, each with its full new contents. Fix code defects.
Do not weaken tests, thresholds or acceptance criteria and do not special-case the data to
make a test pass. If a test fails because the model does not meet a criterion, leave it."""
            bundle = ask(ctx.llm, role="modeler", system=SYSTEM, user=user, model=CodeBundle)
            written = write_files(package_dir, [(f.path, f.content) for f in bundle.files])
            rnd.modeler.files = sorted(set(rnd.modeler.files) | set(written))
            rnd.modeler.repair_attempts = attempt
            self.execute(rnd, package_dir)
        if self._defects(rnd):
            ctx.emit(AGENT, "repair", "warning", "Defects remain after the repair budget; delivering as is for validation")

    def document(self, rnd: Round, package_dir: Path) -> None:
        ctx = self.ctx
        spec = rnd.modeler.spec
        assert spec is not None
        ctx.emit(AGENT, "document", "started", "Writing the modelling document")
        results = {
            "metrics": rnd.modeler.pipeline.metrics if rnd.modeler.pipeline else {},
            "pipeline_error": rnd.modeler.pipeline.error if rnd.modeler.pipeline else "",
            "tests": [r.model_dump() for r in (rnd.modeler.tests.results if rnd.modeler.tests else [])],
            "acceptance": [
                {"criterion": ac.describe(), "value": (rnd.modeler.pipeline.metrics.get(ac.metric) if rnd.modeler.pipeline else None)}
                for ac in spec.acceptance_criteria
            ],
        }
        user = f"""SPECIFICATION:
{to_json(spec, 20000)}

EXECUTED RESULTS (authoritative; do not contradict them):
{to_json(results, 20000)}

{"RESPONSES TO VALIDATION FINDINGS IN THIS VERSION:" + chr(10) + to_json([r.model_dump() for r in rnd.modeler.responses], 8000) if rnd.modeler.responses else ""}

TASK: Write the narrative parts of the modelling document. The results tables, test tables,
equations and code listing are generated automatically, so do not repeat them.
executive_summary: 2 to 4 paragraphs: what the model does, the method, headline results with
numbers, test outcome, main limitations, and whether it is ready for validation.
sections: exactly these headings, each 2 to 5 paragraphs or bullet lists:
"Methodology rationale", "Interpretation of results", "Model risk and limitations assessment",
"Guidance for use". Report failed tests and unmet criteria plainly."""
        rnd.modeler.narrative = ask(ctx.llm, role="modeler", system=SYSTEM, user=user, model=DocumentNarrative)
        requirements = self.requirements(spec.category)
        out_dir = ctx.store.modeler_dir(ctx.project.project_id, rnd.number)
        docx = build_modelling_document(
            ctx.project, rnd, package_dir, out_dir / f"modelling_document_v{rnd.number}.docx", requirements,
        )
        rnd.modeler.document_docx = ctx.store.relpath(ctx.project.project_id, docx)
        pdf = to_pdf(docx, ctx.settings.pdf_via)
        rnd.modeler.document_pdf = ctx.store.relpath(ctx.project.project_id, pdf) if pdf else None
        ctx.emit(AGENT, "document", "ok", "Modelling document written" + ("" if pdf else " (PDF conversion unavailable)"))

    # ------------------------------------------------------------------
    def run_round(self, rnd: Round, previous: Round | None, open_findings: list[Finding]) -> None:
        ctx = self.ctx
        package_dir = ctx.store.package_dir(ctx.project.project_id, rnd.number)
        package_dir.mkdir(parents=True, exist_ok=True)

        if previous is None:
            rnd.modeler.spec = self.design()
            ctx.store.save(ctx.project)
            rnd.modeler.files = self.implement(rnd.modeler.spec, package_dir)
        else:
            plan = self.remediation_plan(previous, open_findings)
            rnd.modeler.spec = plan.spec
            rnd.modeler.responses = plan.responses
            ctx.store.save(ctx.project)
            prev_pkg = ctx.store.package_dir(ctx.project.project_id, previous.number)
            notes = "\n".join(f"- {r.finding_id} ({r.action}): {r.explanation} Changes: {'; '.join(r.changes)}" for r in plan.responses)
            rnd.modeler.files = self.implement(
                plan.spec, package_dir,
                previous_files=clip(_read_files(prev_pkg, previous.modeler.files), 40000),
                change_notes=notes,
            )
        ctx.store.save(ctx.project)
        self.execute(rnd, package_dir)
        ctx.store.save(ctx.project)
        self.repair(rnd, package_dir)
        ctx.store.save(ctx.project)
        self.document(rnd, package_dir)
        ctx.store.save(ctx.project)


def _read_files(root: Path, files: list[str]) -> str:
    parts = []
    for rel in files:
        path = root / rel
        if path.suffix in (".py", ".txt", ".md", ".cfg", ".ini", ".toml", ".json") and path.exists():
            parts.append(f"===== {rel} =====\n{path.read_text(encoding='utf-8', errors='replace')}")
    return "\n\n".join(parts)


def _ensure_init(package_dir: Path, written: list[str]) -> None:
    init = package_dir / "model" / "__init__.py"
    if (package_dir / "model").exists() and not init.exists():
        init.write_text("", encoding="utf-8")
