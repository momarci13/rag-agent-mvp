"""Command-line entry point.

Examples:
  python -m studio.cli develop --title "Mortgage PD" --brief brief.txt --data loans.csv \\
      --concept-paper concept.pdf --regulation eba_gl.pdf --framework eu_banking
  python -m studio.cli validate --title "Vendor PD" --brief "Validate the vendor model" \\
      --data loans.csv --package vendor_model.zip --framework eu_banking --framework eu_ai_act
  python -m studio.cli status <project_id>
  python -m studio.cli signoff <project_id> --name "Jane Validator" --role "Head of validation"
  python -m studio.cli library-add --kind regulation eu_ai_act.pdf
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import StudioSettings, load_config, make_llm_config
from .frameworks import FRAMEWORK_LABELS
from .schemas import ProjectInput, Signoff


def _studio():
    from agents.llm import HostedLLM

    from .library import Library, make_rag
    from .orchestrator import Studio
    from .storage import ProjectStore

    cfg = load_config()
    settings = StudioSettings.from_config(cfg)
    library = Library(settings.output_dir.parent / "library", make_rag(cfg, settings.library_collection))
    library.ensure_builtin(settings.builtin_corpus_dir)
    return Studio(cfg, HostedLLM(make_llm_config(cfg)), library, ProjectStore(settings.output_dir), settings)


def _files(paths: list[str] | None) -> list[tuple[str, bytes]]:
    return [(Path(p).name, Path(p).read_bytes()) for p in paths or []]


def _brief(value: str) -> str:
    p = Path(value)
    return p.read_text(encoding="utf-8") if p.is_file() else value


def _print_event(e) -> None:
    print(f"[{e.agent:9}] {e.step:18} {e.status:8} {e.message.splitlines()[0] if e.message else ''}", flush=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="studio", description="Modeler and Validator agents")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("develop", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--title", required=True)
        p.add_argument("--brief", required=True, help="Text, or a path to a text file")
        p.add_argument("--data", action="append", required=True)
        p.add_argument("--regulation", action="append")
        p.add_argument("--concept-paper", action="append")
        p.add_argument("--framework", action="append", choices=[k for k in FRAMEWORK_LABELS if k != "sound_practice"])
        p.add_argument("--challenger", choices=["auto", "on", "off"], default="auto")
        if name == "develop":
            p.add_argument("--max-rounds", type=int, default=3)
        else:
            p.add_argument("--package", action="append", required=True, help="Code, zip and model documents")
    s = sub.add_parser("status")
    s.add_argument("project_id")
    so = sub.add_parser("signoff")
    so.add_argument("project_id")
    so.add_argument("--name", required=True)
    so.add_argument("--role", default="")
    so.add_argument("--decision", choices=["accept", "reject"], default="accept")
    so.add_argument("--comment", default="")
    la = sub.add_parser("library-add")
    la.add_argument("--kind", choices=["regulation", "concept_paper", "reference"], required=True)
    la.add_argument("files", nargs="+")
    args = ap.parse_args(argv)

    studio = _studio()
    if args.cmd in ("develop", "validate"):
        inp = ProjectInput(
            mode="develop_and_validate" if args.cmd == "develop" else "validate_external",
            title=args.title, brief=_brief(args.brief), frameworks=args.framework or [],
            max_rounds=getattr(args, "max_rounds", 1), challenger=args.challenger,
        )
        project = studio.create_project(
            inp, data=_files(args.data), regulations=_files(args.regulation),
            concept_papers=_files(args.concept_paper), package=_files(getattr(args, "package", None)),
        )
        print(f"Project {project.project_id}")
        project = studio.run(project.project_id, progress=_print_event)
        print(f"\nStatus: {project.status}  Outcome: {project.final_outcome}")
        print(f"Files: {studio.store.project_dir(project.project_id)}")
        return 0 if project.status == "awaiting_signoff" else 1
    if args.cmd == "status":
        p = studio.store.load(args.project_id)
        print(f"{p.input.title}: {p.status}, outcome {p.final_outcome}, rounds {len(p.rounds)}")
        for f in p.open_findings():
            print(f"  {f.finding_id:22} {f.severity:9} {f.title}")
        return 0
    if args.cmd == "signoff":
        p = studio.signoff(args.project_id, Signoff(name=args.name, role=args.role, decision=args.decision, comment=args.comment))
        print(f"Recorded. Report: {p.rounds[-1].validator.report_docx}")
        return 0
    for f in args.files:
        doc = studio.library.add_document(Path(f), kind=args.kind)
        print(f"{doc.doc_id}  {doc.title}  ({doc.chunks} chunks)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
