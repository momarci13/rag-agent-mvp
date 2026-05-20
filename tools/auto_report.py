"""Automated structured markdown research reports.

After each research_run() or autonomous loop iteration, generates a
machine-readable markdown report with executive summary, key findings,
citation table, open questions, and next search directions.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agents.llm import OllamaLLM
    from agents.graph import RunState
    from tools.source_search import SourceResult

logger = logging.getLogger(__name__)

TASKS_DIR = Path(__file__).parent.parent / "output" / "tasks"
REPORTS_DIR = Path(__file__).parent.parent / "output" / "research_reports"


@dataclass
class ResearchReport:
    task_id: str
    topic: str
    generated_at: str
    executive_summary: str
    key_findings: list[str] = field(default_factory=list)
    papers: list[dict] = field(default_factory=list)   # {title, year, source, url, citations}
    open_questions: list[str] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    next_search_directions: list[str] = field(default_factory=list)
    confidence_level: float = 0.0


def _slug(text: str) -> str:
    """URL-safe slug from arbitrary text."""
    text = re.sub(r"[^a-z0-9 ]", "", text.lower())
    return re.sub(r"\s+", "_", text.strip())[:50]


def _extract_narrative_fields(state: "RunState") -> dict[str, Any]:
    """Pull executive_summary and key_findings from the last accepted artifact's narrative."""
    for art in reversed(state.artifacts):
        if art.get("type") in ("literature",):
            continue
        report = art.get("report", {})
        narrative = report.get("narrative", {})
        if narrative:
            return {
                "executive_summary": narrative.get("conclusions") or narrative.get("analysis") or "",
                "key_findings": list(narrative.get("key_results") or []),
                "methodology": narrative.get("methodology") or "",
            }
    return {"executive_summary": "", "key_findings": [], "methodology": ""}


def generate_research_report(
    state: "RunState",
    papers: list,
    llm: "OllamaLLM",
) -> ResearchReport:
    """Generate a ResearchReport from the completed RunState.

    Uses the narrative already attached to the last artifact as the base,
    then asks the LLM to synthesise open_questions, contradictions, and
    next_search_directions from the available context.
    Writes two files:
      - output/tasks/{task_id}/research_report.md
      - output/research_reports/{slug}_{date}.md
    """
    nf = _extract_narrative_fields(state)
    executive_summary = nf["executive_summary"][:1200]
    key_findings = nf["key_findings"][:10]

    paper_rows: list[dict] = []
    for p in papers[:30]:
        paper_rows.append({
            "title": getattr(p, "title", str(p))[:120],
            "year": getattr(p, "year", getattr(p, "published", "")[:4] if hasattr(p, "published") else ""),
            "source": getattr(p, "source", "arxiv"),
            "url": getattr(p, "url", ""),
            "citations": getattr(p, "citation_count", 0),
        })

    # Ask LLM for open questions, contradictions, next directions
    open_questions: list[str] = []
    contradictions: list[str] = []
    next_dirs: list[str] = []
    confidence = 0.5

    if executive_summary or key_findings:
        context = f"Task: {state.task}\n\nSummary:\n{executive_summary}\n\nKey findings:\n" + "\n".join(
            f"- {f}" for f in key_findings[:5]
        )
        prompt_msgs = [
            {
                "role": "system",
                "content": (
                    "Based on the research summary, output JSON with keys: "
                    "open_questions (list ≤4, each ≤15 words), "
                    "contradictions (list ≤3, each ≤20 words), "
                    "next_search_directions (list ≤4, each ≤12 words), "
                    "confidence_level (float 0-1)."
                ),
            },
            {"role": "user", "content": context[:2000]},
        ]
        try:
            result = llm.chat_json(prompt_msgs, temperature=0.0)
            open_questions = [str(q) for q in (result.get("open_questions") or [])[:4]]
            contradictions = [str(c) for c in (result.get("contradictions") or [])[:3]]
            next_dirs = [str(d) for d in (result.get("next_search_directions") or [])[:4]]
            confidence = float(result.get("confidence_level") or 0.5)
        except Exception as exc:
            logger.warning("[AUTO_REPORT] LLM synthesis failed: %s", exc)

    rpt = ResearchReport(
        task_id=state.task_id,
        topic=state.task[:200],
        generated_at=datetime.utcnow().isoformat(),
        executive_summary=executive_summary,
        key_findings=key_findings,
        papers=paper_rows,
        open_questions=open_questions,
        contradictions=contradictions,
        next_search_directions=next_dirs,
        confidence_level=confidence,
    )

    md = render_report_md(rpt)

    # Write per-task report
    task_dir = TASKS_DIR / state.task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "research_report.md").write_text(md, encoding="utf-8")

    # Write global report index
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    slug = _slug(state.task[:60])
    report_path = REPORTS_DIR / f"{slug}_{date_str}.md"
    report_path.write_text(md, encoding="utf-8")

    logger.info("[AUTO_REPORT] Report written to %s and %s", task_dir / "research_report.md", report_path)
    return rpt


def render_report_md(rpt: ResearchReport) -> str:
    """Render a ResearchReport as a structured markdown string."""
    confidence_pct = f"{rpt.confidence_level * 100:.0f}%"
    lines = [
        f"# Research Report: {rpt.topic[:100]}",
        f"",
        f"**Generated:** {rpt.generated_at[:19]} UTC  |  **Confidence:** {confidence_pct}  "
        f"|  **Task ID:** `{rpt.task_id[:8]}`",
        f"",
        f"---",
        f"",
        f"## Executive Summary",
        f"",
        rpt.executive_summary or "_No summary available._",
        f"",
    ]

    if rpt.key_findings:
        lines += ["## Key Findings", ""]
        for f in rpt.key_findings:
            lines.append(f"- {f}")
        lines.append("")

    if rpt.papers:
        lines += [f"## Discovered Papers ({len(rpt.papers)})", ""]
        lines.append("| Title | Year | Source | Citations |")
        lines.append("|-------|------|--------|-----------|")
        for p in rpt.papers[:20]:
            title = p["title"][:70].replace("|", "\\|")
            url = p.get("url", "")
            title_cell = f"[{title}]({url})" if url else title
            lines.append(f"| {title_cell} | {p.get('year','')} | {p.get('source','')} | {p.get('citations',0)} |")
        lines.append("")

    if rpt.open_questions:
        lines += ["## Open Questions", ""]
        for q in rpt.open_questions:
            lines.append(f"- {q}")
        lines.append("")

    if rpt.contradictions:
        lines += ["## Contradictions / Tensions", ""]
        for c in rpt.contradictions:
            lines.append(f"- {c}")
        lines.append("")

    if rpt.next_search_directions:
        lines += ["## Next Search Directions", ""]
        for d in rpt.next_search_directions:
            lines.append(f"- {d}")
        lines.append("")

    lines += ["---", f"*Auto-generated by RAG-Agent research pipeline.*"]
    return "\n".join(lines)


def load_report(task_id: str) -> ResearchReport | None:
    """Load a previously generated report for a task."""
    path = TASKS_DIR / task_id / "research_report.md"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8")  # type: ignore[return-value]
