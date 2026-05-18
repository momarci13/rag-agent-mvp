"""Orchestrator: a tiny state loop, no LangGraph dependency.

For an MVP on a laptop we don't need a graph framework — a dict and a
while-loop does the same job in 80 lines with zero install cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any
from uuid import uuid4

from rag.hybrid import LiteHybridRAG

from .llm import OllamaLLM
from . import roles
from .problem_decoder import decode_problem, validate_requirements
from tools import scholar


@dataclass
class Message:
    """Represents a single message in the task conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    iteration: int = 0  # Which iteration this message belongs to


@dataclass
class RunState:
    task: str
    task_type: str = ""
    subtasks: list[str] = field(default_factory=list)
    artifacts: list[dict] = field(default_factory=list)
    critiques: list[dict] = field(default_factory=list)
    iterations: int = 0
    accepted: bool = False
    decoding: Any = None  # Problem decoding result
    
    # Conversation & metadata fields (NEW)
    task_id: str = field(default_factory=lambda: str(uuid4()))
    messages: list[Message] = field(default_factory=list)
    parent_run_id: str | None = None
    tags: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    branch_name: str | None = None
    template_data: dict[str, Any] | None = None


def _build_feedback(state: "RunState") -> str | None:
    """Extract actionable feedback from last artifact + critique for the next iteration."""
    if state.iterations <= 1 or not state.artifacts:
        return None
    last_payload = state.artifacts[-1].get("payload", {})
    last_critique = state.critiques[-1] if state.critiques else {}
    parts = []
    stderr = last_payload.get("stderr", "").strip()
    if stderr:
        parts.append(f"Previous run stderr:\n{stderr}")
    suggestions = last_critique.get("suggested_revisions", [])
    if suggestions:
        parts.append("Reviewer suggestions:\n" + "\n".join(f"- {s}" for s in suggestions))
    return "\n\n".join(parts) + "\n\nFix these issues in your revised code." if parts else None


def run(
    task: str,
    llm: OllamaLLM,
    rag: LiteHybridRAG,
    *,
    max_iter: int = 2,
    tools: dict[str, Any] | None = None,
) -> RunState:
    """Single-shot pipeline: plan -> execute role -> critique -> revise.

    `tools` is a dict of callables the executing role may need:
      - "run_py": sandbox runner for DS role
      - "backtest": backtest engine for QUANT role
      - "latex_build": LaTeX compiler for WRITE role
    """
    tools = tools or {}
    state = RunState(task=task)

    # 0. DECODE PROBLEM
    docs_for_decoding = rag.retrieve(task, k=3)  # Quick retrieval for context
    decoding = decode_problem(llm, task, docs_for_decoding)
    validation_warnings = validate_requirements(decoding)
    if validation_warnings:
        print(f"[GRAPH] Decoding warnings: {validation_warnings}")
    state.decoding = decoding  # Add to state for later use

    # 1. PLAN
    p = roles.plan(llm, task)
    state.task_type = p.task_type
    state.subtasks = p.subtasks

    # 1.5. SCHOLAR AUGMENTATION: Search arXiv for relevant papers
    papers, scholar_context = [], ""
    try:
        papers, scholar_context = scholar.scholar_augment_task(task, n_papers=5)
        if papers:
            chunks_added = rag.ingest_papers(papers)
            print(f"[GRAPH] Ingested {len(papers)} papers ({chunks_added} chunks) into KB")
        else:
            print("[GRAPH] No relevant papers found for scholar augmentation")
    except Exception as e:
        print(f"[GRAPH] Scholar augmentation failed, continuing without it: {e}")
        scholar_context = ""

    while state.iterations < max_iter and not state.accepted:
        state.iterations += 1
        docs = rag.retrieve(task, k=6)
        
        # Add scholar context if available
        if scholar_context:
            docs.insert(0, {
                "id": "scholar_context",
                "text": scholar_context,
                "meta": {"kind": "scholar", "source": "arxiv_dynamic"}
            })

        # 2. EXECUTE according to task type
        if state.task_type == "data_science":
            feedback = None
            if state.iterations > 1 and state.artifacts:
                last_payload = state.artifacts[-1]["payload"]
                last_critique = state.critiques[-1] if state.critiques else {}
                parts = []
                stderr = last_payload.get("stderr", "").strip()
                if stderr:
                    parts.append(f"Previous run stderr:\n{stderr}")
                suggestions = last_critique.get("suggested_revisions", [])
                if suggestions:
                    parts.append("Reviewer suggestions:\n" + "\n".join(f"- {s}" for s in suggestions))
                if parts:
                    feedback = "\n\n".join(parts) + "\n\nFix these issues in your revised code."
            code_md = roles.analyze(llm, task, docs, feedback=feedback, decoding=state.decoding)
            lang, code = _extract_code(code_md)
            if code:
                if lang == "python":
                    code = _NUMPY_COMPAT + "\n" + code
                elif lang == "r":
                    code = _R_COMPAT + "\n" + code
            run_result = {"code": "", "stdout": "", "stderr": "", "returncode": None}
            if tools.get("run_py") and code and lang == "python":
                run_result = tools["run_py"](code)
                run_result["code"] = code
            elif lang == "r":
                # For R, just store the code without execution
                run_result["code"] = code
            elif tools.get("run_cpp") and code and lang == "cpp":
                run_result = tools["run_cpp"](code)
                run_result["code"] = code
            elif tools.get("run_c") and code and lang == "c":
                run_result = tools["run_c"](code)
                run_result["code"] = code
            state.artifacts.append({"type": "ds", "payload": run_result, "raw": code_md, "iteration": state.iterations})

            # 3. CRITIQUE
            c = roles.critique(
                llm,
                f"Code:\n{code}\n\nstdout:\n{run_result.get('stdout','')[:1500]}\n"
                f"stderr:\n{run_result.get('stderr','')[:500]}",
                code_run_code=run_result.get("returncode"),
            )

        elif state.task_type == "trading_research":
            spec = roles.design_strategy(llm, task, docs)
            bt_result = None
            if tools.get("backtest"):
                bt_result = tools["backtest"](spec)
            state.artifacts.append({
                "type": "quant",
                "payload": {"spec": spec.model_dump(), "backtest": bt_result},
                "iteration": state.iterations,
            })
            c = roles.critique(
                llm,
                f"Strategy: {spec.model_dump_json()}\nBacktest: {bt_result}",
            )

        elif state.task_type == "writing":
            outline = roles.outline_paper(llm, task, docs)
            sections_out = []
            bib_keys = _collect_bib_keys(rag)
            for sec in outline.sections:
                sec_docs = rag.retrieve(sec.title + " " + " ".join(sec.key_points), k=4)
                body = roles.draft_section(
                    llm, sec.title, sec.key_points, sec_docs, bib_keys, sec.target_words,
                )
                sections_out.append({"title": sec.title, "body": body})
            full_tex = _assemble_tex(outline, sections_out)
            pdf_path = None
            if tools.get("latex_build"):
                rag_bib = _extract_bibtex_from_rag(rag)
                pdf_path = tools["latex_build"](full_tex, rag_bib)
            state.artifacts.append({
                "type": "writing",
                "payload": {"tex": full_tex, "pdf": pdf_path, "outline": outline.model_dump()},
                "iteration": state.iterations,
            })
            c = roles.critique(llm, full_tex[:3000])

        elif state.task_type == "mixed":
            # naive mixed: run DS then writing
            state.task_type = "data_science"
            continue

        else:
            raise ValueError(f"Unknown task_type: {state.task_type}")

        state.critiques.append(c.model_dump())
        state.accepted = c.accept

        # Generate narrative + PDF report for the last artifact
        _attach_narrative_report(state.artifacts[-1], state, llm, tools)

        # Feed accepted artifact back into RAG for KB expansion
        if state.accepted:
            _ingest_artifact_into_rag(state.artifacts[-1], state, rag)

    return state


def research_run(
    task: str,
    llm: OllamaLLM,
    rag: LiteHybridRAG,
    *,
    max_iter: int = 2,
    tools: dict[str, Any] | None = None,
    n_papers: int = 8,
    kg_enabled: bool = True,
) -> RunState:
    """Full staged research pipeline:

    Stage 1 — Literature acquisition (arXiv search + RAG ingest)
    Stage 2 — Planning + Literature gap analysis + Hypothesis formation
    Stage 3 — Execute (DS / trading / writing with hypothesis context)
    Stage 4 — Knowledge graph update from results

    Falls back gracefully if any stage fails.
    """
    from tools.literature import acquire_literature, literature_context

    tools = tools or {}
    state = RunState(task=task, tags=["research"])

    # ── Stage 0: decode ──────────────────────────────────────────────────────
    docs_decode = rag.retrieve(task, k=3)
    decoding = decode_problem(llm, task, docs_decode)
    validation_warnings = validate_requirements(decoding)
    if validation_warnings:
        print(f"[RESEARCH] Decoding warnings: {validation_warnings}")
    state.decoding = decoding

    # ── Stage 1: Literature ───────────────────────────────────────────────────
    print("[RESEARCH] Stage 1: Literature acquisition...")
    papers: list = []
    lit_ctx = ""
    try:
        papers, skipped = acquire_literature(task, n_papers=n_papers, skip_known=True, rag=rag)
        lit_ctx = literature_context(papers)
        state.tags.append(f"papers:{len(papers)}")
        print(f"[RESEARCH] {len(papers)} new papers acquired, {len(skipped)} already known")
    except Exception as e:
        print(f"[RESEARCH] Literature acquisition failed (continuing): {e}")

    # ── Stage 2: Plan + Gap analysis + Hypotheses ─────────────────────────────
    print("[RESEARCH] Stage 2: Plan + Hypothesis formation...")
    p = roles.plan(llm, task)
    state.task_type = p.task_type
    state.subtasks = p.subtasks

    gap_analysis = None
    hypotheses = None
    try:
        lit_docs = rag.retrieve(task + " " + " ".join(p.subtasks[:3]), k=8)
        gap_analysis = roles.analyze_literature_gaps(llm, task, lit_docs, lit_ctx)
        hypotheses = roles.form_hypotheses(llm, task, gap_analysis)
        print(f"[RESEARCH] Primary hypothesis: {hypotheses.primary[:80]}...")
    except Exception as e:
        print(f"[RESEARCH] Gap analysis / hypothesis formation failed (continuing): {e}")

    # Store Stage 1+2 as a literature artifact
    state.artifacts.append({
        "type": "literature",
        "payload": {
            "papers": [
                {"id": p.arxiv_id, "title": p.title, "year": p.published[:4]}
                for p in papers
            ],
            "gap_analysis": gap_analysis.model_dump() if gap_analysis else {},
            "hypotheses": hypotheses.model_dump() if hypotheses else {},
        },
    })

    # ── Stage 3: Execute ──────────────────────────────────────────────────────
    print(f"[RESEARCH] Stage 3: Execute ({state.task_type})...")

    # Scholar augmentation (same as in `run`)
    scholar_context = ""
    try:
        from tools import scholar
        arxiv_papers, scholar_context = scholar.scholar_augment_task(task, n_papers=3)
        if arxiv_papers:
            rag.ingest_papers(arxiv_papers)
    except Exception as e:
        print(f"[RESEARCH] Scholar augmentation skipped: {e}")

    while state.iterations < max_iter and not state.accepted:
        state.iterations += 1
        docs = rag.retrieve(task, k=6)
        if scholar_context:
            docs.insert(0, {
                "id": "scholar_context",
                "text": scholar_context,
                "meta": {"kind": "scholar", "source": "arxiv_dynamic"},
            })

        hyp_extra = (
            f"\n\nResearch hypothesis: {hypotheses.primary}"
            f"\nMethodology: {hypotheses.methodology}"
        ) if hypotheses else ""

        if state.task_type == "data_science":
            feedback = _build_feedback(state)
            decoding_ctx = state.decoding
            code_md = roles.analyze(
                llm, task, docs,
                feedback=feedback,
                decoding=decoding_ctx,
            )
            # Inject hypothesis context into the extra_system via a second analyze call
            # if hypotheses exist and this is the first iteration
            if hypotheses and state.iterations == 1 and hyp_extra:
                try:
                    code_md = roles.analyze(
                        llm, task, docs,
                        feedback=f"Additional research context:{hyp_extra}",
                        decoding=decoding_ctx,
                    )
                except Exception:
                    pass  # Use first attempt

            lang, code = _extract_code(code_md)
            if code:
                if lang == "python":
                    code = _NUMPY_COMPAT + "\n" + code
                elif lang == "r":
                    code = _R_COMPAT + "\n" + code
            run_result = {"code": "", "stdout": "", "stderr": "", "returncode": None}
            if tools.get("run_py") and code and lang == "python":
                run_result = tools["run_py"](code)
                run_result["code"] = code
            elif lang == "r":
                run_result["code"] = code
            elif tools.get("run_cpp") and code and lang == "cpp":
                run_result = tools["run_cpp"](code)
                run_result["code"] = code
            elif tools.get("run_c") and code and lang == "c":
                run_result = tools["run_c"](code)
                run_result["code"] = code
            state.artifacts.append({"type": "ds", "payload": run_result, "raw": code_md, "iteration": state.iterations})
            c = roles.critique(
                llm,
                f"Code:\n{code}\nstdout:\n{run_result.get('stdout','')[:1500]}"
                f"\nstderr:\n{run_result.get('stderr','')[:500]}",
                code_run_code=run_result.get("returncode"),
            )

        elif state.task_type == "trading_research":
            spec = roles.design_strategy(llm, task, docs)
            bt_result = tools["backtest"](spec) if tools.get("backtest") else None
            state.artifacts.append({
                "type": "quant",
                "payload": {"spec": spec.model_dump(), "backtest": bt_result},
                "iteration": state.iterations,
            })
            c = roles.critique(llm, f"Strategy: {spec.model_dump_json()}\nBacktest: {bt_result}")

        elif state.task_type == "writing":
            outline = roles.outline_paper(llm, task, docs)
            sections_out = []
            bib_keys = _collect_bib_keys(rag)
            for sec in outline.sections:
                sec_docs = rag.retrieve(sec.title + " " + " ".join(sec.key_points), k=4)
                result = roles.draft_section_with_citation_check(
                    llm, sec.title, sec.key_points, sec_docs, bib_keys, sec.target_words
                )
                sections_out.append({"title": sec.title, "body": result["text"]})
                if result["uncited_claims"]:
                    print(f"[RESEARCH] Section '{sec.title}': {len(result['uncited_claims'])} uncited claims")
            full_tex = _assemble_tex(outline, sections_out)
            pdf_path = None
            if tools.get("latex_build"):
                rag_bib = _extract_bibtex_from_rag(rag)
                pdf_path = tools["latex_build"](full_tex, rag_bib)
            state.artifacts.append({
                "type": "writing",
                "payload": {"tex": full_tex, "pdf": pdf_path, "outline": outline.model_dump()},
                "iteration": state.iterations,
            })
            c = roles.critique(llm, full_tex[:3000])

        elif state.task_type == "mixed":
            state.task_type = "data_science"
            continue

        else:
            raise ValueError(f"Unknown task_type: {state.task_type}")

        state.critiques.append(c.model_dump())
        state.accepted = c.accept

        # Generate narrative + PDF report for the last artifact
        _attach_narrative_report(state.artifacts[-1], state, llm, tools)

        # Feed accepted artifact back into RAG for KB expansion
        if state.accepted:
            _ingest_artifact_into_rag(state.artifacts[-1], state, rag)

    # ── Stage 4: Knowledge graph ──────────────────────────────────────────────
    if kg_enabled:
        print("[RESEARCH] Stage 4: Updating knowledge graph...")
        try:
            from kg.graph import ResearchKnowledgeGraph
            kg = ResearchKnowledgeGraph()
            kg.ingest_run_state(state, papers=papers)
            print(f"[RESEARCH] {kg.summarize()}")
        except Exception as e:
            print(f"[RESEARCH] Knowledge graph update failed (non-critical): {e}")

    return state


# ---------- helpers ----------

# Patches np.random.choice to accept pandas Series/DataFrames.
# Newer numpy rejects non-array 1-D inputs; this is a common LLM codegen issue.
_NUMPY_COMPAT = """\
import numpy as _compat_np
def _compat_choice(a, *args, _orig=_compat_np.random.choice, **kwargs):
    if hasattr(a, "to_numpy"):
        a = a.to_numpy().ravel()
    return _orig(a, *args, **kwargs)
_compat_np.random.choice = _compat_choice
del _compat_np, _compat_choice

try:
    import yfinance as _compat_yf, pandas as _compat_pd
    def _compat_yf_dl(*args, _orig=_compat_yf.download, _pd=_compat_pd, **kwargs):
        df = _orig(*args, **kwargs)
        if isinstance(df.columns, _pd.MultiIndex) and df.columns.nlevels == 2:
            if df.columns.get_level_values(1).nunique() == 1:
                df.columns = df.columns.droplevel(1)
        return df
    _compat_yf.download = _compat_yf_dl
    del _compat_yf, _compat_pd, _compat_yf_dl
except ImportError:
    pass
"""

# R compatibility patches for common LLM codegen issues
_R_COMPAT = """\
# Load tidyverse if available
if (!require(tidyverse, quietly = TRUE)) {
  install.packages("tidyverse", repos = "https://cran.rstudio.com/")
  library(tidyverse)
}
# Load quantmod for financial data if available
if (!require(quantmod, quietly = TRUE)) {
  install.packages("quantmod", repos = "https://cran.rstudio.com/")
  library(quantmod)
}
"""


def _extract_code(md: str) -> tuple[str, str]:
    """Pull the first code block out of markdown, detecting language."""
    if "```r" in md:
        body = md.split("```r", 1)[1]
        return "r", body.split("```", 1)[0].strip()
    if "```python" in md:
        body = md.split("```python", 1)[1]
        return "python", body.split("```", 1)[0].strip()
    # Check cpp/c++ before plain c to avoid prefix collision
    for marker in ("```cpp", "```c++"):
        if marker in md:
            body = md.split(marker, 1)[1]
            return "cpp", body.split("```", 1)[0].strip()
    # Newline guard prevents ```cpp from matching here
    if "```c\n" in md or "```c\r" in md:
        body = md.split("```c", 1)[1]
        return "c", body.split("```", 1)[0].strip()
    if "```" in md:
        body = md.split("```", 1)[1]
        return "unknown", body.split("```", 1)[0].strip()
    return "unknown", md.strip()


def _collect_bib_keys(rag: LiteHybridRAG) -> list[str]:
    """Return all BibTeX keys registered in the RAG store."""
    try:
        keys = []
        for m in rag.iter_metadata():
            if m.get("kind") == "bib" and m.get("key"):
                keys.append(m["key"])
        return sorted(set(keys))
    except Exception:
        return []


def _ingest_artifact_into_rag(artifact: dict, state: "RunState", rag: LiteHybridRAG) -> None:
    """Chunk and add accepted task output back into the RAG knowledge base."""
    try:
        from rag.ingest import chunk_text
        import hashlib
        art_type = artifact.get("type", "")
        payload = artifact.get("payload", {})
        text_to_ingest = ""
        if art_type == "ds":
            code = payload.get("code", "")
            stdout = payload.get("stdout", "")
            text_to_ingest = f"Task: {state.task}\n\nCode:\n{code}\n\nOutput:\n{stdout}"
        elif art_type == "writing":
            text_to_ingest = payload.get("tex", "")[:4000]
        elif art_type == "quant":
            spec = payload.get("spec", {})
            text_to_ingest = (
                f"Strategy: {spec.get('name', '')}\n"
                f"Signal: {spec.get('signal_code', '')}\n"
                f"Universe: {spec.get('universe', [])}"
            )
        if not text_to_ingest.strip():
            return
        task_hash = hashlib.md5(state.task_id.encode()).hexdigest()[:8]
        chunks = chunk_text(text_to_ingest, target_tokens=256, overlap=32)
        existing = set(rag._ids)
        new_docs = [
            {
                "id": f"result::{state.task_id}::{i}",
                "text": ch,
                "meta": {"kind": "doc", "source": f"task_{task_hash}", "task_id": state.task_id},
            }
            for i, ch in enumerate(chunks)
            if f"result::{state.task_id}::{i}" not in existing
        ]
        if new_docs:
            rag.add(new_docs)
            print(f"[GRAPH] Ingested {len(new_docs)} chunks from accepted artifact into RAG")
    except Exception as e:
        print(f"[GRAPH] Artifact RAG ingest failed (non-critical): {e}")


def _extract_bibtex_from_rag(rag: LiteHybridRAG) -> str:
    """Generate BibTeX string from RAG-stored bib entries."""
    entries = []
    try:
        seen: set[str] = set()
        for m in rag.iter_metadata():
            if m.get("kind") == "bib" and m.get("key"):
                key = m["key"]
                if key in seen:
                    continue
                seen.add(key)
                title = m.get("title", "")
                year = m.get("year", "")
                entries.append(
                    f"@misc{{{key},\n"
                    f"  title = {{{title}}},\n"
                    f"  year = {{{year}}},\n"
                    f"}}"
                )
    except Exception:
        pass
    return "\n\n".join(entries)


def _assemble_tex(outline, sections: list[dict]) -> str:
    body = "\n\n".join(f"\\section{{{s['title']}}}\n{s['body']}" for s in sections)
    return rf"""\documentclass[11pt]{{article}}
\usepackage{{amsmath,amssymb,graphicx,hyperref,natbib}}
\usepackage[margin=1in]{{geometry}}
\title{{{outline.title}}}
\date{{\today}}
\begin{{document}}
\maketitle
\begin{{abstract}}
{outline.abstract}
\end{{abstract}}

{body}

\bibliographystyle{{plainnat}}
\bibliography{{refs}}
\end{{document}}
"""


def _tex_escape(text: str) -> str:
    """Escape special LaTeX characters in plain text."""
    replacements = [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"), ("%", r"\%"), ("$", r"\$"),
        ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}"),
        ("~", r"\textasciitilde{}"), ("^", r"\textasciicircum{}"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text


def _generate_report_tex(
    task: str,
    narrative: Any,
    code: str,
    stdout: str,
    artifact_type: str = "ds",
    language: str = "python",
) -> str:
    """Build a LaTeX results-report document from a narrative + code + stdout."""
    _LANG_MAP = {"python": "Python", "cpp": "C++", "c": "C", "r": "R"}
    lst_lang = _LANG_MAP.get(language, "Python")
    lang_label = _LANG_MAP.get(language, "Python")

    title = _tex_escape(task[:120])
    objective = _tex_escape(narrative.objective)
    methodology = _tex_escape(narrative.methodology)
    analysis = _tex_escape(narrative.analysis)
    conclusions = _tex_escape(narrative.conclusions)
    limitations = _tex_escape(narrative.limitations) if narrative.limitations else ""

    results_items = "\n".join(
        f"  \\item {_tex_escape(r)}" for r in narrative.key_results
    ) if narrative.key_results else "  \\item No quantitative results captured."

    # Truncate and escape code for lstlisting (no LaTeX escaping needed inside verbatim)
    code_block = code[:4000] if code else f"// No {lang_label} code generated."
    stdout_block = stdout[:2000] if stdout.strip() else "No output captured."

    limitations_section = (
        f"\\section{{Limitations}}\n{limitations}\n\n"
        if limitations else ""
    )

    return rf"""\documentclass[11pt]{{article}}
\usepackage{{amsmath,amssymb,graphicx,hyperref}}
\usepackage[margin=1in]{{geometry}}
\usepackage{{listings,xcolor}}
\usepackage{{parskip}}

\lstset{{
  basicstyle=\ttfamily\small,
  keywordstyle=\color{{blue}},
  stringstyle=\color{{red!60!black}},
  commentstyle=\color{{gray}},
  breaklines=true,
  frame=single,
  numbers=left,
  numberstyle=\tiny\color{{gray}},
  tabsize=4,
}}

\title{{\textbf{{Research Report — {lang_label}}}\\[0.5em]\large {title}}}
\date{{\today}}

\begin{{document}}
\maketitle
\tableofcontents
\newpage

\section{{Objective}}
{objective}

\section{{Methodology}}
{methodology}

\section{{Implementation}}
\begin{{lstlisting}}[language={lst_lang}]
{code_block}
\end{{lstlisting}}

\section{{Results}}
\begin{{verbatim}}
{stdout_block}
\end{{verbatim}}

\section{{Analysis}}
{analysis}

\section{{Key Findings}}
\begin{{itemize}}
{results_items}
\end{{itemize}}

\section{{Conclusions}}
{conclusions}

{limitations_section}\end{{document}}
"""


def _attach_narrative_report(
    artifact: dict,
    state: "RunState",
    llm: OllamaLLM,
    tools: dict,
) -> None:
    """Generate narrative + LaTeX PDF and attach to artifact in-place."""
    try:
        art_type = artifact.get("type", "ds")
        payload = artifact.get("payload", {})

        # Detect language from the raw markdown stored at artifact creation time
        raw_md = artifact.get("raw", "")
        lang, _ = _extract_code(raw_md) if raw_md else ("python", "")
        if lang in ("unknown", "", None):
            lang = "python"

        # Extract code and stdout from artifact
        if art_type == "ds":
            code = payload.get("code", "")
            stdout = payload.get("stdout", "")
            stderr = payload.get("stderr", "")
        elif art_type == "quant":
            spec = payload.get("spec", {})
            code = f"# Strategy: {spec.get('name','')}\n# Signal: {spec.get('signal_code','')}"
            bt = payload.get("backtest") or {}
            stdout = "\n".join(f"{k}: {v}" for k, v in bt.items()) if isinstance(bt, dict) else str(bt)
            stderr = ""
            lang = "python"  # quant specs are always Python signal code
        elif art_type == "writing":
            code = payload.get("tex", "")[:2000]
            stdout = ""
            stderr = ""
            lang = "python"  # LaTeX, fallback label
        else:
            return  # skip literature artifacts

        narrative = roles.generate_narrative(
            llm, state.task, state.subtasks, code, stdout, stderr
        )
        report_tex = _generate_report_tex(state.task, narrative, code, stdout, art_type, language=lang)

        pdf_result = None
        if tools.get("latex_build"):
            try:
                pdf_result = tools["latex_build"](report_tex)
            except Exception as e:
                print(f"[GRAPH] PDF compilation failed (non-critical): {e}")

        artifact["report"] = {
            "narrative": narrative.model_dump(),
            "tex": report_tex,
            "pdf": pdf_result,
        }
    except Exception as e:
        print(f"[GRAPH] Narrative generation failed (non-critical): {e}")
