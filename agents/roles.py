"""Role-based agent dispatch through hosted model routes."""
from __future__ import annotations

import json
from typing import Any

from pydantic import ValidationError

from .llm import HostedLLM
from .schemas import (
    AnalysisPlan,
    Critique,
    LiteratureGapAnalysis,
    HypothesisSet,
    Outline,
    Plan,
    StrategySpec,
    TaskNarrative,
)


SYSTEM_PROMPTS: dict[str, str] = {
    # ----------------------- PLANNER -----------------------
    "PLAN": """You are a research orchestrator for a hosted multi-agent system.
Given a user task, classify it and break it into concrete subtasks.

Valid task_type values:
  - "data_science": EDA, statistics, ML modeling on a dataset.
  - "trading_research": design / backtest a quantitative strategy.
  - "writing": produce an academic-style report or paper section.
  - "mixed": multiple of the above; list subtasks in execution order.

Respond ONLY with JSON matching this schema:
{"task_type": "...", "subtasks": ["..."], "rationale": "..."}""",

    # ---------------------- DATA SCIENCE --------------------
    "DS": """You are a senior quantitative data scientist. You write clean,
correct, reproducible code using pandas, numpy, scipy, statsmodels, and
scikit-learn for Python. You prefer vectorized ops over loops.

Return ONLY a fenced code block (```python) that:
1. Fetches data (use yfinance.download or tools.data.MultiSourceFetcher)
2. Preprocesses data (handle missing values, feature engineering)
3. Performs the requested statistical analysis or ML modelling
4. Prints key results with units and confidence intervals

For ML experiments you may optionally use tools.experiment.ExperimentSpec,
run_experiment, and registry.save() for tracked, reproducible runs.

Rules:
  1. ALWAYS use time-series cross-validation for temporal data (never k-fold).
  2. Report point estimates WITH confidence intervals or standard errors.
  3. State the null hypothesis and test assumptions explicitly.
  4. No fabricated data — if a file/path is not given, say so.
  5. The code must be self-contained and runnable.
  6. Explicitly state the assumed distribution, independence, and stationarity
     when relevant.
  7. Use Python by default unless the task explicitly requests R (e.g., "using R", "in R").
  8. Differentiate between stochastic variability and structural model error.

Mathematical preferences:
  - Lasso: $\\hat{\\beta} = \\arg\\min (1/(2n))\\|y - X\\beta\\|^2 + \\lambda\\|\\beta\\|_1$
  - For feature importance on nonlinear models use permutation importance,
    not just gini / gain.
  - Bootstrap CIs (B>=1000) when parametric assumptions fail.
  - Use explicit probability notation when reasoning about uncertainty,
    e.g. $P(\cdot)$, $\mathbb{E}[\cdot]$, $\mathrm{Var}(\cdot)$.
  9. When passing pandas Series or DataFrame columns to numpy functions
     (e.g. np.random.choice, np.std, np.mean), always extract a plain
     1-D array first: use `.to_numpy()` or `.values`.

 10. Use C++ (```cpp) when the task requires:
     - HFT / microsecond-nanosecond latency simulation or benchmarking
     - Ultra-fast order book processing or market microstructure simulation
     - Real-time risk aggregation at tick speed
     - Explicit computational speed comparison vs Python

     When writing C++: C++17 assumed (-O3 -march=native at compile time).
     Include only STL headers: <iostream>, <vector>, <algorithm>, <numeric>,
     <chrono>, <cmath>, <random>, <string>, <unordered_map>.
     All output via std::cout. Use <chrono> to measure and print timing in
     microseconds. End main() with return 0. Self-contained single file.
     Return ONLY a fenced code block (```cpp).

 11. Use C (```c) only when the task explicitly requires C (not C++).
     C11 assumed (-O3 -march=native). All output via printf.
     Self-contained single file. Return ONLY a fenced code block (```c).

 12. Macroeconomic data — use FRED (no API key needed):
     from tools.fred import get_series, get_multiple
     Common series IDs (pass as string):
       'GDP'       nominal GDP (quarterly)    'GDPC1'   real GDP
       'CPIAUCSL'  CPI all-items (monthly)    'PCE'     personal consumption
       'FEDFUNDS'  fed funds rate             'DGS10'   10Y Treasury yield
       'DGS2'      2Y Treasury yield          'T10Y2Y'  yield-curve spread
       'VIXCLS'    VIX (daily)                'UNRATE'  unemployment rate
       'PAYEMS'    nonfarm payrolls            'HOUST'   housing starts
     Each returns a pd.Series with a DatetimeIndex.
     Example: vix = get_series('VIXCLS', start='2015-01-01', end='2023-12-31')

 13. Factor data — use the Ken French Data Library (requires pandas-datareader):
     from tools.ken_french import get_factors, get_portfolios, ff3_alpha
     get_factors('ff3'|'ff5'|'mom', start, end) -> DataFrame (decimal returns)
       FF3 columns: Mkt-RF, SMB, HML, RF
       FF5 adds:    RMW, CMA
     get_portfolios('25_Portfolios_5x5', start, end) -> DataFrame
     ff3_alpha(monthly_returns_series, start, end) -> dict:
       keys: alpha (annualised), alpha_t, mkt_beta, smb_beta, hml_beta, r_squared
     Example:
       ff3 = get_factors('ff3', start='2010-01-01')
       result = ff3_alpha(my_series, start='2010-01-01')""",

    # --------------------- TRADING RESEARCH -----------------
    "QUANT": """You are a quantitative researcher designing systematic
strategies. You produce a StrategySpec (JSON) that can be backtested by
a downstream tool. You NEVER place real orders — only propose specs.

You understand:
  - Fractional Kelly sizing: f_use = lambda * (mu_hat - rf) / sigma_hat^2,
    with lambda in [0.25, 0.5] to absorb estimation error in mu_hat.
  - Deflated Sharpe ratio (Bailey & Lopez de Prado) to penalize
    multiple-testing bias when sweeping hyperparameters.
  - Mean-variance optimization with L2 turnover penalty:
    max_w w'mu - (gamma/2) w'Sigma w - (tau/2)||w - w_prev||^2
  - Stochastic process assumptions, Markov properties, and Bayesian
    parameter uncertainty when modeling regime changes or risk forecasts.
  - Always describe the probability distribution or state-transition
    model that underlies any risk or return forecast.

You respect these hard caps (set by the user's config):
  - Max per-position weight, max leverage, max turnover per rebalance.

The signal_code field must be a SINGLE Python expression that, given a
pandas DataFrame `df` with columns ['open','high','low','close','volume']
indexed by date, returns a pandas Series of positions in [-1, 1].

Do NOT put complex code, assignments, or multi-line code in signal_code.
Keep it simple, e.g., "1.0" for long-only, or "(df['close'] > df['close'].rolling(50).mean()).astype(float)".
Only `df`, a small allowlist of pandas rolling/series methods, and bounded
NumPy functions such as `np.where`, `np.log`, and `np.sqrt` are available.

Respond ONLY with JSON matching the StrategySpec schema.""",

    # ----------------------- WRITER -------------------------
    "WRITE": """You are an academic writer producing publication-quality
LaTeX sections. You write precisely and cite rigorously.

Rules:
  1. Every factual claim with a citation must be supported by the
     retrieved context. If context is missing, write [CITATION NEEDED]
     instead of inventing a key.
  2. Use \\cite{key} only with keys present in the provided bibliography.
  3. Equations go in \\begin{equation} ... \\end{equation} or inline $...$.
  4. No flowery language. Prefer precise quantitative statements.
  5. When discussing statistical or probabilistic conclusions, state the
     underlying assumptions and the probability model explicitly.
  6. Do not repeat content across sections.""",

    # -------------------- LITERATURE ANALYST ----------------
    "LITERATURE_ANALYST": """You are a systematic literature reviewer for a research agent.

Given a research topic and retrieved paper abstracts, produce a structured
gap analysis that will guide hypothesis formation and experiment design.

Respond ONLY with JSON matching the LiteratureGapAnalysis schema:
{"gaps": [...], "key_findings": [...], "suggested_experiments": [...], "related_topics": [...]}

Be specific and quantitative. Gaps should reference what is NOT yet studied.
Suggested experiments must be concise and actionable (one sentence each).""",

    # ------------------- HYPOTHESIS FORMER ------------------
    "HYPOTHESIS_FORMER": """You are a research scientist forming testable hypotheses.

Given a research topic and a literature gap analysis, produce a HypothesisSet.
The primary hypothesis MUST be:
  - Falsifiable (can be proven false with data)
  - Quantitative (references measurable quantities, e.g. "exceeds 0.5 Sharpe")
  - Specific (names the variables, dataset, and time period where possible)

Respond ONLY with JSON matching the HypothesisSet schema:
{"primary": "...", "secondary": [...], "null_hypotheses": [...], "methodology": "..."}""",

    # ----------------------- NARRATOR -----------------------
    "NARRATOR": """You are a research communicator writing a structured report.

Given a task description, the plan, the generated code, and the execution output,
write a clear plain-English narrative explaining what was done and WHY at each step.

Respond ONLY with JSON matching the TaskNarrative schema:
{
  "objective": "<what the task asked for and why it matters>",
  "methodology": "<step-by-step explanation of the approach and reasoning behind each choice>",
  "key_results": ["<finding 1>", "<finding 2>", ...],
  "analysis": "<interpretation of the numbers — what they mean, how they compare to benchmarks>",
  "conclusions": "<actionable takeaways; what should be done next or what this proves>",
  "limitations": "<caveats: data quality, time period, model assumptions, execution errors>"
}

Rules:
  - methodology must explain WHY each modelling choice was made, not just what was done.
  - key_results must be quantitative where possible (e.g. "Sharpe = 1.23 vs benchmark 0.8").
  - If code failed (stderr present), describe what went wrong and why in limitations.
  - Be concise but precise. No marketing language.""",

    # ----------------------- CRITIC -------------------------
    "CRITIC": """You are a strict reviewer. Evaluate the provided output
against these criteria and return ONLY JSON matching the Critique schema.

Criteria:
  - grounded: every empirical claim is supported by retrieved context
    or tool output.
  - numerics_ok: units, signs, magnitudes are plausible.
  - code_runs: if code was produced, did the sandbox return 0?
  - accept: true ONLY if grounded AND numerics_ok AND no blocking issues.
  - assumptions_explicit: probabilistic and statistical assumptions are
    clearly stated when used.
  - bayesian_consistency: Bayesian claims should mention prior/posterior
    updating or evidence when appropriate.
  - markov_reasoning: Markov/regime assumptions should be justified if used.

Be skeptical of:
  - Sharpe ratios > 3 without explanation.
  - Model R^2 > 0.95 on financial returns (likely leakage).
  - Citations without evidence in the retrieved docs.
  - Any 'guaranteed', 'risk-free', or overclaim language.
  - Probability statements without clear assumptions or supporting math.""",
}


# ---------------------- citation mapping ----------------------

def map_citations_to_response(response: str, context_docs: list[dict]) -> str:
    """Replace [doc:id] references in response with proper citations.
    
    For BibTeX entries: [doc:bib::key] → \cite{key}
    For document chunks: [doc:filename::hash] → [1] with footnote
    
    Args:
        response: LLM response text with [doc:id] references
        context_docs: list of retrieved documents with metadata
        
    Returns:
        response with citations instead of [doc:id] references
    """
    import re
    
    # Build citation mapping
    citation_map = {}
    bib_refs = []
    doc_refs = []
    
    for i, doc in enumerate(context_docs):
        doc_id = doc.get('id', '')
        meta = doc.get('meta', {})
        
        if meta.get('kind') == 'bib':
            # BibTeX citation
            key = meta.get('key', doc_id.split('::')[-1])
            citation_map[doc_id] = f"\\cite{{{key}}}"
            if key not in bib_refs:
                bib_refs.append(key)
        else:
            # Document reference with footnote
            source = meta.get('source', 'unknown')
            citation_map[doc_id] = f"[{len(doc_refs) + 1}]"
            doc_refs.append(f"[{len(doc_refs) + 1}] {source}")
    
    # Replace [doc:id] patterns in response
    def replace_citation(match):
        doc_id = match.group(1)
        return citation_map.get(doc_id, f"[UNKNOWN:{doc_id}]")
    
    cited_response = re.sub(r'\[doc:([^\]]+)\]', replace_citation, response)
    
    # Add footnotes for document references
    if doc_refs:
        cited_response += "\n\n" + "\n".join(doc_refs)
    
    return cited_response


def build_messages(
    role: str,
    user_msg: str,
    *,
    context_docs: list[dict] | None = None,
    extra_system: str | None = None,
) -> list[dict]:
    sys = SYSTEM_PROMPTS[role]
    if extra_system:
        sys = sys + "\n\n" + extra_system
    msgs: list[dict] = [{"role": "system", "content": sys}]
    if context_docs:
        ctx = "\n\n".join(
            f"[doc:{d.get('id','?')}] {d['text']}" for d in context_docs
        )
        msgs.append({
            "role": "system",
            "content": f"Retrieved context (use for grounding, do not paste verbatim):\n{ctx}",
        })
    msgs.append({"role": "user", "content": user_msg})
    return msgs


# ---------------------- typed helpers ----------------------


def plan(llm: HostedLLM, task: str) -> Plan:
    msgs = build_messages("PLAN", task)
    raw = llm.chat_json(msgs, schema_hint=json.dumps(Plan.model_json_schema()), role="planner")
    return Plan.model_validate(raw)


def analyze(llm: HostedLLM, task: str, docs: list[dict], *, feedback: str | None = None, decoding: Any = None) -> str:
    """DS agent returns pipeline execution code as a markdown fenced block."""
    extra_sys = ""
    if decoding:
        extra_sys += f"\n\nProblem Decoding:\n{decoding.model_dump_json(indent=2)}"
    if feedback:
        extra_sys += f"\n\nFeedback: {feedback}"

    msgs = build_messages("DS", task, context_docs=docs, extra_system=extra_sys)
    response = llm.chat(msgs, temperature=0.2, role="model")
    # Map citations in the response
    return map_citations_to_response(response, docs)


def design_strategy(llm: HostedLLM, task: str, docs: list[dict]) -> StrategySpec:
    msgs = build_messages("QUANT", task, context_docs=docs)
    raw = llm.chat_json(msgs, schema_hint=json.dumps(StrategySpec.model_json_schema()), role="quant")
    try:
        return StrategySpec.model_validate(raw)
    except ValidationError as e:
        # Retry once with the error message fed back
        retry_msgs = msgs + [
            {"role": "assistant", "content": json.dumps(raw)},
            {"role": "user", "content": f"Your JSON failed validation: {e}. Fix and resend."},
        ]
        raw2 = llm.chat_json(retry_msgs, schema_hint=json.dumps(StrategySpec.model_json_schema()), role="quant")
        return StrategySpec.model_validate(raw2)


def outline_paper(llm: HostedLLM, task: str, docs: list[dict]) -> Outline:
    msgs = build_messages("WRITE", f"Produce an outline for: {task}", context_docs=docs)
    raw = llm.chat_json(msgs, schema_hint=json.dumps(Outline.model_json_schema()), role="writer")
    return Outline.model_validate(raw)


def draft_section(
    llm: HostedLLM,
    section_title: str,
    key_points: list[str],
    docs: list[dict],
    bib_keys: list[str],
    target_words: int = 500,
) -> str:
    extra = f"Available bibliography keys: {bib_keys}\nTarget length: ~{target_words} words."
    user = (
        f"Write the LaTeX section titled: {section_title}\n"
        f"Key points to cover:\n" + "\n".join(f"- {p}" for p in key_points)
    )
    msgs = build_messages("WRITE", user, context_docs=docs, extra_system=extra)
    response = llm.chat(msgs, temperature=0.6, role="writer")
    return map_citations_to_response(response, docs)


def critique(
    llm: HostedLLM,
    artifact: str,
    context: str = "",
    code_run_code: int | None = None,
) -> Critique:
    tail = f"\n\nCode sandbox exit code: {code_run_code}" if code_run_code is not None else ""
    msgs = build_messages(
        "CRITIC",
        f"Evaluate this output:\n---\n{artifact}\n---\n{context}{tail}",
    )
    raw = llm.chat_json(msgs, schema_hint=json.dumps(Critique.model_json_schema()), role="risk")
    return Critique.model_validate(raw)


# ---------------------- research pipeline roles ----------------------


def analyze_literature_gaps(
    llm: HostedLLM,
    topic: str,
    docs: list[dict],
    lit_context: str = "",
) -> LiteratureGapAnalysis:
    """Identify research gaps given retrieved papers and topic."""
    extra = lit_context if lit_context else None
    msgs = build_messages(
        "LITERATURE_ANALYST",
        f"Research topic: {topic}\n\nAnalyze gaps in the retrieved literature.",
        context_docs=docs,
        extra_system=extra,
    )
    raw = llm.chat_json(msgs, schema_hint=json.dumps(LiteratureGapAnalysis.model_json_schema()), role="research")
    try:
        return LiteratureGapAnalysis.model_validate(raw)
    except Exception:
        return LiteratureGapAnalysis(
            gaps=["Insufficient literature context to identify gaps"],
            suggested_experiments=["Conduct broad exploratory analysis first"],
        )


def form_hypotheses(
    llm: HostedLLM,
    topic: str,
    gap_analysis: LiteratureGapAnalysis,
) -> HypothesisSet:
    """Form testable hypotheses from a literature gap analysis."""
    user_msg = (
        f"Research topic: {topic}\n\n"
        f"Literature gaps:\n" + "\n".join(f"- {g}" for g in gap_analysis.gaps) + "\n\n"
        f"Suggested experiments:\n" + "\n".join(f"- {e}" for e in gap_analysis.suggested_experiments)
    )
    msgs = build_messages("HYPOTHESIS_FORMER", user_msg)
    raw = llm.chat_json(msgs, schema_hint=json.dumps(HypothesisSet.model_json_schema()), role="research")
    try:
        return HypothesisSet.model_validate(raw)
    except Exception:
        return HypothesisSet(
            primary=f"Investigate: {topic}",
            methodology="Exploratory data analysis with statistical testing",
        )


def draft_section_with_citation_check(
    llm: HostedLLM,
    section_title: str,
    key_points: list[str],
    docs: list[dict],
    bib_keys: list[str],
    target_words: int = 500,
) -> dict:
    """Draft LaTeX section and report any [CITATION NEEDED] markers.

    Returns {"text": str, "uncited_claims": list[str], "citation_ok": bool}.
    """
    text = draft_section(llm, section_title, key_points, docs, bib_keys, target_words)
    import re
    uncited = re.findall(r"[^\n]*\[CITATION NEEDED\][^\n]*", text)
    return {
        "text": text,
        "uncited_claims": uncited,
        "citation_ok": len(uncited) == 0,
    }


def generate_narrative(
    llm: HostedLLM,
    task: str,
    subtasks: list[str],
    code: str,
    stdout: str,
    stderr: str,
) -> TaskNarrative:
    """Generate a structured plain-English narrative report from task execution results."""
    subtask_str = "\n".join(f"- {s}" for s in subtasks) if subtasks else "N/A"
    user_msg = (
        f"Task: {task}\n\n"
        f"Plan subtasks:\n{subtask_str}\n\n"
        f"Code produced (first 3000 chars):\n```\n{code[:3000]}\n```\n\n"
        f"Execution output (stdout, first 2000 chars):\n{stdout[:2000]}\n\n"
        f"Execution errors (stderr, first 800 chars):\n{stderr[:800]}"
    )
    msgs = build_messages("NARRATOR", user_msg)
    raw = llm.chat_json(msgs, schema_hint=json.dumps(TaskNarrative.model_json_schema()), role="narrator")
    try:
        return TaskNarrative.model_validate(raw)
    except Exception:
        ran_ok = not stderr.strip() or "Error" not in stderr
        return TaskNarrative(
            objective=task,
            methodology="Automated analysis pipeline executed.",
            key_results=[stdout[:300]] if stdout.strip() else ["No output captured."],
            analysis="See raw output above for details.",
            conclusions="Review the generated code and output for findings.",
            limitations=stderr[:300] if stderr.strip() else "",
        )
