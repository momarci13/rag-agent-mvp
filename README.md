# RAG-Agent MVP

```bash
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000
```

A laptop-grade **autonomous research platform** for data science, quantitative
trading research, and academic writing — running entirely on a local
free LLM via Ollama.

Designed to fit on a single consumer GPU (6–8 GB VRAM, e.g. ASUS ROG
Flow X13 with mobile RTX 4060/4070), with a clean fallback path down
to 4 GB VRAM or pure CPU.

## What it does

- **Plan → Execute → Critique** pipeline driven by a small typed state
  machine (no LangGraph dependency).
- **Multi-turn conversations** per task: follow-up messages refine or
  extend the previous artifact; each task is its own chatblock. Tasks
  can be branched from any prior iteration.
- **Eight roles** sharing one warm model via system-prompt switching:
  - *Data Science* — pandas/statsmodels code generation, sandboxed
    execution, structured output. Supports Python and R code generation.
  - *Trading Research* — produces typed `StrategySpec` JSON, runs an
    event-driven backtest, computes Sharpe / Sortino / **Deflated
    Sharpe** / MDD / VaR.
  - *Academic Writer* — outline + section drafting + LaTeX assembly,
    with citation validation against your `refs.bib`.
  - *Planner, Critic, Narrator, Literature Analyst, Hypothesis Former* —
    supporting roles for research pipelines and evaluation.
- **Hybrid RAG** — Chroma (dense, BGE-small embeddings) fused with BM25,
  greedy knapsack context packing under a token budget. Optional
  cross-encoder reranker (disabled by default to save VRAM).
  - **LLM query expansion** — the model rewrites queries into 3 academic
    variants before retrieval; falls back to rule-based synonyms.
  - **Failure-driven KB expansion** — if retrieval quality falls below a
    configurable threshold, the system automatically searches for and
    ingests gap-filling papers.
- **Multi-source paper discovery** across arXiv, OpenAlex (250 M works),
  and Semantic Scholar. Discovered papers appear in a checkbox panel in
  the web UI so you choose exactly what enters the KB.
- **Persistent citation DAG** (NetworkX + JSON). After each search, the
  system BFS-traverses up to 2 citation hops via OpenAlex and stores the
  directed graph in `output/citation_dag.json`. PageRank and novelty
  detection surface the most important unseen papers.
- **Conversation memory compression** — threads longer than 20 messages
  are summarised by the LLM and stored as a rolling `MemorySummary`,
  preventing context-window blowup on long sessions.
- **Cross-task semantic memory** — accepted artifact findings are embedded
  and stored in Chroma with `kind=finding` metadata, enabling dense
  retrieval of prior results across all past tasks.
- **Structured research reports** auto-generated after every research
  pipeline run: executive summary, key findings, paper table, open
  questions, contradictions, next search directions.
- **Non-blocking web server** — `/run-task` and `/research-task` return
  `{status: "queued", task_id}` immediately; the browser opens a
  Server-Sent Events stream and animates a live progress bar until the
  task completes.
- **Autonomous research loop** (`--auto-research` CLI flag or
  `POST /api/research/autonomous`) — iterative
  SEARCH → RETRIEVE → GAP-DETECT → EXPAND → REPORT cycle that keeps
  running until knowledge-base quality converges.
- **KB grows automatically**: accepted task artifacts are chunked and
  ingested back into the RAG store; papers fetched during research
  iterations persist across sessions.
- **Knowledge graph** (NetworkX + JSON) links papers, findings, and
  tasks across sessions for the staged research pipeline.
- **Risk gates** on every trading decision: position concentration,
  leverage cap, turnover cap, 99 % historical VaR.
- **Structured logging** throughout — every module emits to
  `output/agent.log` via Python's standard `logging` module.
- **Paper trading only by default.** Live execution requires an
  explicit env flag and is not wired into the agent loop.

## Architecture

```
        User task / follow-up message
            │
            ▼
        ┌────────┐      ┌──────────────────────────────────────┐
        │ PLAN   │◄────►│ Hybrid RAG                           │
        └────┬───┘      │  Chroma (dense, BGE-small)           │◄──────────────────────┐
             │          │  + BM25 (sparse)                     │                       │
             │          │  + LLM query expansion               │◄── kind=finding ──────┤
   ┌─────────┼──────────┴──────────────────────────────────────┘                       │
   ▼         ▼          ▼                                                               │
┌─────┐  ┌──────┐   ┌──────┐                                                           │
│ DS  │  │QUANT │   │WRITE │   ← same Qwen2.5-7B, different system prompts             │
└──┬──┘  └──┬───┘   └──┬───┘                                                           │
   │        │          │                                                                │
   ▼        ▼          ▼               KB expansion:                                   │
sandbox  backtest   LaTeX         accepted artifact chunks ────────────────────────────┤
(stats)  (vectorised (validate    multi-source papers (arXiv + OpenAlex + S2) ─────────┤
          Python)   citations)    citation DAG 2-hop BFS ───────────────────────────────┤
   │        │          │          semantic findings (cross-task) ────────────────────────┘
   └────────┼──────────┘
            ▼
        ┌────────┐
        │ CRITIC │  ── revise once if not accepted
        └────────┘
            │ accepted
            ▼
   ┌─────────────────┐   ┌────────────────────────┐   ┌─────────────────────┐
   │  Task Storage   │   │  Knowledge Graph        │   │  Research Report    │
   │  (per-task      │   │  (cross-session links:  │   │  (auto-generated    │
   │  conversation   │   │  papers / findings /    │   │  markdown + index)  │
   │  + branching    │   │  tasks)                 │   │                     │
   │  + memory       │   └────────────────────────┘   └─────────────────────┘
   │  compression)   │
   └─────────────────┘

  Autonomous loop (--auto-research):
  SEARCH → RETRIEVE → GAP-DETECT → EXPAND → REPORT  (×N iterations)
```

## Why these choices for laptop hardware

- **One 7B model, role-switched.** Loading two GGUFs onto an 8 GB
  laptop GPU thrashes VRAM. We pay ~50 ms in extra prompt tokens
  instead of 30 s of swap.
- **Chroma over Qdrant.** Single-file SQLite store, no server, no
  Docker.
- **Cross-encoder reranker optional.** Disabled by default (saves
  ~300 MB VRAM); enable in `configs/config.yaml` on machines with
  headroom.
- **No LangGraph.** A typed `dataclass` state and a `while` loop is 80
  lines and zero install cost.
- **CPU-side stats.** `statsmodels`, `scikit-learn`, `cvxpy` all run on
  CPU, leaving the GPU for the LLM only.
- **SSE instead of WebSockets.** Server-Sent Events are one-way and
  trivially supported by every modern browser with zero library overhead
  on the client side.

## Quickstart

See **`USER_MANUAL.md`** for full setup, troubleshooting, and the
math reference. The 60-second version:

```bash
# Linux / macOS / WSL2
bash setup.sh

# Windows native
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Then:

```bash
python run.py --ingest data/papers/                                   # build KB (~30 s)
python run.py "Backtest a 50/200 SMA crossover on SPY since 2015"     # ~1-3 min
python run.py --research "Low-volatility anomaly in equity markets"   # staged research
python run.py --auto-research "GARCH volatility clustering" --iterations 3  # autonomous loop
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000    # launch web UI
```

## Repository layout

```
rag-agent-mvp/
├── README.md                ← you are here
├── USER_MANUAL.md           ← setup, troubleshooting, math reference
├── WEB_SETUP.md             ← web server quick-start
├── requirements.txt
├── setup.sh / setup.ps1
├── run.py                   ← CLI entrypoint (includes --auto-research)
├── server.py                ← FastAPI web server (BackgroundTasks + SSE)
├── configs/
│   └── config.yaml          ← model, RAG, risk, research, loop knobs
├── agents/
│   ├── llm.py               ← Ollama client w/ JSON mode
│   ├── schemas.py           ← Pydantic models
│   ├── roles.py             ← 8 role prompts + typed helpers
│   ├── problem_decoder.py   ← structured task decomposition
│   └── graph.py             ← state machine (run + research_run)
├── rag/
│   ├── hybrid.py            ← Chroma + BM25 fusion (LLM query expansion)
│   ├── ingest.py            ← PDF/MD/TeX/BibTeX loaders
│   ├── query_expansion.py   ← LLM-based + rule-based query rewriting
│   └── metrics.py           ← retrieval eval helpers
├── tools/
│   ├── sandbox.py           ← subprocess + rlimit code execution
│   ├── risk.py              ← Sharpe, DSR, Kelly, VaR, MVO
│   ├── backtest.py          ← event-driven backtester
│   ├── tex.py               ← citation validator + LaTeX build
│   ├── data.py              ← multi-source market data fetcher
│   ├── scholar.py           ← arXiv scholar augmentation
│   ├── source_search.py     ← multi-source search (arXiv+OpenAlex+S2)
│   ├── citation_dag.py      ← persistent N-hop citation graph (NetworkX)
│   ├── auto_report.py       ← structured markdown research reports
│   ├── memory.py            ← conversation compression (rolling summary)
│   ├── semantic_memory.py   ← cross-task finding embeddings in Chroma
│   ├── kb_expansion.py      ← failure-driven auto-expansion of the KB
│   ├── research_loop.py     ← autonomous iterative research loop
│   ├── literature.py        ← literature acquisition registry
│   ├── analysis_pipeline.py ← data analysis pipeline
│   ├── experiment.py        ← hypothesis experiment runner
│   ├── report.py            ← LaTeX report builder
│   ├── task_conversation.py ← multi-turn conversation handler
│   ├── task_storage.py      ← task persistence + search + branching
│   ├── fred.py              ← FRED macro data
│   ├── ken_french.py        ← Fama-French factor data
│   ├── openalex.py          ← OpenAlex literature search + citation fetch
│   └── multifidelity_kan.py ← residual KAN model
├── kg/
│   └── graph.py             ← knowledge graph (papers/findings/tasks)
├── web/                     ← static frontend (vanilla JS + SSE client)
├── data/
│   ├── papers/              ← seed docs: refs.bib, quant_basics.md, …
│   └── market/              ← yfinance cache
├── kb/                      ← Chroma + BM25 index (gitignored)
├── output/
│   ├── runs/                ← legacy CLI run snapshots
│   ├── tasks/               ← per-task dirs (conversations, artifacts, reports)
│   ├── research_reports/    ← auto-generated markdown research reports
│   ├── citation_dag.json    ← persistent citation graph
│   └── agent.log            ← structured application log
├── examples/EXAMPLES.md     ← copy-paste tasks
└── tests/
    ├── test_risk.py
    ├── test_backtest.py
    └── test_rag.py
```

## What this MVP cannot do (honest list)

- Match GPT-4 / Claude planning quality. Expect to retry tasks. The
  CRITIC loop helps but doesn't close the gap.
- Long-form generation > 8k output tokens reliably. Split into sections.
- Intraday tick research — yfinance gives daily / hourly only; tick data
  is a separate problem.
- Run all roles in parallel. One model, one query at a time.
- Sustained heavy loops on battery without thermal throttling.

## Caveats

- **Not investment advice.** The Sharpe and Deflated Sharpe a backtest
  reports are estimates with substantial error. Paper-trade for months
  before risking real capital.
- **Not a security boundary.** The sandbox limits memory and CPU time
  but does not isolate the LLM from your filesystem. Don't run untrusted
  LLM-generated code on machines with sensitive data.
- **Citations need human review.** The validator strips invented BibTeX
  keys, but it can't tell whether the cited claim is correctly
  represented. Read the output before publishing anything.

## License

MIT — do whatever you want, no warranty.
