# ROG-Agent MVP
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000
A laptop-grade **multi-agent RAG system** for data science, quantitative
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
- **KB grows automatically**: accepted task artifacts are chunked and
  ingested back into the RAG store; arXiv papers are fetched on each
  iteration and persisted.
- **Knowledge graph** (NetworkX + JSON) links papers, findings, and
  tasks across sessions for the staged research pipeline.
- **Risk gates** on every trading decision: position concentration,
  leverage cap, turnover cap, 99 % historical VaR.
- **Paper trading only by default.** Live execution requires an
  explicit env flag and is not wired into the agent loop.

## Architecture

```
        User task / follow-up message
            │
            ▼
        ┌────────┐      ┌──────────────────────────────┐
        │ PLAN   │◄────►│ Hybrid RAG                   │
        └────┬───┘      │  Chroma + BM25               │◄─────────────┐
             │          │  (BGE-small embeddings)      │              │
   ┌─────────┼──────────┴──────────────────────────────┘              │
   ▼         ▼          ▼                                              │
┌─────┐  ┌──────┐   ┌──────┐                                          │
│ DS  │  │QUANT │   │WRITE │   ← all the same Qwen2.5-7B model        │
└──┬──┘  └──┬───┘   └──┬───┘     just different system prompts        │
   │        │          │                                               │
   ▼        ▼          ▼                                               │
sandbox  backtest   LaTeX                                              │
(stats)  (vector-   (validate        KB expansion:                    │
         ized py)   citations)  accepted artifact chunks ─────────────┘
   │        │          │        arXiv papers per iteration
   └────────┼──────────┘
            ▼
        ┌────────┐
        │ CRITIC │  ── revise once if not accepted
        └────────┘
            │ accepted
            ▼
   ┌─────────────────┐      ┌──────────────────────────┐
   │  Task Storage   │      │  Knowledge Graph         │
   │  (per-task      │      │  (cross-session links:   │
   │  conversation   │      │  papers / findings /     │
   │  + branching)   │      │  tasks)                  │
   └─────────────────┘      └──────────────────────────┘
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
python run.py --ingest data/papers/ --skip-existing                  # append only new content
python run.py --ingest data/papers/ --chunk-tokens 300 --overlap-tokens 40
python run.py "Backtest a 50/200 SMA crossover on SPY since 2015"     # ~1-3 min
python run.py --kan-demo                                                # run a built-in multifidelity KAN demo
python -m uvicorn server:app --reload --host 127.0.0.1 --port 8000    # launch the local web UI
```

## Repository layout

```
rag-agent-mvp/
├── README.md                ← you are here
├── USER_MANUAL.md           ← setup, troubleshooting, math reference
├── WEB_SETUP.md             ← web server quick-start
├── requirements.txt
├── setup.sh / setup.ps1
├── run.py                   ← CLI entrypoint
├── server.py                ← FastAPI web server
├── configs/
│   └── config.yaml          ← model, RAG, risk, research knobs
├── agents/
│   ├── llm.py               ← Ollama client w/ JSON mode
│   ├── schemas.py           ← Pydantic models
│   ├── roles.py             ← 8 role prompts + typed helpers
│   ├── problem_decoder.py   ← structured task decomposition
│   └── graph.py             ← state machine (run + research_run)
├── rag/
│   ├── hybrid.py            ← Chroma + BM25 fusion
│   ├── ingest.py            ← PDF/MD/TeX/BibTeX loaders
│   ├── query_expansion.py   ← optional query expansion
│   └── metrics.py           ← retrieval eval helpers
├── tools/
│   ├── sandbox.py           ← subprocess + rlimit code execution
│   ├── risk.py              ← Sharpe, DSR, Kelly, VaR, MVO
│   ├── backtest.py          ← event-driven backtester
│   ├── tex.py               ← citation validator + LaTeX build
│   ├── data.py              ← multi-source market data fetcher
│   ├── scholar.py           ← arXiv scholar augmentation
│   ├── literature.py        ← literature acquisition registry
│   ├── analysis_pipeline.py ← data analysis pipeline
│   ├── experiment.py        ← hypothesis experiment runner
│   ├── report.py            ← LaTeX report builder
│   ├── task_conversation.py ← multi-turn conversation handler
│   ├── task_storage.py      ← task persistence + search + branching
│   ├── fred.py              ← FRED macro data
│   ├── ken_french.py        ← Fama-French factor data
│   ├── openalex.py          ← OpenAlex literature search
│   └── multifidelity_kan.py ← residual KAN model
├── kg/
│   └── graph.py             ← knowledge graph (papers/findings/tasks)
├── web/                     ← static frontend (vanilla JS)
├── data/
│   ├── papers/              ← seed docs: refs.bib, quant_basics.md, …
│   └── market/              ← yfinance cache
├── kb/                      ← Chroma + BM25 index (gitignored)
├── output/
│   ├── runs/                ← legacy CLI run snapshots
│   └── tasks/               ← per-task directories (conversations, artifacts)
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
