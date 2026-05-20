# RAG-Agent MVP — User Manual

## Contents

1. [Hardware reality check](#1-hardware-reality-check)
2. [Installation](#2-installation)
3. [First run](#3-first-run)
4. [Web interface and API](#4-web-interface-and-api)
5. [How tasks work](#5-how-tasks-work)
6. [Building your knowledge base](#6-building-your-knowledge-base)
7. [Configuration reference](#7-configuration-reference)
8. [Mathematical framework](#8-mathematical-framework)
9. [Troubleshooting](#9-troubleshooting)
10. [Extending the system](#10-extending-the-system)

---

## 1. Hardware reality check

This MVP is tuned for the ASUS ROG Flow X13. It works on anything similar.

| Resource | Recommended | Minimum | Notes |
|---|---|---|---|
| OS | Win 11 + WSL2, or native Linux | Win 10 | WSL2 gives proper resource limits in the sandbox |
| GPU VRAM | 8 GB (RTX 4060/4070 mobile) | 4 GB or none | At 4 GB, switch to the 3B model — see §6 |
| System RAM | 32 GB | 16 GB | 16 GB works but no other heavy apps |
| Disk free | 20 GB | 10 GB | ~5 GB for the model, ~2 GB for Python deps, rest for KB |
| Network | required first time | n/a | Pulling Ollama, the model, and Python deps |

Find your numbers fast:

```bash
# Linux / WSL
nvidia-smi                       # VRAM column
free -h                          # System RAM
df -h .                          # Disk

# Windows PowerShell
Get-WmiObject Win32_VideoController | Select Name, AdapterRAM
Get-WmiObject Win32_PhysicalMemory | Measure-Object Capacity -Sum
```

If you have <6 GB VRAM, **stop and edit `configs/config.yaml` first**: change
`model:` to `qwen2.5:3b-instruct-q5_K_M` (about 2.3 GB on disk, weaker math
but workable).

---

## 2. Installation

### Step 2.1 — Install Ollama

Ollama is the local LLM runtime. It's free.

| OS | Install |
|---|---|
| **Linux** | `curl -fsSL https://ollama.com/install.sh \| sh` |
| **macOS** | Download from https://ollama.com or `brew install ollama` |
| **Windows** | Download installer from https://ollama.com/download/windows. Run it. The Ollama tray icon should appear. |
| **WSL2** (recommended on X13) | Run the Linux command inside your WSL distro |

Verify:

```bash
ollama --version            # should print a version
curl http://localhost:11434/api/tags    # should return JSON (possibly empty)
```

If `curl` fails, the service isn't running:
- Linux: `ollama serve &` in another terminal.
- Windows: open the Ollama app from the Start menu.
- macOS: open Ollama.app from Applications.

### Step 2.2 — Clone / unzip the project

```bash
cd ~/projects                # or wherever you keep code
# unzip rog-agent-mvp.zip   OR  git clone <your fork>
cd rog-agent-mvp
```

### Step 2.3 — Run the setup script

This creates a Python venv, installs dependencies, pulls the model, and runs
a healthcheck.

```bash
# Linux / macOS / WSL
bash setup.sh

# Windows native
powershell -ExecutionPolicy Bypass -File setup.ps1
```

Time budget on a typical ROG Flow X13:
- Python deps: ~3-5 min (sentence-transformers brings in ~1.5 GB of torch)
- Model download: ~3-10 min depending on connection (~4.7 GB)
- First embedding model download (BGE-small): ~30 s, ~130 MB

### Step 2.4 — Optional: install LaTeX

For PDF output of writing tasks. Skip if you only want `.tex` files.

| OS | Recommended |
|---|---|
| Linux | `sudo apt install tectonic` (Debian/Ubuntu 22.04+) |
| macOS | `brew install tectonic` |
| Windows | Download `tectonic.exe` from https://github.com/tectonic-typesetting/tectonic/releases and put it on your PATH |
| WSL2 | Use the Linux instructions inside WSL |

Tectonic auto-fetches LaTeX packages on first compile. The fallback is
MiKTeX (Win) or texlive-full (Linux); both are larger.

### Step 2.5 — Activate the venv (every shell session)

```bash
# Linux / macOS / WSL
source .venv/bin/activate

# Windows
.venv\Scripts\Activate.ps1

 # on cmd
.venv\Scripts\activate.bat
```

---

## 3. First run

### Healthcheck

```bash
python run.py --healthcheck
```

You should see four green `[OK]` lines. If anything is `[FAIL]`, jump to
[Troubleshooting](#8-troubleshooting).

### Build the seed knowledge base

```bash
python run.py --ingest data/papers/
```

This indexes the seed BibTeX (10 references) and the `quant_basics.md`
primer. Adds ~30 chunks to `kb/chroma`. Takes ~30 s on first run because
embeddings download.

To extend the KB without re-ingesting duplicates:

```bash
python run.py --ingest data/papers/ --skip-existing
```

To tune ingestion behavior for larger documents:

```bash
python run.py --ingest data/papers/ --chunk-tokens 300 --overlap-tokens 40
```

### Terminal

Run tasks via command line:

```bash
python run.py "Compute the mean and standard deviation of SPY daily returns from 2022-01-01 to 2023-12-31 using yfinance, and report a 1000-resample bootstrap 95% CI on the mean."
```

Expected runtime on an X13 with RTX 4060 mobile: 60–180 s end to end. The console prints the task classification, iteration count, critic verdict, and artifact summary. Full runs are saved to `output/runs/run_NNNN.json`.

Additional CLI modes:

```bash
# Full staged research pipeline (literature + hypothesis + experiment + KG)
python run.py --research "Low-volatility anomaly in equity markets"

# Autonomous iterative research loop (no human input required mid-loop)
python run.py --auto-research "GARCH volatility clustering" --iterations 3

# Inspect the knowledge graph
python run.py --kg-summary
```

### Web App

Start the local web server for a graphical interface:

```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Then open http://127.0.0.1:8000 in your browser.

---

## 4. Web interface and API

The web interface provides a local dashboard at http://127.0.0.1:8000 for interactive use.

### Starting the server

```bash
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

### How the async task flow works

When you submit a task, the server queues it as a background job and returns
`{status: "queued", task_id: "..."}` immediately. The browser opens a
Server-Sent Events stream to `/api/tasks/{id}/stream` and shows an animated
progress bar. When the worker completes, the stream emits a `completed` event
and the result card renders automatically.

Requires `sse-starlette`: `pip install sse-starlette>=1.8.2`.

### Features

- **Run tasks**: Enter natural-language tasks and press Enter. A live progress
  bar tracks the background job; results render automatically when done.
- **Source approval panel**: After a research task, discovered papers from
  arXiv, OpenAlex, and Semantic Scholar appear as checkboxes. Select which ones
  to add to your KB and click "Add selected to KB".
- **Ingest documents**: Specify a file or folder path to add content to the KB.
- **Multifidelity KAN demo**: Run a built-in demonstration of the KAN model.
- **Saved conversations**: Click any task in the sidebar to reload it.

### API reference

**Async task execution** (return `{status: "queued", task_id}` immediately)
- `GET /health` — LLM + RAG health check.
- `POST /run-task` — Queue a standard agent task (`{"task": "..."}`).
- `POST /research-task` — Queue full staged research pipeline (`{"task": "...", "n_papers": 8, "kg_enabled": true}`).
- `POST /api/research/autonomous` — Queue autonomous iterative loop (`{"topic": "...", "n_iterations": 3, "n_papers_per_iter": 6}`).
- `POST /ingest` — Ingest a file or directory (`{"path": "..."}`).
- `GET /kan-demo` — Run the built-in KAN demo (synchronous).

**Progress streaming**
- `GET /api/tasks/{task_id}/stream` — SSE stream emitting `progress`, `completed`, or `failed` events for a background task.

**Task management**
- `GET /api/tasks` — List all tasks (`?limit=50&offset=0&sort_by=-updated_at`).
- `GET /api/tasks/{task_id}` — Get full task with all messages and artifacts.
- `GET /api/tasks/search?q=...` — Keyword search across task titles and tags.
- `POST /api/tasks/{task_id}/messages` — Send a follow-up message (`{"content": "...", "iteration": 0}`).
- `POST /api/tasks/{task_id}/branch` — Branch from an iteration (`{"branch_name": "...", "from_iteration": 1}`).
- `POST /api/tasks/{task_id}/artifacts/{artifact_id}/re-run` — Re-execute artifact code after editing (`{"code": "..."}`).
- `POST /api/tasks/{task_id}/template` — Export a task as a reusable template.

**Reports and knowledge**
- `GET /api/tasks/{task_id}/research-report` — Serve auto-generated markdown research report.
- `GET /api/reports/{task_id}/pdf` — Download compiled LaTeX PDF.
- `POST /api/kb/approve` — Ingest selected discovered sources (`{"task_id": "...", "source_ids": [...]}`).
- `GET /api/kg/summary` — Knowledge graph summary.
- `GET /api/literature/registry` — Literature acquisition registry stats.

**Legacy**
- `GET /runs` — List legacy CLI run files in `output/runs/`.
- `GET /runs/{filename}` — Retrieve a legacy run by filename.

### Multi-turn conversations and branching

Each task is a persistent chatblock. After a task runs, you can continue
refining it without starting over.

**Follow-up messages** are classified automatically:
- **Question** — answered from RAG context, no re-execution.
- **Refinement** — re-runs the role with the prior artifact as context; returns a new artifact.
- **New iteration** — fetches fresh papers, adds them to the KB, then re-executes the full pipeline.

```bash
curl -X POST http://127.0.0.1:8000/api/tasks/<task_id>/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Refine the signal to use EMA instead of SMA"}'
```

**Branching** from any iteration to try a different approach while preserving the original:

```bash
curl -X POST http://127.0.0.1:8000/api/tasks/<task_id>/branch \
  -H "Content-Type: application/json" \
  -d '{"branch_name": "ema-variant", "from_iteration": 1}'
```

---

## 5. How tasks work

### The pipeline

```
USER TASK  (or follow-up message via /api/tasks/{id}/messages)
   │
   ▼
PLAN  ── classify into: data_science | trading_research | writing | mixed
   │
   ▼
RECALL  ── retrieve prior findings (kind=finding) from semantic memory
   │
   ▼
MULTI-SOURCE SEARCH  ── arXiv + OpenAlex + Semantic Scholar
   │                     1-hop citation traversal, user approval panel
   ▼
RAG retrieve  ── hybrid (BM25 + dense + LLM query expansion)
   │            auto-expands KB if retrieval quality < threshold
   ▼
EXECUTE role appropriate to task type:
   • data_science      → DS prompt → code (Python/R) → sandbox.run_py
   • trading_research  → QUANT prompt → StrategySpec JSON → backtest
   • writing           → outline → per-section drafts → LaTeX assemble
   │
   ▼
CRITIC  ── grounded? numerics_ok? code_runs?
   │
   ├── accept = True  → save task → ingest artifact into KB
   │                 → store findings in semantic memory
   │                 → done
   └── accept = False → revise (max_iter times)
```

For the staged **research pipeline** (`--research` / `POST /research-task`):
- Additional `LITERATURE_ANALYST` → `HYPOTHESIS_FORMER` → `NARRATOR` chain
- Citation DAG expanded (2-hop BFS via OpenAlex)
- Structured markdown research report auto-generated
- Knowledge graph updated after acceptance

For the **autonomous research loop** (`--auto-research` / `POST /api/research/autonomous`):

```
for N iterations:
  RETRIEVE from current KB (LLM query expansion)
  GAP-DETECT ── LLM identifies uncovered subtopics
  SEARCH ── multi-source search on topic + gap queries
  EXPAND ── ingest new papers, BFS-expand citation DAG
  check convergence (stop if retrieval quality delta < 0.05)
REPORT ── generate structured research report
STORE ── embed findings in semantic memory for future tasks
```

### Anatomy of a trading task

The QUANT role returns this:

```json
{
  "name": "sma_50_200_crossover",
  "universe": ["SPY", "QQQ", "IWM"],
  "frequency": "daily",
  "lookback_days": 2520,
  "signal": "Long when 50d SMA > 200d SMA, flat otherwise",
  "signal_code": "(df['close'].rolling(50).mean() > df['close'].rolling(200).mean()).astype(float)",
  "position_sizing": "equal_weight",
  "rebalance_days": 5,
  "stop_loss_pct": null,
  "take_profit_pct": null
}
```

`signal_code` must be a **single Python expression** over a DataFrame
called `df` with OHLCV columns. The compiler in `tools/backtest.py`
strips builtins, so `import` and friends won't work — this is
intentional.

The backtest then:

1. Fetches OHLCV via yfinance for the full lookback.
2. Compiles `signal_code` into a callable.
3. Runs the signal on each symbol, executes on the **next bar's** close
   (no look-ahead), applies fees+slippage (default 1+2 bps one-way).
4. Equal-weights the per-symbol return streams.
5. Computes Sharpe, Sortino, Deflated Sharpe (with N=1 trial — you'd
   pass higher N if you swept hyperparameters yourself), MDD, CAGR,
   win rate, 99 % VaR.

### Anatomy of a writing task

1. `WRITE` produces an `Outline` — title, abstract, list of sections
   each with key points and target word count.
2. For each section: a fresh RAG query on the section title + key
   points, then `WRITE` drafts the section in LaTeX.
3. `tex.assemble` wraps it with a preamble and `\bibliography{refs}`.
4. **Citation validator** (`tex.validate_citations`) extracts every
   `\cite{key}`, looks up the key in your `data/papers/refs.bib`, and
   replaces invalid keys with `[CITATION NEEDED]`.
5. If `tectonic` or `pdflatex` is available, compile to PDF.

Output lands in `output/paper/main.tex` and `output/paper/main.pdf`.

---

## 6. Building your knowledge base

The KB lives in `kb/chroma` (Chroma SQLite) plus `kb/main_bm25.pkl` (BM25
index cache). Both are gitignored.

### Adding files manually

```bash
# Single file
python run.py --ingest path/to/paper.pdf
python run.py --ingest path/to/notes.md
python run.py --ingest path/to/refs.bib

# Whole folder (recursive)
python run.py --ingest ~/Documents/finance-papers/
```

Supported: `.pdf`, `.md`, `.txt`, `.tex`, `.bib`. Other extensions are
silently skipped.

### Multi-source paper discovery

For research tasks, the system searches **three sources simultaneously**:

| Source | Coverage | API key? |
|---|---|---|
| **arXiv** | Preprints, CS/Math/Finance | None |
| **OpenAlex** | 250 M academic works, all fields | None |
| **Semantic Scholar** | CS, finance, ML with citation counts | None |

After search, a **1-hop citation traversal** fetches references and
citing papers via OpenAlex, scored by citation count. A persistent
**citation DAG** (NetworkX DiGraph, saved to `output/citation_dag.json`)
accumulates across sessions. PageRank on the DAG surfaces the most
influential papers.

All discovered papers appear as a **checkbox approval panel** in the web
UI so you decide exactly what enters the KB. Papers not yet approved are
saved as `pending_sources.json` in the task directory and can be approved
later via `POST /api/kb/approve`.

### Failure-driven KB auto-expansion

If retrieval quality (mean hybrid score) drops below the configured
`kb_expansion_threshold` (default 0.25), the system automatically:
1. Runs a multi-source search on the current query.
2. Ingests hop=0 papers immediately.
3. Saves remaining papers to `pending_sources.json` for review.
4. Re-retrieves with the expanded KB.

This happens silently in the background; a log entry `[KB_EXPAND] Low
retrieval quality (score=X), triggered expansion → N papers ingested`
records the event.

### Cross-task semantic memory

When a research artifact is accepted, its key findings are embedded and
stored in Chroma with `metadata={"kind": "finding", "task_id": "..."}`.
On subsequent tasks (even in a new session), these findings are retrieved
by dense similarity and prepended to the RAG context as "Prior Finding:"
entries. This means the system gets smarter across sessions without any
manual curation.

### Conversation memory compression

Long task threads (>20 messages by default) are automatically compressed.
Older messages are summarised by the LLM into a `MemorySummary` (saved to
`output/tasks/{id}/memory_summary.json`) and the summary is prepended as
a context document. The 8 most recent messages remain verbatim. This
prevents context-window blowup on long multi-turn sessions.

Thresholds are configurable in `configs/config.yaml`:
```yaml
memory:
  compression_threshold: 20   # messages before compression
  keep_last: 8                # recent messages kept verbatim
  kb_expansion_threshold: 0.25
```

### What gets indexed

- **PDF / MD / TXT / TEX**: split into ~256-token chunks with 32-token
  overlap, embedded with BGE-small-en-v1.5 (384-dim).
- **BibTeX**: each entry becomes one chunk tagged
  `meta={"kind":"bib","key":"..."}`. The writer's citation validator
  reads these to know which keys are valid.
- **Findings** (`kind=finding`): auto-ingested from accepted artifacts;
  retrieved by dense search across tasks.

### Inspecting the KB

```python
from rag.hybrid import LiteHybridRAG
rag = LiteHybridRAG()
print(f"{len(rag)} chunks")
for r in rag.retrieve("momentum factor", k=3):
    print(r["score"], r["id"], r["text"][:80])

# See only prior research findings
for r in rag.retrieve("volatility", k=5, metadata_filters={"kind": "finding"}):
    print(r["meta"].get("task_id"), r["text"][:80])
```

### Resetting

```python
from rag.hybrid import LiteHybridRAG
LiteHybridRAG().reset()
```

Or just `rm -rf kb/chroma kb/*.pkl`.

---

## 7. Configuration reference

Edit `configs/config.yaml`. Key knobs:

### LLM section

```yaml
llm:
  model: "qwen2.5:7b-instruct-q4_K_M"
  num_ctx: 8192          # raise to 16384 if you have 32 GB RAM
  temperature: 0.2       # planner / DS / quant / critic
  temperature_creative: 0.6   # writer (currently used in roles.draft_section)
```

**Choosing a model.** All must be Ollama-pullable.

| VRAM | Model tag | Notes |
|---|---|---|
| 8 GB+ | `qwen2.5:7b-instruct-q4_K_M` (default) | Best balance for stats/code |
| 8 GB+ | `qwen2.5:14b-instruct-q4_K_M` | Better, slower (~9 GB VRAM) |
| 4-6 GB | `qwen2.5:3b-instruct-q5_K_M` | Falls over on hard math |
| any   | `deepseek-r1:7b` | Adds visible `<think>` tokens; verbose |
| any   | `phi3.5:3.8b` | Lighter alternative |

After changing the model, pull it: `ollama pull <tag>`.

### RAG section

```yaml
rag:
  alpha_dense: 0.6     # weight on dense vs BM25 in fusion
  top_k: 6             # number of chunks fed to the LLM
  top_m: 30            # candidates considered before fusion
  token_budget: 3500   # context budget for retrieved docs
  query_expansion:
    enabled: true
    method: "llm_then_local"   # LLM rewrite first, rule-based fallback
    max_expansions: 3          # query variants generated
```

If your corpus is mostly numeric/technical, lower `alpha_dense` to ~0.4
(BM25 wins on rare token matches). Disable query expansion (`enabled: false`)
to save ~100 ms per retrieval on slow hardware.

### Trading section

```yaml
trading:
  initial_equity: 100000.0
  max_leverage: 1.0           # enforced by risk_gate
  max_position_pct: 0.20      # |w_i| <= 0.20
  max_turnover_per_rebalance: 0.50
  var_confidence: 0.99
  var_limit_pct: 0.05         # daily 99% VaR <= 5% equity
  kelly_fraction: 0.33        # haircut on f*
```

These are hard caps. The QUANT role is told about them in its system
prompt; the `risk_gate` function enforces them programmatically before
any (paper) order would be sent.

### Agent section

```yaml
agent:
  max_iterations: 2          # critic revisions
  sandbox_timeout_s: 60      # max wall time for one code run
  sandbox_mem_mb: 2048       # rlimit on Linux; ignored on Windows
```

### Research and autonomous loop sections

```yaml
research:
  n_papers: 8                # papers fetched per research pipeline run
  kg_enabled: true           # update knowledge graph after acceptance
  citation_dag_hops: 2       # BFS depth for citation traversal

autonomous_loop:
  n_iterations: 3            # max loop iterations
  n_papers_per_iter: 6       # papers per search query per iteration
  quality_threshold: 0.40    # convergence threshold (retrieval quality)

memory:
  compression_threshold: 20  # messages before rolling compression
  keep_last: 8               # recent messages kept verbatim
  kb_expansion_threshold: 0.25  # retrieval quality trigger for auto-expansion
```

---

## 8. Mathematical framework

This section documents the formulas the agents use so you can verify
their outputs.

### 7.1 Hybrid retrieval

For a query $q$ and document $d$:

$$
s_{\text{dense}}(q,d) = \frac{\phi(q)^\top \phi(d)}{\|\phi(q)\|\,\|\phi(d)\|}, \qquad
s_{\text{bm25}}(q,d) = \sum_{t\in q} \text{IDF}(t)\,\frac{f(t,d)(k_1+1)}{f(t,d)+k_1\!\left(1-b+b\frac{|d|}{\bar L}\right)}
$$

After min-max normalising BM25 scores to $[0,1]$, fuse:

$$
s(q,d) = \alpha\, s_{\text{dense}}(q,d) + (1-\alpha)\, \tilde{s}_{\text{bm25}}(q,d)
$$

with $\alpha = 0.6$ default. Top-$k$ docs are then packed under a token budget by
greedy density order $s_i / \ell_i$.

### 7.2 Performance metrics (annualised, $P=252$)

$$
\text{SR} = \sqrt{P}\,\frac{\bar r - r_f/P}{s_r}, \qquad
\text{Sortino} = \sqrt{P}\,\frac{\bar r - r_f/P}{s_{r^-}}
$$

$$
\text{MDD} = \max_t\!\left(1 - \frac{E_t}{\max_{s\le t} E_s}\right), \qquad
\text{CAGR} = \left(\frac{E_T}{E_0}\right)^{P/T} - 1
$$

### 7.3 Deflated Sharpe ratio (Bailey & López de Prado, 2014)

For an observed Sharpe $\hat{\text{SR}}$ from $T$ observations across $N$
trials:

$$
\text{DSR} = \Phi\!\left(
  \frac{\big(\hat{\text{SR}} - \mathbb{E}[\max_i \text{SR}_i]\big)\sqrt{T-1}}
       {\sqrt{1 - \hat\gamma_3 \hat{\text{SR}} + \frac{\hat\gamma_4 - 1}{4}\hat{\text{SR}}^2}}
\right)
$$

with $\hat\gamma_3$ skew, $\hat\gamma_4$ raw kurtosis of returns, and

$$
\mathbb{E}[\max_i \text{SR}_i] \approx (1-\gamma)\,\Phi^{-1}\!\left(1-\tfrac{1}{N}\right) + \gamma\,\Phi^{-1}\!\left(1-\tfrac{1}{Ne}\right),\quad \gamma\approx 0.5772
$$

Implemented in `tools/risk.py::deflated_sharpe`.

### 7.4 Position sizing

Kelly:
$$
f^{*} = \frac{\hat\mu - r_f}{\hat\sigma^2}, \qquad f_{\text{use}} = \lambda f^{*},\ \lambda\in[0.25, 0.5]
$$

### 7.5 Mean-variance with L2 turnover

$$
\max_w\ w^\top\hat\mu - \tfrac{\gamma}{2}\,w^\top\hat\Sigma w - \tfrac{\tau}{2}\|w - w_{\text{prev}}\|_2^2
\quad\text{s.t.}\quad \mathbf{1}^\top w = 1,\ |w_i|\le c,\ \|w\|_1 \le L
$$

Solved with `cvxpy`+ECOS. Closed-form fallback when no `cvxpy`:

$$
w^{*} = (\gamma\hat\Sigma + \tau I)^{-1}(\hat\mu + \tau w_{\text{prev}})
$$

### 7.6 Risk gate

A candidate weight vector $w$ is approved iff:

$$
\max_i |w_i| \le c_{\max} \ \land\ \|w\|_1 \le L_{\max} \ \land\ \|w - w_{\text{prev}}\|_1 \le T_{\max} \ \land\ \widehat{\text{VaR}}_\alpha(w) \le V_{\max}
$$

where $\widehat{\text{VaR}}_\alpha$ is historical simulation on trailing
returns.

---

## 9. Troubleshooting

### `httpx.ConnectError` to `http://localhost:11434`

Ollama isn't running.

- Linux: `ollama serve &`
- Windows: open Ollama app from Start menu
- Then `python run.py --healthcheck`

### "model not found" when calling chat

Pull the model:

```bash
ollama pull qwen2.5:7b-instruct-q4_K_M
```

If `ollama pull` itself fails: check disk space and that
`https://ollama.com` isn't blocked by a corporate firewall.

### Out-of-memory / system locks up during `ollama pull` or first inference

Your model is too big for available VRAM+RAM.

1. Stop everything: `Ctrl-C`, close other apps.
2. Edit `configs/config.yaml` → set `model: "qwen2.5:3b-instruct-q5_K_M"`.
3. `ollama pull qwen2.5:3b-instruct-q5_K_M`
4. Retry.

### Sandbox `TIMEOUT after 60s`

The DS agent wrote slow code (e.g. an unintended O(n²) loop). Either:

- raise `agent.sandbox_timeout_s` in `configs/config.yaml`, or
- re-prompt with: "previous attempt timed out. Use vectorised pandas
  ops, no python-level loops over rows."

### `chromadb` errors about SQLite version

```
RuntimeError: Your system has an unsupported version of sqlite3
```

Workaround:

```bash
pip install pysqlite3-binary
```

Then add at the top of `rag/hybrid.py`:

```python
__import__("pysqlite3")
import sys
sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
```

### yfinance returns empty DataFrames

Yahoo throttles IPs that hammer it. Wait a few minutes, or switch the
universe to fewer / different tickers.

### LaTeX builds fail silently

`run.py` returns `pdf: None` but `tex` is fine. Either:
- Install `tectonic` (see §2.4), or
- Manually compile: `cd output/paper && tectonic main.tex`

### Critic always rejects

Two common causes:

1. **The model is too small for the task.** Move from 3B → 7B → 14B.
2. **The critic is too strict.** Edit the `CRITIC` prompt in
   `agents/roles.py` — but be careful, the strictness is the point.

You can also disable the critic loop entirely with `--max-iter 1`.

### Progress bar never completes / `/stream` returns 503

`sse-starlette` is not installed:

```bash
pip install sse-starlette>=1.8.2
```

After installing, restart the server. All other endpoints work without it.

### Multi-source search returns 0 papers

- OpenAlex and Semantic Scholar require internet access.
- Check the log (`output/agent.log`) for `[SOURCE] ... search failed:` lines.
- arXiv alone is always the fallback; tasks continue even if OpenAlex/S2 are unreachable.

### Autonomous loop exits after iteration 1

Retrieval quality improved by less than 0.05 — the loop converged. This
is correct behaviour if the KB was already well-populated for the topic.
Reduce `autonomous_loop.quality_threshold` in `config.yaml` to force more
iterations.

### Tests fail on first install

```bash
pytest -m "not slow"     # skip the RAG retrieval test that downloads BGE
pytest tests/test_risk.py tests/test_backtest.py    # math-only, no models
```

If `test_backtest.py::test_compile_signal_blocks_imports` fails: that's
fine, the test is checking that an import statement raises in the
sandboxed eval; the exact exception type may vary by Python version.

---

## 10. Extending the system

### Adding a new agent role

1. Add a Pydantic schema to `agents/schemas.py`.
2. Add a system prompt to `SYSTEM_PROMPTS` in `agents/roles.py`.
3. Add a typed helper function (e.g. `analyze_<domain>(llm, ...)`).
4. Add a branch in `agents/graph.py::run()` for the new `task_type`.
5. Update `Plan.task_type` in `schemas.py` and the `PLAN` system prompt
   to allow the new type.

### Swapping the embedding model

Edit `configs/config.yaml`:

```yaml
rag:
  embedding_model: "BAAI/bge-base-en-v1.5"   # 440 MB, better recall
  # or: "intfloat/multilingual-e5-base"       # for non-English
```

Then **delete and rebuild the KB** (embeddings are model-specific):

```bash
rm -rf kb/chroma kb/*.pkl
python run.py --ingest data/papers/
```

### Wiring up real (paper) trading

The Alpaca paper API is free.

```bash
pip install alpaca-py
export ALPACA_API_KEY=...
export ALPACA_SECRET_KEY=...
```

Add a `tools/broker.py` that implements `submit_orders(target_weights)`,
calling `risk_gate(...)` first. Wire it into `run.py::make_tools` as
`tools["execute"]` and add a new `task_type="trading_execute"` branch
to `agents/graph.py`. Always require an `ALLOW_LIVE=1` env var before
sending orders to a real account; the agent must never set it itself.

### Adding LangGraph

The current state machine is ~80 lines. If you want LangGraph's
checkpointing, branch parallelism, or streaming:

```bash
pip install langgraph
```

Then port `agents/graph.py::run` to `StateGraph` nodes. The `RunState`
dataclass already maps cleanly onto a `TypedDict`.

---

## Appendix — One-page cheatsheet

```bash
# Setup (once)
bash setup.sh                                  # or setup.ps1 on Windows
pip install sse-starlette>=1.8.2               # SSE streaming support

# Activate venv (every shell)
source .venv/bin/activate                      # or .venv\Scripts\Activate.ps1

# Sanity check
python run.py --healthcheck

# Build / extend KB
python run.py --ingest data/papers/
python run.py --ingest ~/path/to/your/papers/

# Standard tasks
python run.py "Compute volatility of SPY 2020-2024"
python run.py "Backtest 50/200 SMA crossover on SPY since 2015"
python run.py "Write a 4-page report on the momentum anomaly"
python run.py --kan-demo

# Staged research (literature + hypothesis + experiment + KG)
python run.py --research "Low-volatility anomaly in equity markets"
python run.py --research "Momentum crashes" --n-papers 12

# Autonomous iterative research loop
python run.py --auto-research "GARCH volatility clustering" --iterations 3
python run.py --auto-research "Factor investing emerging markets" --n-papers 8

# Inspect KG and logs
python run.py --kg-summary
tail -f output/agent.log

# Run web interface
python -m uvicorn server:app --host 127.0.0.1 --port 8000
# Then open http://127.0.0.1:8000

# Tests
pytest tests/test_risk.py tests/test_backtest.py    # fast, no models
pytest                                              # full suite

# Inspect a saved run
cat output/runs/run_0000.json | python -m json.tool | less

# Read a research report
cat output/tasks/<task_id>/research_report.md
```
