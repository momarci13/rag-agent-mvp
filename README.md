# Quant Research + IBKR RAG Agent Team

A hosted-inference platform for quantitative research, model development,
backtesting, retrieval-augmented analysis, and controlled IBKR paper-order
submission.

The application does **not** run Ollama, a local LLM, or a local embedding
model. Chat and embedding inference use
[OpenRouter](https://openrouter.ai/) through its OpenAI-compatible API. The
default chat route is `openrouter/free`, and the default hosted embedding route
is `nvidia/llama-nemotron-embed-vl-1b-v2:free`; both have zero token price.
Free-route capacity and model choice can change, so this mode is intended for
personal research and prototyping rather than guaranteed production service.

## Agent team

The quant workflow in `agents/quant_team.py` is an explicit sequence of agents
with typed handoffs:

1. **RAG Agent** retrieves grounded research context using hosted embeddings,
   Chroma vector persistence, and BM25.
2. **Quant Research Agent** produces a sourced thesis, counter-evidence,
   assumptions, and data requirements.
3. **Quant Model Agent** turns the research into a typed `StrategySpec` and
   runs the existing event-driven backtester.
4. **Quant Risk Agent** applies deterministic Sharpe, Deflated Sharpe,
   drawdown, and VaR gates. An LLM cannot override these gates.
5. **Trade Proposal Agent** proposes one bounded paper order from the approved
   universe.
6. **IBKR Trader Agent** applies account/order/notional limits and issues a
   short-lived, one-time human approval token before the official TWS API can
   be called.

The older data-science, academic-writing, autonomous-research, knowledge-graph,
and task-conversation workflows remain available and use the same hosted
inference client.

## Safety model

- IBKR is disabled by default.
- Paper-only mode requires an account that normally starts with `DU`.
- Limit orders are the only allowed order type by default.
- Per-order and daily submitted-notional caps are deterministic.
- Research/model gates must pass before an order can be proposed.
- The exact order is bound to an expiring, single-use approval token.
- CLI submission requires a second interactive confirmation after displaying
  the preview.
- `transmit: false` holds the order in TWS by default. Set `IBKR_TRANSMIT=1`
  only after validating the full paper workflow.
- Live accounts also require `IBKR_PAPER_ONLY=0` and
  `ALLOW_LIVE_TRADING=1`; do not enable these until independently reviewed.

## Architecture

```text
user task
   |
   v
RAG Agent -- OpenRouter /embeddings --> free hosted embedding model
   |
   v
Research Agent --> OpenRouter free router /chat/completions
   |
   v
Model Agent --> StrategySpec --> historical data --> backtest
   |
   v
Risk Agent -- deterministic gates -- rejected
   |
   v
Trade Proposal Agent --> typed TradeIntent
   |
   v
IBKR Trader Agent --> deterministic limits --> one-time human approval
   |
   v
TWS / IB Gateway (paper account, transmit disabled by default)
```

## Requirements

- Python 3.11+
- A free OpenRouter account and API key
- For broker submission only: IBKR TWS or IB Gateway with API access enabled
  and a paper account

No GPU is required.

## Setup

### 1. Install dependencies

Windows:

```powershell
./setup.ps1
```

Linux/macOS:

```bash
./setup.sh
```

Or manually:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

### 2. Configure free hosted inference

Create an OpenRouter API key and set it locally. No gateway process is needed.

Windows PowerShell:

```powershell
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
```

Linux/macOS:

```bash
export OPENROUTER_API_KEY="sk-or-v1-..."
```

Secrets are read only from environment variables and are never stored in
`config.yaml`. `OPENROUTER_BASE_URL` is optional; the default is
`https://openrouter.ai/api/v1`.

The free router does not include free Claude inference. To prefer Claude later,
add OpenRouter credits and select a current Anthropic route without changing
code, for example:

```powershell
$env:OPENROUTER_MODEL = "anthropic/claude-sonnet-5"
$env:ALLOW_PAID_INFERENCE = "1"
```

Unset `OPENROUTER_MODEL` and `ALLOW_PAID_INFERENCE` to return to zero-cost
routing. The second variable is an intentional cost gate: the application
rejects non-free chat or embedding routes without it.

Verify the connection:

```bash
python run.py --healthcheck
```

### 3. Build the RAG knowledge base

Embedding inference goes through OpenRouter:

```bash
python run.py --ingest data/papers/
```

The default embedding route is
`nvidia/llama-nemotron-embed-vl-1b-v2:free`. Override it with
`OPENROUTER_EMBEDDING_MODEL`. The default Chroma collection is
`openrouter-free-v1`, which avoids mixing vectors from the earlier backend.

OpenRouter's zero-cost chat tier has a low daily request allowance, and this
multi-agent workflow uses several calls per run. Rate-limit responses are
reported rather than silently switching to a paid model.

## Run research

General research/backtest pipeline:

```bash
python run.py --research "Evaluate cross-asset momentum with walk-forward validation"
```

Explicit quant agent team with order preview:

```bash
python run.py --quant-team "Research a conservative SPY momentum strategy"
```

This command never submits an order. If model/risk gates pass, the result
contains a typed preview and the reasons for approval or rejection.

## Enable IBKR paper submission

1. Log into a paper account in TWS or IB Gateway.
2. Enable API connections in TWS/IB Gateway.
3. Set the paper account and enable the adapter:

```powershell
$env:IBKR_ENABLED = "1"
$env:IBKR_ACCOUNT = "DU1234567"
$env:IBKR_HOST = "127.0.0.1"
$env:IBKR_PORT = "7497"
$env:IBKR_CLIENT_ID = "71"
```

Keep `IBKR_TRANSMIT` unset for the first smoke test. This sends an
untransmitted order to TWS, where it is held for review.

Run with interactive approval:

```bash
python run.py --quant-team --submit-paper "Research a conservative SPY momentum strategy"
```

The CLI displays the exact order and then requires typing
`APPROVE <SYMBOL> <BUY|SELL> <QUANTITY>`. To transmit to the IBKR paper server
after verifying held orders in TWS:

```powershell
$env:IBKR_TRANSMIT = "1"
```

IBKR's official API documentation recommends paper-account validation before
live trading and requires waiting for `nextValidId` before placing orders. The
adapter implements that callback boundary.

## Web API

Start the server:

```bash
export TRADER_API_TOKEN="use-a-long-random-secret"
python -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000`, choose **Quant Team**, enter the research
objective and the same `TRADER_API_TOKEN`, then select **Run agent team**. The
browser displays each agent trace, the evidence-backed research, strategy,
backtest, model-risk decision, and exact paper-order preview. Running the team
never submits an order. Submission requires a separate exact-phrase prompt and
the server-issued, single-use approval token.

Run the agent team:

```http
POST /api/quant-team/run
Content-Type: application/json
X-Trader-Token: use-a-long-random-secret

{"task":"Research a conservative SPY momentum strategy"}
```

If the response contains an approved `execution_risk`, submit the exact
previewed order with its single-use token:

```http
POST /api/quant-team/{run_id}/execute
Content-Type: application/json
X-Trader-Token: use-a-long-random-secret

{"approval_token":"..."}
```

Other existing endpoints include `/run-task`, `/research-task`, `/ingest`,
`/api/research/autonomous`, task history, SSE progress, literature, and
knowledge-graph summaries.

## Configuration

Main settings are in `configs/config.yaml`:

- `llm`: OpenRouter endpoint, free/optional model routes, role routes, timeouts
- `rag`: hosted embedding route, Chroma/BM25 fusion, query expansion
- `trading`: backtest and portfolio limits
- `ibkr`: broker connectivity, paper/transmit gates, notional limits
- `quant_team.model_risk`: minimum Sharpe/Deflated Sharpe and maximum
  drawdown/VaR

Environment variables override secrets and deployment-specific connection
values.

## Tests

```bash
python -m pytest -q
```

The test suite uses HTTP mock transports, deterministic non-model vectors, and
a fake broker. It does not call OpenRouter or IBKR.

The exact count can vary as optional dependency tests are skipped.

## Key files

```text
agents/llm.py             hosted OpenAI-compatible chat client
rag/embeddings.py         hosted OpenAI-compatible embedding client
agents/quant_team.py      typed multi-agent pipeline
agents/quant_factory.py   production dependency composition
broker/ibkr.py            official TWS API adapter + safety boundary
tools/backtest.py         event-driven backtester
tools/risk.py             portfolio statistics and deterministic risk gates
configs/config.yaml       non-secret defaults
server.py                 FastAPI and quant-team endpoints
run.py                    CLI and interactive paper submission
```

This is research software, not investment advice. Backtests and paper fills do
not prove live performance.
