# User manual

See [README.md](README.md) for installation, OpenRouter configuration, the
multi-agent architecture, CLI commands, API examples, IBKR paper setup, and
safety controls.

## Normal operating sequence

1. Create a free OpenRouter API key.
2. Export `OPENROUTER_API_KEY`.
3. Run `python run.py --healthcheck`.
4. Ingest research material with `python run.py --ingest data/papers/`.
5. Run `python run.py --quant-team "..."` and inspect the research, backtest,
   risk decision, and order preview.
6. Configure IBKR paper TWS only when the research-only path is healthy.
7. Run with `--submit-paper`; review the displayed order; type the exact
   confirmation phrase.
8. Keep `IBKR_TRANSMIT=0` until held orders are visible and correct in TWS.

## What is local

Only application code, Chroma vector persistence, BM25 text indexes, task
history, backtests, and deterministic risk checks run locally. LLM and
embedding inference run through OpenRouter-hosted providers.

## Credentials

Never put keys or account IDs in the repository. Supported environment values:

```text
OPENROUTER_API_KEY
OPENROUTER_BASE_URL
OPENROUTER_MODEL
OPENROUTER_EMBEDDING_MODEL
ALLOW_PAID_INFERENCE
TRADER_API_TOKEN
IBKR_ENABLED
IBKR_ACCOUNT
IBKR_HOST
IBKR_PORT
IBKR_CLIENT_ID
IBKR_PAPER_ONLY
IBKR_TRANSMIT
ALLOW_LIVE_TRADING
```

The OpenRouter key authorizes hosted inference. The default model routes are
free. A paid model such as a Claude route requires both its model ID and
`ALLOW_PAID_INFERENCE=1`, and may use account credits. IBKR credentials remain
inside TWS/IB Gateway and are not handled by this application.

## Broker controls

The application rejects orders when any of these checks fail:

- IBKR adapter disabled
- missing account
- non-paper account while paper-only mode is enabled
- disallowed order type
- unavailable limit price
- per-order notional cap
- daily submitted-notional cap
- backtest Sharpe/Deflated Sharpe/drawdown/VaR gate
- unknown, expired, reused, or order-mismatched approval token

IBKR paper fills are simulations and can differ substantially from live fills.
