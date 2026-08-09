# Web setup

The web server uses hosted chat and embedding inference through OpenRouter. It
does not require Ollama, a GPU, or local model files.

## Start

```powershell
./setup.ps1
$env:OPENROUTER_API_KEY = "sk-or-v1-..."
$env:TRADER_API_TOKEN = "use-a-long-random-secret"
./.venv/Scripts/python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

Open <http://127.0.0.1:8000>.

Choose **Quant Team**, enter a research objective and the configured
`TRADER_API_TOKEN`, then select **Run agent team**. The token is used for that
request only and is not persisted by the page. A run only produces research,
risk decisions, and an exact paper-order preview; the separate submit button
requires an exact confirmation phrase and a one-time server approval token.

For Linux/macOS, use `./setup.sh`, `export` the two variables, and run
`.venv/bin/python -m uvicorn ...`.

## Quant team API

```bash
curl -X POST http://127.0.0.1:8000/api/quant-team/run \
  -H "Content-Type: application/json" \
  -H "X-Trader-Token: use-a-long-random-secret" \
  -d '{"task":"Research a conservative SPY momentum strategy"}'
```

An eligible result contains a paper-order preview and a one-time approval
token. Submit that exact run with:

```bash
curl -X POST http://127.0.0.1:8000/api/quant-team/RUN_ID/execute \
  -H "Content-Type: application/json" \
  -H "X-Trader-Token: use-a-long-random-secret" \
  -d '{"approval_token":"TOKEN"}'
```

IBKR remains disabled until `IBKR_ENABLED=1` and a paper `IBKR_ACCOUNT` are
configured. `IBKR_TRANSMIT` defaults to false.

## Troubleshooting

| Symptom | Check |
|---|---|
| `OpenRouter is unavailable` | `OPENROUTER_API_KEY` is set and valid; network access to `openrouter.ai` works |
| Free-model rate limit | Wait for the free quota to reset or select a funded route with `OPENROUTER_MODEL` |
| Claude requested | Claude is not free; add credits, set a current `anthropic/...` model ID, and set `ALLOW_PAID_INFERENCE=1` |
| Embedding failure | Confirm the configured `:free` embedding route is still present in OpenRouter's model catalog |
| IBKR execution disabled | Set `IBKR_ENABLED=1`, paper `IBKR_ACCOUNT`, TWS port/client ID |
| `nextValidId` timeout | Enable TWS API connections and confirm host/port/client ID |
