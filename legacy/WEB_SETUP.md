# Web setup

The web server uses hosted chat and embedding inference through the direct
OpenAI API. It does not require Ollama, a GPU, or local model files. Every
call incurs real, billed OpenAI API cost.

## Start

```powershell
./setup.ps1
$env:OPENAI_API_KEY = "sk-..."
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
| `OpenAI is unavailable` | `OPENAI_API_KEY` is set and valid; network access to `api.openai.com` works |
| Rate limit / 429 | Wait and retry, or reduce concurrent runs; check your OpenAI usage tier |
| Wrong model requested | Set `OPENAI_MODEL` to a model ID your account has access to |
| Embedding failure | Confirm the configured embedding model ID is current and your key has access to it |
| IBKR execution disabled | Set `IBKR_ENABLED=1`, paper `IBKR_ACCOUNT`, TWS port/client ID |
| `nextValidId` timeout | Enable TWS API connections and confirm host/port/client ID |
