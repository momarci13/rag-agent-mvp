# Example Tasks

Copy any of these into `python run.py "..."` to try the system.

## 1. Data Science / Statistics

```bash
python run.py "Download SPY daily prices for 2018-2024 using yfinance. Compute the annualized volatility, the empirical 99% historical VaR, and a 1000-resample bootstrap 95% CI on the mean daily return. Print the results."
```

```bash
python run.py "Fit an OLS regression of SPY daily returns on QQQ and TLT daily returns over 2020-2024. Report the coefficients with standard errors, R-squared, and the Durbin-Watson statistic."
```

```bash
python run.py "Use yfinance to fetch AAPL daily returns for 2020-2024. Test for stationarity with the ADF test, fit an ARIMA(1,0,1), and report the AIC."
```

## 2. Trading Research

```bash
python run.py "Design a 50/200-day SMA crossover strategy on SPY, QQQ, and IWM. Backtest it from 2015 onward and report the Sharpe, deflated Sharpe, max drawdown, and CAGR."
```

```bash
python run.py "Design a long-only mean reversion strategy that goes long when the 14-day RSI of a stock drops below 30 and exits when it crosses 50. Test on SPY, QQQ, EFA, EEM."
```

```bash
python run.py "Backtest a momentum strategy: each week, hold each ETF if its 6-month return exceeds zero, otherwise zero weight. Universe: SPY, EFA, EEM, AGG, GLD, VNQ. Report Sharpe and worst drawdown."
```

```bash
python run.py --kan-demo
```

## 3. Academic Writing

```bash
python run.py "Write a 4-page academic-style report on the momentum factor anomaly: history, theoretical explanations, empirical evidence, and known limitations including momentum crashes."
```

```bash
python run.py "Write a methods section explaining fractional Kelly position sizing: derivation, justification for the haircut, and trade-offs vs equal-weight."
```

## 4. Mixed (DS + writing)

```bash
python run.py "Compute the rolling 1-year Sharpe ratio of SPY from 2010 to 2024 and write a short report (about 800 words) discussing the regimes you observe, with a chart."
```

## 5. Staged research pipeline

Use `--research` to trigger the full literature → hypothesis → experiment → knowledge-graph pipeline. Requires internet (arXiv).

```bash
python run.py --research "What is the empirical evidence for low-volatility anomaly in equity markets?"
python run.py --research "Momentum crashes: causes, magnitude, and mitigation strategies" --n-papers 12
python run.py --research "Machine learning for cross-sectional return prediction" --no-kg
```

## 6. Knowledge graph

```bash
python run.py --kg-summary     # print papers, findings, and tasks in the KG
```

## 7. Multi-turn follow-ups (web API)

After a task runs, send follow-up messages via the API. The `task_id` is returned by `/run-task`.

```bash
# Ask a clarifying question (answered from RAG, no re-execution)
curl -X POST http://127.0.0.1:8000/api/tasks/<task_id>/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "What does the Deflated Sharpe ratio measure?"}'

# Request a refinement (re-runs the role with prior code as context)
curl -X POST http://127.0.0.1:8000/api/tasks/<task_id>/messages \
  -H "Content-Type: application/json" \
  -d '{"content": "Change the signal to use EMA(12)/EMA(26) instead of SMA"}'

# Branch from iteration 1 to try a different approach
curl -X POST http://127.0.0.1:8000/api/tasks/<task_id>/branch \
  -H "Content-Type: application/json" \
  -d '{"branch_name": "ema-variant", "from_iteration": 1}'
```

## Tips

- The first run after `ollama serve` is slow (model load, ~30-60 s). Subsequent runs reuse the loaded model.
- If a task fails the critic on iteration 1, the agent retries once. Set `--max-iter 1` to disable revision.
- CLI runs are saved to `output/runs/run_NNNN.json`. Web-server tasks are saved to `output/tasks/<task_id>/` with full conversation history.
- LaTeX output requires `tectonic` or `pdflatex` installed — see the user manual.
- Trading runs require internet (yfinance) on first fetch; subsequent runs cache via yfinance's own cache.
- Accepted task artifacts are automatically ingested into the KB — the knowledge base grows without manual `--ingest`.

## Adding your own knowledge

```bash
# Single PDF (e.g., a paper you downloaded)
python run.py --ingest /path/to/paper.pdf

# A whole folder of references
python run.py --ingest data/papers/

# Add your own bibliography
python run.py --ingest my_refs.bib
```
