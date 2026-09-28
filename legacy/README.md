# Legacy code (not used)

Superseded by the two-agent studio in `studio/` (September 2026). The former
quant team, bank risk-validation team, IBKR paper trading, chat and research
pipelines are kept here for reference only. Nothing in the running app imports
this folder and its tests are not run. Parts that were reused live on in:

- `studio/regulatory_gates.py`: the deterministic CRR/EBA/FRTB gates from
  `agents/risk_validation_team.py`, unchanged.
- `agents/risk_schemas.py`, `agents/regulatory_refs.py`: input schemas for those gates.
- `tools/statistical_tests.py`, `tools/risk.py`, `tools/backtest.py`: helpers the
  generated model code may import.

Delete this folder when you no longer need it.
