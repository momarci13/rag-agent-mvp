# FRTB P&L attribution (PLA) test -- representative structure

> **PLACEHOLDER / NOT VERBATIM.** Hand-authored summary of the publicly
> known shape of the FRTB P&L attribution test (BCBS d457). The exact
> thresholds below are the commonly cited defaults and are configurable in
> `configs/config.yaml` (`risk_validation.non_credit_risk.market`); confirm
> against the standard in force before reliance.

## What it compares

Per trading desk, for each business day:

- **Hypothetical P&L (HPL)** -- revaluation of the previous day's positions
  using end-of-day market data, with no intraday trading, fees or
  commissions.
- **Risk-theoretical P&L (RTPL)** -- the P&L predicted by the desk's risk
  model using only its included risk factors.

## Test statistics and zones

Two statistics over a rolling window:

- **Spearman rank correlation** between HPL and RTPL (higher is better).
- **Kolmogorov-Smirnov distance** between the two empirical distributions
  (lower is better).

| Zone  | Condition                                   | Consequence                       |
| ----- | ------------------------------------------- | --------------------------------- |
| green | Spearman >= 0.80 and KS <= 0.09             | internal model approach retained  |
| amber | Spearman >= 0.70 and KS <= 0.12             | capital surcharge applies         |
| red   | otherwise                                   | desk moves to standardised approach |

## Why this matters for validation

The deterministic gate computes (or reads) the two statistics, classifies
the zone, and raises a **critical** finding for red and a **medium** finding
for amber, citing BCBS d457.
