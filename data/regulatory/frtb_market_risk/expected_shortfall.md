# FRTB Expected Shortfall -- representative structure

> **PLACEHOLDER / NOT VERBATIM.** Hand-authored summary of the publicly
> known shape of the FRTB Expected Shortfall measure (BCBS d457). Not a
> reproduction of the standard; replace before reliance.

## Measure

FRTB replaces the 99% Value-at-Risk with **Expected Shortfall (ES) at the
97.5% confidence level** as the regulatory market-risk measure. ES is the
average loss in the tail beyond the 97.5% quantile, so it is sensitive to
tail severity in a way VaR is not.

## Liquidity horizons

ES is computed for risk-factor classes scaled to prescribed **liquidity
horizons** (10, 20, 40, 60, 120 days) and aggregated, reflecting that not
all positions can be exited in one day.

## Stressed calibration

The capital measure uses ES calibrated to a **period of significant
financial stress** for the bank's portfolio (stressed ES), scaled by the
ratio of full-set to reduced-set ES. A stressed ES that is not greater than
the current ES is implausible and is flagged by the gate.

## Why this matters for validation

Where a market-risk case file is an FRTB submission (a desk id, PLA
statistics, a 97.5% back-testing series, or a stressed measure is present)
but ES(97.5%) is not reported, the deterministic gate raises a medium
finding citing BCBS d457.
