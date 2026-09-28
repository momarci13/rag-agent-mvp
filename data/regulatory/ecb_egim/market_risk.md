# ECB guide to internal models -- Market risk (representative structure)

> **PLACEHOLDER / NOT VERBATIM.** Hand-authored summary of the *kind* of
> content publicly associated with the market-risk chapter of the ECB guide
> to internal models, read together with the Basel FRTB standard (BCBS
> d457). Not a reproduction of the source texts; replace before reliance.

## Areas commonly covered

1. **Regulatory back-testing** of the VaR model -- exceptions counted over a
   rolling window; the traffic-light (green/amber/red) classification and
   the multiplier plus-factor add-on; escalation on amber/red.
2. **Expected Shortfall (ES)** -- FRTB replaces VaR with ES at the 97.5%
   confidence level, scaled to prescribed liquidity horizons; a stressed ES
   calibrated to a period of significant financial stress.
3. **P&L Attribution (PLA) test** -- per trading desk, comparison of
   hypothetical P&L with risk-theoretical P&L using the Spearman rank
   correlation and the Kolmogorov-Smirnov distance, classified
   green / amber / red. A red desk cannot use the internal model approach;
   an amber desk carries a capital surcharge.
4. **Non-modellable risk factors (NMRF)** and risks-not-in-VaR (RNIV) -- a
   framework identifying, capitalising and monitoring factors excluded from
   the modellable set.
5. **Risk factor coverage and proxies** -- completeness of the risk-factor
   set relative to the desk's positions.
6. **Independent price verification and valuation** -- feeding the P&L used
   in back-testing and PLA.

## Why this matters for validation

Provides retrieval context for the market-risk validation agent and
complements the deterministic gates: dual-level (99% and 97.5%) VaR
back-testing with the Kupiec test, the PLA green/amber/red gate, and the
ES-reported / stressed-ES-plausibility checks.
