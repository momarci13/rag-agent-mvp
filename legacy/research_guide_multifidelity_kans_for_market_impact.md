# Research Guide: Multifidelity KANs for Intraday Market Impact

This document is a step-by-step, publication-ready research guide for developing and validating **multifidelity Kolmogorov–Arnold Networks (KANs)** for intraday market impact modeling, using broker or exchange market data (e.g., IBKR).

---

## 1. Define the Research Contribution

Formulate a single, falsifiable claim.

Example:
> A residual multifidelity KAN achieves comparable or lower impact prediction error than neural-network and tree-based baselines while requiring significantly fewer high-fidelity labels and offering interpretability.

Predefine:
- Assets
- Date ranges
- Regimes (normal vs stress)
- Evaluation metrics

---

## 2. Define the Target Variable (Impact Label)

Choose a precise and defensible definition.

Temporary impact for a meta-order:

- Arrival price: mid-price at start time $t_0$
- Execution price: VWAP of fills over $[t_0, t_1]$
- Impact:

$$
I = \text{sign(side)} \cdot (p_{\text{exec}} - p(t_0))
$$

Optionally define permanent impact at $t_1 + \Delta$.

---

## 3. Define the Multifidelity Hierarchy

### Option A: Model vs Reality (recommended)

- Low fidelity: analytical model (e.g. square-root or Almgren–Chriss)
- High fidelity: replayed or realized execution impact

### Option B: Coarse vs Microstructure

- Low fidelity: top-of-book + bar/tick features
- High fidelity: order-book depth features

---

## 4. Data Acquisition Strategy (IBKR-Oriented)

### 4.1 Trades and Quotes

- Historical tick data via IBKR `reqHistoricalTicks`
- Real-time tick-by-tick streams

### 4.2 Market Depth (Level II)

- Live order book via `reqMktDepth`
- Requires Level II market data subscriptions
- No native historical depth endpoint

### 4.3 Practical Implication

To obtain historical depth:
- Run a collector
- Record depth updates continuously
- Store locally (e.g. Parquet)

---

## 5. Build the Data Collection Pipeline

Minimum requirements:
- TWS or IB Gateway
- Subscription management
- Timestamped storage of:
  - Depth updates
  - Trades and quotes

Store raw events, not snapshots.

---

## 6. Meta-Order Construction

Two approaches:

1. Controlled execution (paper or live trading)
2. Synthetic meta-orders replayed against recorded books

Meta-order parameters:
- Size $Q$
- Duration $T$
- Participation rate
- Side

---

## 7. High-Fidelity Label Generation (Replay)

Implement a replay simulator:
- Market orders consume depth
- Limit orders queue at price levels (simplified allowed)

Compute VWAP and impact for each synthetic meta-order.

This yields high-fidelity labels.

---

## 8. Low-Fidelity Label Generation

Use a standard functional form:

$$
I_L = \alpha \, \sigma \, (Q/V)^\beta
$$

- $V$: intraday volume proxy
- $\sigma$: realized volatility
- $\beta$: fixed or calibrated

Compute for all samples.

---

## 9. Feature Engineering

Recommended features:

- Order parameters: $Q$, duration, participation, side
- Market state: spread, volatility, volume rate
- Liquidity: depth at top $k$ levels, imbalance
- Regime indicators: time-of-day, volatility regime

Avoid excessive feature proliferation.

---

## 10. Multifidelity KAN Model

Residual formulation:

$$
I_H(x) = I_L(x) + \Delta(x, I_L(x))
$$

Where $\Delta$ is a KAN with spline-based edge functions.

Regularize spline curvature to control extrapolation.

---

## 11. Baselines

Mandatory comparisons:

- Low-fidelity analytical model
- XGBoost
- MLP residual model
- (Optional) GAM / splines
- (Optional) Gaussian Process

Match parameter counts where possible.

---

## 12. Training Protocol

- Time-based train/test split
- Regime-based stress tests
- Vary fraction of high-fidelity labels: 1–100%

Avoid leakage across time.

---

## 13. Evaluation Metrics

- RMSE / MAE
- Tail error (95th / 99th percentile)
- Calibration curves
- Monotonicity and stability checks
- Economic error (bps slippage)

---

## 14. Interpretability Analysis

Visualize learned spline functions:

- Impact vs $Q/V$
- Spread and imbalance nonlinearities
- Regime-dependent behavior

Demonstrate where classical models fail.

---

## 15. Robustness and Ablations

- Remove low-fidelity input
- Replace KAN with MLP
- Remove regularization
- Remove depth features
- Vary book depth $k$

---

## 16. Reproducibility and Release

- Public code
- Synthetic data generator
- Document data licensing constraints
- Release trained weights and scripts

---

## 17. Paper Structure

1. Introduction
2. Related Work
3. Data and Labels
4. Methodology
5. Experiments
6. Interpretability
7. Limitations
8. Conclusion

---

## Notes on Data Alternatives

If IBKR depth collection is impractical, consider academic datasets such as LOBSTER for clean order-book reconstruction. The methodology remains unchanged; only the data source differs.

