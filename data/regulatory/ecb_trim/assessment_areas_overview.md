# ECB TRIM-style model risk assessment areas (representative structure)

> **PLACEHOLDER / NOT VERBATIM.** This is a hand-authored summary of the
> general assessment areas publicly associated with the ECB's Targeted
> Review of Internal Models (TRIM) exercise and ongoing internal model
> supervision practice. It is not a reproduction of the ECB's actual TRIM
> guide or methodology documents and must not be cited as such. Replace
> with the real ECB guidance before relying on it for anything beyond
> structural drafting scaffolding.

## Representative assessment areas

Model risk validation informed by this style of supervisory review
commonly covers:

1. **Conceptual soundness review** -- is the model design, including risk
   driver selection and functional form, justified and documented?
2. **Data quality assessment** -- is the data used for development,
   calibration, and ongoing application fit for purpose?
3. **Outcomes analysis** -- do realised outcomes (defaults, losses, P&L)
   remain consistent with model predictions over time?
4. **Benchmarking** -- how does the model perform against independent
   challenger models or industry benchmarks?
5. **Sensitivity analysis** -- how do model outputs respond to
   perturbations in key inputs and assumptions?
6. **Stability testing** -- Population Stability Index (PSI), Gini, and
   Kolmogorov-Smirnov statistics tracked over time to detect degradation.
7. **Implementation testing** -- does the production/IT implementation
   match the approved model specification exactly?
8. **Ongoing monitoring review** -- is there an adequate monitoring
   framework with escalation triggers between full revalidation cycles?

## Illustrative model tiering and revalidation cadence

Models are commonly tiered by materiality (e.g. Tier 1 high materiality,
Tier 2 medium, Tier 3 low), with more frequent full revalidation required
for higher tiers. `configs/config.yaml`'s `risk_validation.model_risk`
block encodes illustrative revalidation-cadence defaults
(`tier_1_revalidation_days`, etc.) that must be replaced with the bank's
actual model risk management policy.

## Illustrative overall rating scale

Overall model risk is commonly summarised on a scale such as low / medium
/ high / unacceptable, distinct from the compliant/non-compliant scale
used for individual credit or non-credit risk validation findings.
