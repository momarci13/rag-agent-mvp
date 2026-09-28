# Delegated Regulation (EU) 529/2014-style model-change materiality (representative structure)

> **PLACEHOLDER / NOT VERBATIM.** Hand-authored summary of the publicly
> known structure of Commission Delegated Regulation (EU) No 529/2014 on the
> assessment of the materiality of extensions and changes of the Internal
> Ratings Based Approach and the Advanced Measurement Approach. Not a
> reproduction of the text; replace before reliance.

## Classification

A change or extension to an approved internal model is classified as:

1. **Material -- requires prior competent-authority approval.** Triggered by
   quantitative thresholds (impact on risk-weighted exposure amounts at the
   relevant level) and/or qualitative criteria (change of rating
   methodology, scope of application, default definition, etc.).
2. **Non-material -- requires ex-ante notification** (before implementation)
   where certain thresholds are met.
3. **Non-material -- requires ex-post notification** for the remainder.

## Quantitative thresholds (illustrative)

- A change is material if it produces a change of 1.5% or more in the
  consolidated RWA for the relevant risk category, or a defined percentage
  at range/model level. (Confirm the exact figures against the regulation in
  force.)

## Why this matters for validation

The deterministic gate raises a **critical** finding when a credit case file
reports `model_change_type = "material"` without evidence that prior
competent-authority approval was obtained, since deploying a material change
without approval is a governance breach, not merely a modelling weakness.
