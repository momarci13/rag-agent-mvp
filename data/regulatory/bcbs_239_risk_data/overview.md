# BCBS 239-style risk data aggregation and reporting (representative structure)

> **PLACEHOLDER / NOT VERBATIM.** Hand-authored summary of the publicly
> known structure of BCBS 239 (Principles for effective risk data
> aggregation and risk reporting). Not a reproduction of the text; replace
> before reliance.

## Principle groups

1. **Governance and infrastructure** -- a data architecture and IT
   infrastructure that fully supports risk data aggregation.
2. **Risk data aggregation capabilities**
   - **Accuracy and integrity** -- data is correct and reconciled to source.
   - **Completeness** -- all material risk data is captured, by legal
     entity, asset type, industry, region.
   - **Timeliness** -- data can be produced within the required timeframe,
     including in stress.
   - **Adaptability** -- ad hoc requests and new regulatory demands can be
     met.
3. **Risk reporting practices** -- accuracy, comprehensiveness, clarity,
   frequency and distribution of risk reports.
4. **Supervisory review, tools and cooperation.**

## Why this matters for validation

Model validation relies on the same data. The deterministic gate raises a
data-quality finding when a credit case file reports completeness or
accuracy below the configured floors, or a historical observation period
shorter than the CRR minimum. Weak data also drives the Margin of
Conservatism (category A).
