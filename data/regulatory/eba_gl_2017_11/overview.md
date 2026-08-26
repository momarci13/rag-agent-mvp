# EBA GL 2017/11-style internal model validation -- representative structure

> **PLACEHOLDER / NOT VERBATIM.** This is a hand-authored summary of the
> *kind* of heading structure a bank's internal ratings-based (IRB) model
> validation guideline commonly follows, written from general public
> knowledge of the EBA Guidelines on PD/LGD estimation and the treatment of
> defaulted exposures. It is not a reproduction of the actual EBA/GL/2017/11
> text and must not be cited as such. Replace this file with the real
> guideline text (or a licensed summary you are permitted to ingest) before
> relying on it for anything beyond structural drafting scaffolding.

## 1. Scope and applicable models

Internal model validation guidance of this kind typically applies to
models used to estimate:

- Probability of Default (PD)
- Loss Given Default (LGD)
- Exposure at Default (EAD) / Credit Conversion Factors (CCF)
- IFRS 9 Expected Credit Loss (ECL) models that reuse IRB risk parameters

## 2. Independent validation function

A validation function independent of model development is expected to
perform, at minimum on an annual basis for material models:

1. **Conceptual soundness review** -- methodology, assumptions, and risk
   driver selection are reviewed for theoretical and empirical
   justification.
2. **Data quality assessment** -- completeness, accuracy, and
   representativeness of the data used to build and apply the model.
3. **Discriminatory power assessment** -- e.g. Gini coefficient / AUC,
   Kolmogorov-Smirnov statistic, compared against internal minimum
   thresholds and prior validation cycles.
4. **Calibration and back-testing** -- comparison of predicted vs. realised
   default/loss rates, including a Population Stability Index (PSI) check
   for input population drift, and a binomial or similar test for
   calibration accuracy with a defined exception-rate tolerance.
5. **Override analysis** -- review of the rate and pattern of manual
   overrides to automated ratings/scores, looking for systematic bias.
6. **IT implementation review** -- confirmation the deployed scoring engine
   matches the validated model specification.
7. **Use test** -- evidence the model is genuinely used in credit decision
   and risk management processes, not solely for regulatory reporting.
8. **Ongoing monitoring** -- a defined monitoring plan with escalation
   triggers between full validation cycles.

## 3. Illustrative severity/verdict conventions

A validation finding for each area above is commonly rated on a scale such
as: compliant / partially compliant / non-compliant / not applicable, with
a severity (critical/high/medium/low/observation) reflecting the potential
prudential impact and urgency of remediation.

## 4. Illustrative citation format used by this system

`regulatory_reference` fields produced by the credit risk validation agent
use a placeholder citation style such as `"EBA/GL/2017/11 Title IV, para
NN"` purely to indicate *which structural area* a finding maps to. These
paragraph numbers are illustrative placeholders, not verified citations.
