# EBA/GL/2017/16-style PD/LGD estimation -- representative structure

> **PLACEHOLDER / NOT VERBATIM.** Hand-authored summary of the *kind* of
> structure the EBA Guidelines on PD estimation, LGD estimation and the
> treatment of defaulted exposures (EBA/GL/2017/16) commonly follow, from
> general public knowledge. Not a reproduction of the actual text; replace
> with the real guideline (or a licensed summary) before any reliance.

## 1. Scope

Estimation of the risk parameters used in the IRB approach: Probability of
Default (PD), Loss Given Default (LGD), Expected Loss Best Estimate (ELBE)
and the treatment of exposures in default. Downturn LGD estimation is
covered separately in EBA/GL/2019/03 (`../eba_gl_2019_03_downturn_lgd/`).

## 2. Data requirements

- Completeness, accuracy and appropriateness of the reference data set.
- **Representativeness** of the data used for development and calibration
  relative to the current application portfolio (obligor/facility mix,
  lending standards, workout processes, macro conditions).
- Minimum length of the historical observation period; treatment of missing
  or externally sourced data.
- Data governance consistent with BCBS 239 (`../bcbs_239_risk_data/`).

## 3. Model methodology and human judgement

- Risk-driver selection with a documented rationale; ranking vs. grouping
  approaches.
- Use of human judgement and the framework governing **overrides**
  (recording, rate, direction, back-testing of overridden cases).

## 4. Calibration

- PD calibrated to the **long-run average default rate** at rating-grade
  level and at portfolio level.
- Grade-level and portfolio-level back-testing (e.g. Jeffreys test,
  binomial/chi-square style tests) with a defined tolerance and exception
  handling.
- LGD / ELBE / CCF estimation from realised recoveries and drawings, with
  discounting and direct/indirect cost treatment.

## 5. Margin of Conservatism (MoC)

An explicit MoC framework addressing:

- **Category A** -- identified deficiencies in data and methods.
- **Category B** -- relevant changes to underwriting/risk appetite/recovery
  processes and the general estimation error.
- **Category C** -- the general estimation error where not already in A/B.

The MoC is quantified, attributed to categories, added to the final
parameter, and reduced only when the underlying deficiency is remediated.

## 6. Review of estimates

At least annual review of parameters and their appropriateness, including
recalibration triggers, and treatment of the deficiency register.

## 7. Illustrative citation format

`regulatory_reference` strings such as `"EBA/GL/2017/16"` indicate a finding
concerns PD/LGD estimation, calibration, representativeness or the Margin of
Conservatism. Section/paragraph numbers, where shown, are placeholders.
