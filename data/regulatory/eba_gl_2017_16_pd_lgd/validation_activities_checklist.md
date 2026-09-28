# IRB credit model validation activities checklist (placeholder)

> **PLACEHOLDER / NOT VERBATIM.** See `overview.md` in this folder for the
> disclaimer. This file gives the RAG index a compact, retrievable checklist
> matching the areas used by `agents/risk_schemas.py::
> CREDIT_VALIDATION_AREAS` and the credit-risk validation agent's prompt
> (`agents/risk_roles.py`).

For each credit risk model (PD, LGD, EAD/CCF, rating scorecard, or IFRS 9
ECL model reusing IRB parameters), a validation review assesses:

1. **Conceptual soundness** -- is the modelling approach and choice of risk
   drivers theoretically and empirically justified for the exposure class?
2. **Data quality** -- is the development and application data complete,
   accurate, representative, and of sufficient historical depth (BCBS 239)?
3. **Discriminatory power** -- does the model still separate good/bad risk
   (Gini / AUC, KS) versus its own history and peer benchmarks?
4. **Calibration and back-testing** -- do realised outcomes match
   predictions within tolerance? PSI for population drift; a Jeffreys /
   binomial test of PD per rating grade and at portfolio level; a
   predicted-vs-realised mean test for LGD and CCF.
5. **Margin of Conservatism** -- is there a documented MoC framework
   (categories A/B/C), quantified, attributed, and released only on
   remediation?
6. **Downturn LGD estimation** -- is an identified downturn period reflected
   in the LGD estimate (add-on or reference value), per EBA/GL/2019/03?
7. **Override analysis** -- is the override rate, direction and pattern
   consistent with sound risk management, and are overrides back-tested?
8. **Model change management** -- is any change/extension classified as
   material or non-material per Delegated Regulation (EU) 529/2014, with
   ex-ante approval or notification as required?
9. **IT implementation** -- does the deployed engine match the approved
   specification exactly?
10. **Use test** -- is the model actually used in credit granting, pricing,
    provisioning and risk management?
11. **Ongoing monitoring** -- is there a monitoring plan with trigger levels
    and a revalidation cadence appropriate to the model's materiality?

A finding against any area carries a verdict (compliant / partially
compliant / non-compliant / not applicable), a severity, a grounded
description, and -- where not fully compliant -- a concrete recommendation
with an owner and remediation deadline.
