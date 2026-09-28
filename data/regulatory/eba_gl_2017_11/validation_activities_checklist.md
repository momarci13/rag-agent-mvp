# Definition-of-default implementation review checklist (placeholder)

> **PLACEHOLDER / NOT VERBATIM.** See `overview.md` in this folder for the
> disclaimer that applies to this whole subfolder. The IRB *model*
> validation checklist that matches `agents/risk_schemas.py::
> CREDIT_VALIDATION_AREAS` now lives in
> `../eba_gl_2017_16_pd_lgd/validation_activities_checklist.md`.

When a credit risk model or process relies on the definition of default,
a validation review commonly checks:

1. **Days past due** -- correct counting, treatment of technical past due,
   and the 90-day trigger for material obligations.
2. **Materiality threshold** -- absolute and relative components set per the
   applicable RTS and the bank's approved policy.
3. **Unlikeliness-to-pay triggers** -- the full trigger set is implemented
   and evidenced (non-accrued status, specific credit risk adjustment, sale
   at a material credit-related loss, distressed restructuring, bankruptcy).
4. **Distressed restructuring / forbearance** -- diminished-obligation
   threshold applied; forbearance flags feed the default identification.
5. **Level of application** -- obligor level for non-retail; facility level,
   if used, is limited to retail.
6. **Return to non-default** -- minimum probation period enforced; stricter
   treatment after distressed restructuring.
7. **Contagion / pulling effect** -- default propagation across an obligor's
   exposures and connected clients is implemented.
8. **Consistency** -- the same definition is used in risk data, IRB
   estimation and IFRS 9 staging; any change is handled as a model change.

A finding against any item carries a verdict (compliant / partially
compliant / non-compliant / not applicable), a severity, a grounded
description and, where not fully compliant, a concrete recommendation with
an owner and remediation deadline.
