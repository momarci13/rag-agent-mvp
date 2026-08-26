# Credit model validation activities checklist (placeholder)

> **PLACEHOLDER / NOT VERBATIM.** See `overview.md` in this folder for the
> disclaimer that applies to this whole subfolder. This file exists to give
> the RAG index a compact, retrievable checklist matching the areas used by
> `agents/risk_schemas.py::CREDIT_VALIDATION_AREAS` and the credit risk
> validation agent's prompt (`agents/risk_roles.py`).

For each credit risk model (PD, LGD, EAD, rating scorecard, or IFRS 9 ECL
model reusing IRB parameters), a validation review commonly assesses:

1. **Conceptual soundness** -- is the modelling approach and choice of risk
   drivers theoretically and empirically justified for the exposure class?
2. **Data quality** -- is the development and application data complete,
   accurate, and representative of the current portfolio?
3. **Discriminatory power** -- does the model still separate good/bad risk
   effectively (Gini coefficient, KS statistic) versus its own history and
   peer benchmarks?
4. **Calibration and back-testing** -- do realised outcomes match
   predictions within tolerance (PSI for population drift, binomial-style
   exception testing for calibration)?
5. **Override analysis** -- is the override rate and pattern consistent
   with sound risk management, or does it suggest the model is being
   routinely second-guessed for a specific reason?
6. **IT implementation** -- does the deployed scoring/estimation engine
   match the documented and approved model specification?
7. **Use test** -- is the model actually used in credit granting, pricing,
   provisioning, and risk management decisions?
8. **Ongoing monitoring** -- is there a monitoring plan with defined
   trigger levels between full validation cycles?

A finding against any area above should carry: a verdict (compliant /
partially compliant / non-compliant / not applicable), a severity, a
grounded description, and -- where not fully compliant -- a concrete
recommendation and, where relevant, an owner and remediation deadline.
