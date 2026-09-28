# EBA/GL/2019/03-style downturn LGD estimation -- representative structure

> **PLACEHOLDER / NOT VERBATIM.** Hand-authored summary of the *kind* of
> structure the EBA Guidelines for the estimation of LGD appropriate for an
> economic downturn (EBA/GL/2019/03) commonly follow, from general public
> knowledge. Not a reproduction of the actual text; replace before reliance.

## 1. Purpose

Where realised losses on defaulted exposures are materially higher during an
economic downturn than the long-run average, LGD estimates must reflect that
downturn severity rather than a through-the-cycle average.

## 2. Steps commonly followed

1. **Nature of the downturn** -- identify the relevant economic factors for
   the exposure class (e.g. GDP, unemployment, property price indices,
   sector indicators).
2. **Severity and duration** -- determine the downturn period(s) and the
   severity of each identified factor over an appropriate historical span.
3. **Impact on LGD** -- quantify the downturn effect using one of:
   - a **reference value** derived from observed downturn LGDs where data
     permit;
   - an **add-on** to the long-run average LGD where direct observation is
     not possible, with a haircut/uplift rationale;
   - a **model-based** downturn projection.
4. **Documentation** -- record the identified downturn periods, the method
   chosen, and the conservatism applied where data are weak (linking to the
   Margin of Conservatism framework).

## 3. Why this matters for validation

A validation review confirms an identified downturn period exists, the LGD
estimate carries an effective downturn component (not zero), the method is
justified for the data available, and the choice is revisited when new
downturn data emerge. The deterministic gate flags an LGD case file where
`downturn_lgd_applied` is false or the add-on is absent / non-positive.
