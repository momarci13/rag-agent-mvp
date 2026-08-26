# MNB (Magyar Nemzeti Bank) circular structure conventions (generic)

> **PLACEHOLDER / NOT VERBATIM.** This is a hand-authored, generic summary
> of the kind of structure Hungarian National Bank (MNB) supervisory
> circulars and recommendations commonly follow, based on general public
> knowledge of how national competent authority guidance is typically
> organised. It is not a reproduction of any specific MNB circular,
> recommendation (ajánlás), or regulation, and must not be cited as such.
> Replace with the real MNB source documents before relying on this for
> anything beyond structural drafting scaffolding. Note: `tools/data.py`'s
> `MNBSource` class is an unrelated, unimplemented market-data (FX/interest
> rate) stub -- this folder is about MNB *supervisory/regulatory*
> conventions, not that market-data integration.

## Representative structure

MNB supervisory guidance documents of this general type commonly include:

1. **Scope and applicability** -- which institution types and portfolios
   the guidance applies to.
2. **Definitions** -- terminology used consistently through the document.
3. **Expectations / requirements** -- the substantive supervisory
   expectations, often numbered and cross-referenced to EU-level
   guidelines (e.g. EBA Guidelines) where the national guidance
   implements or supplements them.
4. **Reporting/notification obligations** -- what institutions must report
   to MNB, in what format, and on what cadence.
5. **Supervisory review approach** -- how MNB assesses compliance,
   including any risk-based or proportionality considerations.

## Why this matters for risk validation

Where MNB-specific considerations apply (e.g. Hungarian portfolio
segments, HUF-denominated exposures, or national supervisory
expectations layered on top of EU-level guidance), a validation finding
should note this distinctly rather than assuming EU-level guidance alone
is sufficient, once real MNB source material has replaced this
placeholder.

## Replacing this placeholder

Ingest the actual current MNB circulars/recommendations relevant to credit
risk, model risk, and non-credit risk validation into this subfolder and
re-run `python -m rag.ingest data/regulatory/ --collection
regulatory-corpus-v1`.
