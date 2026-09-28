# COREP / FINREP template numbering conventions (illustrative)

> **PLACEHOLDER / NOT VERBATIM.** This file describes the general *shape*
> of COREP (Common Reporting) and FINREP (Financial Reporting) template
> numbering as publicly known convention, using a small number of
> illustrative example template names. It is not a reproduction of the
> actual EBA reporting taxonomy, DPM (Data Point Model), or XBRL
> architecture, and the example template codes below must not be treated
> as authoritative or current. Replace with the real EBA ITS (Implementing
> Technical Standards) reporting templates and taxonomy documentation
> before relying on this for anything beyond structural drafting
> scaffolding.

## General shape

- **COREP** templates report prudential/own-funds and risk-weighted
  exposure information (capital adequacy, credit risk, market risk,
  operational risk, leverage ratio, liquidity ratios such as LCR/NSFR).
- **FINREP** templates report financial statement-style information
  (balance sheet, profit and loss, breakdowns of financial assets/
  liabilities, non-performing exposures) for supervisory purposes.
- Templates are conventionally identified by a letter prefix (`C` for
  COREP, `F` for FINREP), a template number, and a version suffix, e.g.
  illustrative examples: `C 01.00` (own funds), `C 07.00` (credit risk
  standardised approach), `F 18.00` (non-performing and forborne
  exposures). These example codes are illustrative only -- confirm current
  codes and structure against the actual EBA reporting framework in force.

## Why this matters for risk validation

When a credit risk or non-credit risk validation finding references a
"template area" (e.g. "NPE classification", "own funds calculation",
"LCR reporting"), it should be understood as pointing at the *kind* of
regulatory return the underlying model/process feeds, not a specific,
verified template cell reference, unless a real template mapping has been
ingested to replace this placeholder.

## Current instrument

Supervisory reporting for EU institutions is set out in the EBA
**Implementing Technical Standards on supervisory reporting**, given legal
force by **Commission Implementing Regulation (EU) 2021/451** (as amended),
with the reporting content defined by the EBA reporting framework's
technical package: the instructions, the validation rules, and the **Data
Point Model (DPM)** / XBRL taxonomy. Template codes, rows and columns
change between framework releases, so any `template_code` / `row` / `column`
citation this system emits is a structural pointer only until the real
technical package has been ingested.

## Replacing this placeholder

Ingest the current EBA reporting framework's technical package
(instructions, validation rules, DPM) into this subfolder and re-run
`python -m rag.ingest data/regulatory/ --collection regulatory-corpus-v1`
before using this system's outputs to inform actual COREP/FINREP
submissions.
