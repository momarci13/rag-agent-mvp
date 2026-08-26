# Regulatory reference corpus (PLACEHOLDER CONTENT)

> **DISCLAIMER:** The files in this directory and its subfolders are
> hand-authored placeholder summaries of publicly known regulatory
> document *structures* and naming conventions (EBA guidelines, COREP/
> FINREP reporting template numbering, ECB TRIM methodology, generic MNB
> circular conventions). They are **NOT verbatim regulatory text**, have
> **not** been validated against the actual EBA/ECB/MNB source PDFs, and
> **must be replaced or augmented with real source documents** before this
> system's outputs are relied upon for any real regulatory-adjacent
> purpose. Treat every "regulatory_reference" citation string produced by
> the risk-validation agents as illustrative, not authoritative.

## Purpose

This corpus is ingested into a dedicated Chroma collection
(`regulatory_rag.collection` in `configs/config.yaml`, default
`regulatory-corpus-v1`) so the bank risk-validation agent team
(`agents/risk_validation_team.py`) has retrieval-augmented context that
*resembles* the structure of real EBA/ECB/MNB validation and reporting
frameworks, without claiming to reproduce them.

## Layout

```
data/regulatory/
  eba_gl_2017_11/            EBA Guidelines on PD/LGD estimation and
                              internal model validation -- representative
                              heading structure and checklist only.
  corep_finrep_templates/    Generic COREP/FINREP template-numbering
                              conventions (illustrative examples only).
  ecb_trim/                  ECB Targeted Review of Internal Models (TRIM)
                              -- representative assessment-area structure.
  mnb_circulars/              Generic Magyar Nemzeti Bank (MNB) circular/
                              template conventions -- representative
                              structure only.
```

## Replacing placeholders with real source documents

Once real regulatory PDFs/text are available, drop them into the matching
subfolder and re-run ingestion -- no code changes are needed:

```bash
python -m rag.ingest data/regulatory/ --collection regulatory-corpus-v1
```

Ingestion is corpus-content-agnostic (see `rag/ingest.py`); it does not
distinguish placeholder content from verified source text, so replacing a
placeholder file's content in place (or adding new files alongside it) is
sufficient. Re-ingest after any change to this corpus.
