# Model Studio: Modeler and Validator agents

Two AI agents for model development and independent validation in a regulated
financial institution. Covers risk models (credit, market, operational,
liquidity, counterparty), pricing, trading strategies, portfolio models and
general AI/ML models.

| Agent | Delivers |
|---|---|
| **Modeler** | A model specification with equations, assumptions and acceptance criteria; a Python package (`model/api.py`, `pipeline.py`); a pytest suite implementing every planned test; a modelling document (DOCX + PDF) whose result tables come from the executed run. |
| **Validator** | A requirement checklist (built-in frameworks plus requirements extracted from attached regulations and concept papers, cited by page or section); replication from a clean copy; re-run of the developer's tests; deterministic regulatory gates; a document and conceptual-soundness review; its own independent test suite; an independent challenger model; a validation report (DOCX + PDF) with a rule-based outcome. |

Two modes:

- **Develop and validate**: brief + data (+ concept papers, regulations). The
  Modeler builds, the Validator challenges, findings go back to the Modeler
  for up to `max_rounds` (default 3), then a human signs off.
- **Validate an existing model**: brief + data + the model's code (zip or .py)
  and documentation. The Validator writes a thin adapter to the standard API
  (the developer's code is not modified), replicates the documented figures,
  runs the developer's tests and its own, and reports.

## Quick start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=...          # billed OpenAI account
export STUDIO_API_TOKEN=...        # required for starting projects, library uploads, sign-off
uvicorn server:app --reload        # http://127.0.0.1:8000
```

PowerShell: `$env:OPENAI_API_KEY="..."; $env:STUDIO_API_TOKEN="..."`.
PDF output needs LibreOffice (`soffice` on PATH); on Windows with Word you can
set `studio.pdf_via: "docx2pdf"` and `pip install docx2pdf`. Without either,
DOCX files are still produced.

Command line:

```bash
python -m studio.cli develop --title "Mortgage PD" --brief brief.txt --data loans.csv \
    --concept-paper concept.pdf --regulation eba_gl_2017_16.pdf --framework eu_banking
python -m studio.cli validate --title "Vendor PD" --brief "Validate the vendor model" \
    --data loans.csv --package vendor_model.zip --package model_doc.pdf --framework eu_ai_act
python -m studio.cli signoff <project_id> --name "Jane Validator" --role "Head of validation"
```

## How a project runs

```
upload ─► library ingest (page/section locators) ─► data profile
   │
   ▼  round n = 1..max_rounds
Modeler:   spec ─► code + tests ─► run pipeline + pytest ─► repair crashes (≤2) ─► modelling document
Validator: requirements ─► replicate ─► re-run developer tests ─► regulatory gates
           ─► independent tests ─► challenger ─► review ─► merge findings ─► outcome ─► report
   │
   ├─ outcome ∈ {rejected, remediation_required} and n < max_rounds ─► Modeler answers every open finding
   ▼
awaiting sign-off ─► human accepts or rejects ─► report regenerated with the sign-off
```

### Package contract

`model/api.py` exposes `load_data(data_dir)`, `fit(data, seed)`,
`predict(model, df)`, `evaluate(model, data)` and `run(data_dir, out_dir, seed)`;
`run` writes `out_dir/metrics.json` (flat numbers) and `out_dir/figures/*.png`.
Every planned test `MT-k` is a pytest function named `test_MTk...`; the
Validator's are `test_VTk...`. Test results are read from JUnit XML, metrics
from `metrics.json`; nothing the LLM says about results is used.

### Decision framework (deterministic)

Let $F$ be the set of open findings with severity
$s(f) \in \{\text{critical}, \text{high}, \text{medium}, \text{low}, \text{observation}\}$.
The outcome is

$$
O(F)=\begin{cases}
\text{rejected} & \exists f\in F: s(f)=\text{critical}\\
\text{remediation required} & \text{else if } \exists f: s(f)=\text{high}\\
\text{approved with conditions} & \text{else if } \exists f: s(f)=\text{medium}\\
\text{approved} & \text{otherwise}
\end{cases}
$$

and another round runs while $O \in \{\text{rejected}, \text{remediation required}\}$
and $n < n_{\max}$. The LLM cannot set, soften or remove rule findings:

| Rule | Condition | Severity |
|---|---|---|
| R01 | pipeline fails from a clean copy | critical |
| R02 | reported metric $m$ not reproduced: $\lvert \hat m - m\rvert > \max(\varepsilon_r \max(\lvert m\rvert,\lvert\hat m\rvert), \varepsilon_a)$ with $\varepsilon_r=10^{-6}$, $\varepsilon_a=10^{-9}$ (1% and $10^{-3}$ for figures quoted in an external document) | high |
| R03 | planned test not implemented | medium |
| R04 | developer test fails (severity from the plan) or errors | plan / high |
| R05 | acceptance criterion $g(\hat m)$ false on the replicated metric | high if primary, else medium |
| R06 | independent test fails | validator's plan |
| R07 | challenger better on the primary metric: $\hat m_c > \hat m_0 + \delta\lvert\hat m_0\rvert$ (higher-is-better; mirrored otherwise), $\delta=0.02$ | medium |
| R08 | developer test suite cannot be collected | high |
| R09 | no assumptions, limitations, monitoring plan, equations or acceptance criteria | medium |
| G-* | CRR/EBA/FRTB gates (Gini, PSI, PD calibration via Jeffreys/binomial, LGD back-testing, Kupiec/traffic light, PLA, data quality, observation period) | from the gate |

Finding IDs are stable across rounds (`F-R05-auc`, `F-G-credit-risk-gini`,
`F-Q-003`), so a finding raised in round 1 is closed in round 2 when its
condition no longer holds. Review findings (`F-Q-*`) are closed only by the
Validator's review of the Modeler's response.

### Citations

Attached PDFs are indexed per page (`p. 12`), DOCX and Markdown per heading.
A requirement or finding that cites a chunk ID not present in the library is
dropped or stripped of that citation, so every citation in a report resolves
to a real passage.

## Frameworks

- **Generic practice** (always): purpose, data, conceptual soundness,
  implementation, reproducibility, out-of-sample performance, sensitivity,
  benchmarking, monitoring, limitations, test coverage.
- **EU banking**: CRR Art. 179, 180, 185, 366; EBA/GL/2017/16; EBA/GL/2019/03;
  BCBS 239; Delegated Regulation 529/2014; MNB expectations.
- **EU AI Act** (Regulation (EU) 2024/1689): Art. 6 and Annex III 5(b)
  classification, Art. 9 to 15.

Built-in library texts under `data/regulatory/` are short summaries, not the
legal texts. Attach official documents for anything you rely on.

## Layout

```
studio/            the two agents and their machinery
  modeler.py       Modeler agent
  validator.py     Validator agent (incl. external-package adapter)
  rules.py         deterministic findings and outcome
  regulatory_gates.py  CRR/EBA/FRTB threshold checks
  execution.py     sandboxed pipeline and pytest runs (no secrets in env, timeouts)
  library.py       shared document library with page/section citations
  documents.py     DOCX/PDF builders
  orchestrator.py  rounds, remediation loop, sign-off
  cli.py           command line
server.py          FastAPI app and API
web/               UI
agents/llm.py      OpenAI client
rag/               hybrid dense + BM25 retrieval
tools/             statistics, risk and backtest helpers usable by generated code
legacy/            superseded code, not used
output/studio/     projects (created at runtime)
output/library/    shared library registry and copies of attached documents
```

## Security notes

Generated code runs locally as your user. The sandbox strips API keys from the
environment and enforces time and (on Linux/macOS) memory limits, but it is
not an isolation boundary. Run the server on a machine or container where
executing model code written by an LLM is acceptable.

## Tests

```bash
python -m pytest            # ~70 s; the end-to-end tests execute real model code with a scripted LLM
```
