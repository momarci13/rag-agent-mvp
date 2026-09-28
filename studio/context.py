"""Shared runtime context passed to both agents."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .config import StudioSettings
from .library import Library, Passage
from .llm_io import JsonLLM
from .schemas import Project, TraceEvent
from .storage import ProjectStore

ProgressFn = Callable[[TraceEvent], None]

CODE_CONTRACT = """\
PACKAGE CONTRACT (all paths relative to the package root)
- model/__init__.py (may be empty)
- model/api.py defining exactly these public functions:
    load_data(data_dir: str) -> dict[str, pandas.DataFrame]
    fit(data: dict[str, pandas.DataFrame], seed: int = 42) -> object
    predict(model, df: pandas.DataFrame) -> numpy.ndarray | pandas.Series | pandas.DataFrame
    evaluate(model, data: dict[str, pandas.DataFrame]) -> dict[str, float]
    run(data_dir: str, out_dir: str, seed: int = 42) -> dict[str, float]
  run() loads the data, splits it deterministically (out-of-time where the data has a time
  dimension, otherwise a seeded split) and documents the split, fits, evaluates on held-out
  data, writes out_dir/metrics.json as a flat JSON object of numbers that includes every
  acceptance-criterion metric, saves charts as out_dir/figures/*.png (matplotlib, Agg
  backend), and returns the metrics dict. fit/evaluate must use the same split as run().
- pipeline.py: `python pipeline.py --data-dir DIR --out-dir DIR` calls model.api.run.
- tests/conftest.py with session-scoped fixtures: data_dir (from os.environ["STUDIO_DATA_DIR"]),
  data (load_data), model (fit with seed 42) and metrics (evaluate).
- tests/test_*.py: at least one test function per planned test; every function name starts
  with test_<ID> where <ID> is the plan ID without the dash, e.g. test_MT03_calibration_binomial.
  Assertions carry messages that show the observed value and the threshold.

RULES
- Python 3.10+. Allowed imports: standard library, numpy, pandas, scipy, scikit-learn,
  statsmodels, matplotlib, and the repository helpers tools.statistical_tests, tools.risk
  and tools.backtest.
- No network access. Read only files under data_dir. Write only under out_dir or pytest tmp_path.
- Fix every random seed. The whole pipeline plus tests must finish within 10 minutes on a laptop.
- Never hard-code results, thresholds met or test outcomes. Tests must compute what they check.
"""

CREDIT_GATE_METRICS = """\
REGULATORY GATE INPUTS (deterministic CRR/EBA/FRTB checks read these exact names):
- credit_risk: write to metrics.json when they apply: gini_coefficient, ks_statistic,
  population_stability_index, backtesting_exceptions_count, backtesting_observations_count,
  lgd_predicted_mean, lgd_observed_mean, lgd_observation_count, data_completeness_pct,
  historical_observation_period_years, override_rate_pct. Put model_type (PD, LGD, EAD,
  rating_scorecard or IFRS9_ECL), exposure_class (retail, corporate, institutions, sovereign,
  equity, securitisation, other), portfolio_segment, estimation_approach
  (internal_ratings_based, standardised or hybrid) and jurisdiction into regulatory_profile.
- market_risk: write var_confidence_level, var_horizon_days, var_backtesting_exceptions,
  var_backtesting_observations, expected_shortfall_975, stressed_var, pla_spearman_correlation,
  pla_ks_statistic when they apply; put business_unit, reporting_date and jurisdiction into
  regulatory_profile.
"""


@dataclass
class AgentContext:
    llm: JsonLLM
    library: Library | None
    settings: StudioSettings
    store: ProjectStore
    project: Project
    progress: ProgressFn = field(default=lambda _e: None)
    cfg: dict[str, Any] = field(default_factory=dict)

    @property
    def data_dir(self) -> Path:
        return self.store.data_dir(self.project.project_id)

    def emit(self, agent: str, step: str, status: str, message: str = "") -> None:
        event = TraceEvent(agent=agent, step=step, status=status, message=message[:2000])  # type: ignore[arg-type]
        self.project.trace.append(event)
        self.store.save(self.project)
        self.progress(event)

    def evidence(self, query: str, k: int = 8) -> list[Passage]:
        """Attached documents first, then the wider library filtered to chosen frameworks."""
        if self.library is None:
            return []
        out: list[Passage] = []
        if self.project.attachment_doc_ids:
            out.extend(self.library.retrieve(query, k=k, doc_ids=self.project.attachment_doc_ids))
        seen = {p.chunk_id for p in out}
        for p in self.library.retrieve(query, k=k, frameworks=self.project.input.frameworks):
            if p.chunk_id not in seen:
                out.append(p)
        return out[: k + len(self.project.attachment_doc_ids) * 2]


def passages_block(passages: list[Passage], limit: int = 24000) -> str:
    from .llm_io import clip

    if not passages:
        return "(no library passages retrieved)"
    return clip("\n\n".join(p.as_prompt() for p in passages), limit)
