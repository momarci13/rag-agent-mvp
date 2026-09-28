"""Structured ML/DL experiment framework with tracking and comparison.

Step 3: Provides typed ExperimentSpec, ExperimentResult, ExperimentRegistry,
sklearn/timeseries-CV runners, and a markdown comparison table generator.
Designed to be callable from the agent sandbox code for reproducible experiments.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional
from uuid import uuid4

import numpy as np

EXPERIMENTS_DIR = Path(__file__).parent.parent / "output" / "experiments"

ModelType = Literal[
    "linear_regression", "ridge", "lasso", "elastic_net",
    "random_forest", "gradient_boosting", "xgboost",
    "mlp", "lstm", "arima",
]


@dataclass
class ExperimentSpec:
    name: str
    model_type: ModelType
    target: str
    features: list[str]
    data_source: str
    train_ratio: float = 0.8
    validation: Literal["holdout", "timeseries_cv", "walk_forward"] = "timeseries_cv"
    n_splits: int = 5
    hyperparams: dict[str, Any] = field(default_factory=dict)
    random_state: int = 42
    experiment_id: str = field(default_factory=lambda: str(uuid4())[:8])


@dataclass
class ExperimentResult:
    experiment_id: str
    spec_name: str
    model_type: str
    metrics: dict[str, float]
    train_time_s: float
    n_train: int
    n_test: int
    feature_importance: dict[str, float] = field(default_factory=dict)
    predictions: list[float] = field(default_factory=list)
    actuals: list[float] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    notes: str = ""


class ExperimentRegistry:
    """JSON-backed experiment tracking registry."""

    def __init__(self, dir_path: Path = EXPERIMENTS_DIR):
        self.dir = dir_path
        self.dir.mkdir(parents=True, exist_ok=True)
        self._index_path = dir_path / "index.json"

    def save(self, result: ExperimentResult) -> str:
        result_path = self.dir / f"{result.experiment_id}.json"
        result_path.write_text(json.dumps(asdict(result), indent=2, default=str), encoding="utf-8")
        self._upsert_index(result)
        return result.experiment_id

    def load(self, experiment_id: str) -> Optional[ExperimentResult]:
        path = self.dir / f"{experiment_id}.json"
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return ExperimentResult(**data)

    def list_all(self) -> list[dict]:
        if not self._index_path.exists():
            return []
        return json.loads(self._index_path.read_text(encoding="utf-8")).get("experiments", [])

    def compare_table(self, experiment_ids: list[str] | None = None) -> str:
        """Return a markdown table comparing experiments by metric."""
        entries = self.list_all()
        if experiment_ids:
            entries = [e for e in entries if e["experiment_id"] in experiment_ids]
        if not entries:
            return "No experiments to compare."

        all_metrics: set[str] = set()
        for e in entries:
            all_metrics.update(e.get("metrics", {}).keys())
        metric_cols = sorted(m for m in all_metrics if not m.endswith("_std"))

        cols = ["Name", "Model"] + metric_cols + ["Train(s)", "N_train", "Date"]
        header = "| " + " | ".join(cols) + " |"
        sep = "| " + " | ".join("---" for _ in cols) + " |"
        rows = [header, sep]

        for e in entries:
            m = e.get("metrics", {})
            row = (
                [e.get("spec_name", "")[:22], e.get("model_type", "")]
                + [f"{m[k]:.4f}" if isinstance(m.get(k), float) else str(m.get(k, "")) for k in metric_cols]
                + [f"{e.get('train_time_s', 0):.1f}", str(e.get("n_train", "")), e.get("created_at", "")[:10]]
            )
            rows.append("| " + " | ".join(row) + " |")

        return "\n".join(rows)

    def _upsert_index(self, result: ExperimentResult) -> None:
        existing = [e for e in self.list_all() if e["experiment_id"] != result.experiment_id]
        existing.append({
            "experiment_id": result.experiment_id,
            "spec_name": result.spec_name,
            "model_type": result.model_type,
            "metrics": result.metrics,
            "train_time_s": result.train_time_s,
            "n_train": result.n_train,
            "n_test": result.n_test,
            "created_at": result.created_at,
        })
        self._index_path.write_text(json.dumps({"experiments": existing}, indent=2, default=str), encoding="utf-8")


# ---------- model builders ----------

def _build_sklearn_model(spec: ExperimentSpec):
    hp, rs = spec.hyperparams, spec.random_state
    mt = spec.model_type

    if mt == "linear_regression":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    if mt == "ridge":
        from sklearn.linear_model import Ridge
        return Ridge(alpha=hp.get("alpha", 1.0), random_state=rs)
    if mt == "lasso":
        from sklearn.linear_model import Lasso
        return Lasso(alpha=hp.get("alpha", 0.1), random_state=rs, max_iter=3000)
    if mt == "elastic_net":
        from sklearn.linear_model import ElasticNet
        return ElasticNet(alpha=hp.get("alpha", 0.1), l1_ratio=hp.get("l1_ratio", 0.5),
                          random_state=rs, max_iter=3000)
    if mt == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=hp.get("n_estimators", 100),
                                     max_depth=hp.get("max_depth"), random_state=rs, n_jobs=-1)
    if mt == "gradient_boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=hp.get("n_estimators", 100),
                                         learning_rate=hp.get("learning_rate", 0.1),
                                         max_depth=hp.get("max_depth", 3), random_state=rs)
    if mt == "xgboost":
        try:
            from xgboost import XGBRegressor
            return XGBRegressor(n_estimators=hp.get("n_estimators", 100),
                                learning_rate=hp.get("learning_rate", 0.1),
                                max_depth=hp.get("max_depth", 6),
                                random_state=rs, verbosity=0)
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            print("[EXPERIMENT] xgboost not installed, falling back to GradientBoostingRegressor")
            return GradientBoostingRegressor(random_state=rs)
    if mt == "mlp":
        from sklearn.neural_network import MLPRegressor
        return MLPRegressor(hidden_layer_sizes=hp.get("hidden_layer_sizes", (64, 32)),
                            max_iter=hp.get("max_iter", 500), random_state=rs, early_stopping=True)
    raise ValueError(f"Unknown model_type for sklearn: {mt}")


def _regression_metrics(y_true, y_pred) -> dict[str, float]:
    from sklearn import metrics as skm
    return {
        "mse": float(skm.mean_squared_error(y_true, y_pred)),
        "rmse": float(np.sqrt(skm.mean_squared_error(y_true, y_pred))),
        "mae": float(skm.mean_absolute_error(y_true, y_pred)),
        "r2": float(skm.r2_score(y_true, y_pred)),
    }


def _feat_imp(model, features: list[str]) -> dict[str, float]:
    if hasattr(model, "feature_importances_") and features:
        imp = model.feature_importances_
        return {features[i]: float(imp[i]) for i in range(min(len(features), len(imp)))}
    if hasattr(model, "coef_") and features:
        coef = np.abs(np.array(model.coef_).ravel())
        return {features[i]: float(coef[i]) for i in range(min(len(features), len(coef)))}
    return {}


# ---------- runners ----------

def run_holdout(spec: ExperimentSpec, X, y) -> ExperimentResult:
    """Train/test split experiment."""
    n = len(y)
    split = int(n * spec.train_ratio)
    X_tr, X_te = X[:split], X[split:]
    y_tr, y_te = y[:split], y[split:]

    model = _build_sklearn_model(spec)
    t0 = time.time()
    model.fit(X_tr, y_tr)
    elapsed = time.time() - t0

    y_pred = model.predict(X_te)
    return ExperimentResult(
        experiment_id=spec.experiment_id,
        spec_name=spec.name,
        model_type=spec.model_type,
        metrics=_regression_metrics(y_te, y_pred),
        train_time_s=round(elapsed, 3),
        n_train=len(y_tr),
        n_test=len(y_te),
        feature_importance=_feat_imp(model, spec.features),
        predictions=list(np.array(y_pred[:100]).tolist()),
        actuals=list(np.array(y_te[:100]).tolist()),
        notes="holdout split",
    )


def run_timeseries_cv(spec: ExperimentSpec, X, y) -> ExperimentResult:
    """Time-series cross-validation (no future leakage)."""
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=spec.n_splits)

    fold_metrics: list[dict] = []
    feat_imps: list[dict] = []
    t0 = time.time()

    for tr_idx, te_idx in tscv.split(X):
        model = _build_sklearn_model(spec)
        model.fit(X[tr_idx], y[tr_idx])
        y_pred = model.predict(X[te_idx])
        fold_metrics.append(_regression_metrics(y[te_idx], y_pred))
        fi = _feat_imp(model, spec.features)
        if fi:
            feat_imps.append(fi)

    elapsed = time.time() - t0
    avg = {k: float(np.mean([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    std = {f"{k}_std": float(np.std([m[k] for m in fold_metrics])) for k in fold_metrics[0]}
    avg.update(std)

    avg_fi: dict[str, float] = {}
    if feat_imps:
        all_keys = set().union(*feat_imps)
        avg_fi = {k: float(np.mean([fi.get(k, 0.0) for fi in feat_imps])) for k in all_keys}

    n_tr = int(len(y) * spec.train_ratio)
    return ExperimentResult(
        experiment_id=spec.experiment_id,
        spec_name=spec.name,
        model_type=spec.model_type,
        metrics=avg,
        train_time_s=round(elapsed, 3),
        n_train=n_tr,
        n_test=len(y) - n_tr,
        feature_importance=avg_fi,
        notes=f"TimeSeriesSplit CV, {spec.n_splits} folds",
    )


def run_experiment(spec: ExperimentSpec, X, y) -> ExperimentResult:
    """Dispatch to the appropriate runner based on spec.validation."""
    X_arr = np.array(X)
    y_arr = np.array(y)
    if spec.validation == "holdout":
        return run_holdout(spec, X_arr, y_arr)
    return run_timeseries_cv(spec, X_arr, y_arr)


# Module-level registry singleton
registry = ExperimentRegistry()
