from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import SplineTransformer


ArrayLike = np.ndarray | pd.DataFrame | pd.Series


def _to_array(x: ArrayLike) -> np.ndarray:
    if isinstance(x, pd.DataFrame):
        return x.to_numpy(dtype=float)
    if isinstance(x, pd.Series):
        return x.to_numpy(dtype=float).reshape(-1, 1)
    return np.asarray(x, dtype=float)


def evaluate_regression(y_true: ArrayLike, y_pred: ArrayLike) -> dict[str, float]:
    """Return common regression metrics for a prediction set."""
    y_true_arr = _to_array(y_true).ravel()
    y_pred_arr = _to_array(y_pred).ravel()
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true_arr, y_pred_arr))),
        "mae": float(mean_absolute_error(y_true_arr, y_pred_arr)),
        "r2": float(r2_score(y_true_arr, y_pred_arr)),
    }


def generate_multifidelity_dataset(
    n_samples: int = 500,
    n_low_features: int = 3,
    n_high_features: int = 3,
    noise_low: float = 0.5,
    noise_high: float = 0.2,
    random_state: int | None = None,
) -> dict[str, Any]:
    """Create a generic low/high-fidelity regression dataset.

    The low-fidelity target is a noisy approximation of the high-fidelity
    truth. The high-fidelity target adds a residual term that depends on
    the high-fidelity feature set.
    """
    rng = np.random.default_rng(random_state)
    X_low = rng.normal(size=(n_samples, n_low_features))
    X_high = rng.normal(size=(n_samples, n_high_features))

    low_signal = (
        1.8 * X_low[:, 0]
        - 0.6 * X_low[:, 1] ** 2
        + 0.9 * np.sin(X_low[:, 2] if n_low_features > 2 else X_low[:, 0])
    )
    y_low = low_signal + noise_low * rng.standard_normal(n_samples)

    high_residual = 1.2 * np.tanh(X_high[:, 0]) + 0.8 * X_high[:, 1] * X_low[:, 0]
    if n_high_features > 2:
        high_residual += 0.4 * X_high[:, 2] ** 2
    else:
        high_residual += 0.2 * X_high[:, 0]
    y_high = low_signal + high_residual + noise_high * rng.standard_normal(n_samples)

    X_low_df = pd.DataFrame(
        X_low,
        columns=[f"low_{i}" for i in range(n_low_features)],
    )
    X_high_df = pd.DataFrame(
        X_high,
        columns=[f"high_{i}" for i in range(n_high_features)],
    )

    return {
        "X_low": X_low_df,
        "X_high": X_high_df,
        "y_low": pd.Series(y_low, name="y_low"),
        "y_high": pd.Series(y_high, name="y_high"),
    }


@dataclass
class ResidualKAN:
    """Residual multifidelity Kolmogorov–Arnold Network.

    The KAN fits a low-fidelity model on the low-fidelity features,
    then fits a residual model on the high-fidelity features.
    """

    low_model: Any = None
    residual_model: Any = None
    n_spline_knots: int = 6
    spline_degree: int = 3

    def __post_init__(self) -> None:
        if self.low_model is None:
            self.low_model = Ridge(alpha=1.0)
        if self.residual_model is None:
            self.residual_model = Pipeline(
                [
                    (
                        "spline",
                        SplineTransformer(
                            n_knots=self.n_spline_knots,
                            degree=self.spline_degree,
                            include_bias=False,
                        ),
                    ),
                    ("ridge", Ridge(alpha=1.0)),
                ]
            )
        self._is_fitted = False

    def fit(
        self,
        X_low: ArrayLike,
        y_low: ArrayLike,
        X_high: ArrayLike,
        y_high: ArrayLike,
    ) -> "ResidualKAN":
        X_low_arr = _to_array(X_low)
        X_high_arr = _to_array(X_high)
        y_low_arr = _to_array(y_low).ravel()
        y_high_arr = _to_array(y_high).ravel()

        self.low_model.fit(X_low_arr, y_low_arr)
        low_pred = self.low_model.predict(X_low_arr)
        residual = y_high_arr - low_pred
        self.residual_model.fit(X_high_arr, residual)
        self._is_fitted = True
        return self

    def predict(self, X_low: ArrayLike, X_high: ArrayLike) -> np.ndarray:
        if not getattr(self, "_is_fitted", False):
            raise ValueError("ResidualKAN must be fitted before calling predict()")
        low_pred = self.low_model.predict(_to_array(X_low))
        residual_pred = self.residual_model.predict(_to_array(X_high))
        return low_pred + residual_pred

    def predict_low(self, X_low: ArrayLike) -> np.ndarray:
        if not getattr(self, "_is_fitted", False):
            raise ValueError("ResidualKAN must be fitted before calling predict_low()")
        return self.low_model.predict(_to_array(X_low))

    def predict_residual(self, X_high: ArrayLike) -> np.ndarray:
        if not getattr(self, "_is_fitted", False):
            raise ValueError("ResidualKAN must be fitted before calling predict_residual()")
        return self.residual_model.predict(_to_array(X_high))

    def fit_predict(
        self,
        X_low: ArrayLike,
        y_low: ArrayLike,
        X_high: ArrayLike,
        y_high: ArrayLike,
    ) -> np.ndarray:
        self.fit(X_low, y_low, X_high, y_high)
        return self.predict(X_low, X_high)

    def score(self, X_low: ArrayLike, X_high: ArrayLike, y_true: ArrayLike) -> dict[str, float]:
        y_pred = self.predict(X_low, X_high)
        return evaluate_regression(y_true, y_pred)
