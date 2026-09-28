from __future__ import annotations

import numpy as np
import pandas as pd

from tools.multifidelity_kan import (
    ResidualKAN,
    evaluate_regression,
    generate_multifidelity_dataset,
)


def test_synthetic_dataset_shapes():
    data = generate_multifidelity_dataset(
        n_samples=120,
        n_low_features=3,
        n_high_features=4,
        random_state=0,
    )

    assert isinstance(data["X_low"], pd.DataFrame)
    assert isinstance(data["X_high"], pd.DataFrame)
    assert data["X_low"].shape == (120, 3)
    assert data["X_high"].shape == (120, 4)
    assert data["y_low"].shape == (120,)
    assert data["y_high"].shape == (120,)


def test_residual_kan_improves_baseline():
    data = generate_multifidelity_dataset(
        n_samples=250,
        n_low_features=3,
        n_high_features=4,
        noise_low=0.4,
        noise_high=0.2,
        random_state=7,
    )
    model = ResidualKAN()
    model.fit(data["X_low"], data["y_low"], data["X_high"], data["y_high"])

    y_pred = model.predict(data["X_low"], data["X_high"])
    baseline_pred = model.predict_low(data["X_low"])

    baseline_metrics = evaluate_regression(data["y_high"], baseline_pred)
    kan_metrics = evaluate_regression(data["y_high"], y_pred)

    assert kan_metrics["rmse"] < baseline_metrics["rmse"]
    assert kan_metrics["r2"] > baseline_metrics["r2"]


def test_evaluate_regression_returns_metrics():
    y_true = np.array([0.0, 1.0, 2.0, 3.0])
    y_pred = np.array([0.1, 0.9, 2.2, 3.1])
    metrics = evaluate_regression(y_true, y_pred)

    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert metrics["r2"] <= 1.0
    assert metrics["r2"] >= 0.0
