"""Tests for the backtest engine — synthetic prices with known properties."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tools.backtest import (
    BacktestConfig,
    compile_signal,
    run_backtest,
    run_portfolio_backtest,
)


def _synthetic_prices(n: int = 252, trend: float = 0.0005, vol: float = 0.01, seed: int = 0):
    """Generate a GBM-ish price series with a deterministic trend."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(trend, vol, n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "open": close * (1 + rng.normal(0, 0.001, n)),
        "high": close * (1 + np.abs(rng.normal(0, 0.002, n))),
        "low":  close * (1 - np.abs(rng.normal(0, 0.002, n))),
        "close": close,
        "volume": rng.integers(1_000_000, 5_000_000, n),
    }, index=idx)
    return df


def test_always_long_equals_buy_and_hold_minus_fees():
    df = _synthetic_prices(n=500, trend=0.0005, seed=42)

    def always_long(df):
        return pd.Series(1.0, index=df.index)

    cfg = BacktestConfig(fee_bps=0.0, slippage_bps=0.0)
    res = run_backtest(df, always_long, cfg)
    # Equity should approximately match price ratio (minus 1st day lag)
    price_return = df["close"].iloc[-1] / df["close"].iloc[1] - 1
    strat_return = res["final_equity"] / cfg.initial_equity - 1
    assert abs(price_return - strat_return) < 0.03


def test_zero_position_gives_flat_equity():
    df = _synthetic_prices(n=300, seed=1)

    def flat(df):
        return pd.Series(0.0, index=df.index)

    cfg = BacktestConfig(fee_bps=0.0, slippage_bps=0.0)
    res = run_backtest(df, flat, cfg)
    assert abs(res["final_equity"] - cfg.initial_equity) < 1e-3


def test_fees_reduce_equity():
    df = _synthetic_prices(n=300, seed=2)

    def whipsaw(df):
        # alternate long/short — big turnover
        return pd.Series(np.where(np.arange(len(df)) % 2 == 0, 1.0, -1.0), index=df.index)

    res_no_fee = run_backtest(df, whipsaw, BacktestConfig(fee_bps=0.0, slippage_bps=0.0))
    res_fee = run_backtest(df, whipsaw, BacktestConfig(fee_bps=5.0, slippage_bps=5.0))
    assert res_fee["final_equity"] < res_no_fee["final_equity"]


def test_compile_signal_sma_crossover():
    df = _synthetic_prices(n=400, seed=3)
    code = (
        "(df['close'] > df['close'].rolling(50).mean()).astype(float) * 1.0"
    )
    sig = compile_signal(code)
    out = sig(df)
    assert isinstance(out, pd.Series)
    assert out.min() >= 0 and out.max() <= 1
    assert len(out) == len(df)


def test_portfolio_backtest_runs():
    prices = {
        "A": _synthetic_prices(n=400, trend=0.0005, seed=10),
        "B": _synthetic_prices(n=400, trend=-0.0002, seed=11),
        "C": _synthetic_prices(n=400, trend=0.0001, seed=12),
    }
    sig = compile_signal("(df['close'] > df['close'].rolling(20).mean()).astype(float)")
    res = run_portfolio_backtest(prices, sig, BacktestConfig())
    assert set(res["symbols"]) == {"A", "B", "C"}
    assert "sharpe" in res
    assert res["final_equity"] > 0


def test_compile_signal_blocks_imports():
    with pytest.raises(ValueError, match="not allowed"):
        compile_signal("__import__('os').system('echo bad')")


@pytest.mark.parametrize(
    "code",
    [
        "df.__class__.__mro__",
        "df['close'].to_pickle('signal.pkl')",
        "np.load('untrusted.npy')",
        "(lambda: 1)()",
    ],
)
def test_compile_signal_rejects_object_traversal_and_io(code):
    with pytest.raises(ValueError, match="not allowed"):
        compile_signal(code)


def test_compile_signal_allows_bounded_numpy_expression():
    df = _synthetic_prices(n=100, seed=9)
    signal = compile_signal(
        "np.where(df['close'] > df['close'].rolling(20).mean(), 1.0, 0.0)"
    )
    out = signal(df)
    assert isinstance(out, pd.Series)
    assert set(out.unique()) <= {0.0, 1.0}
