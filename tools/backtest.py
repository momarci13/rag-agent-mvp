"""Simple, correct event-driven backtester.

No vectorbt / backtrader dependency — pandas + numpy is enough for daily
bars on <100 tickers. Execution occurs on next-bar open (no lookahead).
"""
from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from .risk import (
    cagr,
    deflated_sharpe,
    historical_var,
    max_drawdown,
    sharpe_ratio,
    sortino_ratio,
)


@dataclass
class BacktestConfig:
    initial_equity: float = 100_000.0
    fee_bps: float = 1.0          # one-way commission in basis points
    slippage_bps: float = 2.0     # one-way slippage in bps
    rebalance_days: int = 5
    max_weight: float = 0.2
    long_only: bool = True


def _pct_returns(prices: pd.DataFrame) -> pd.DataFrame:
    return prices.pct_change().fillna(0.0)


def run_backtest(
    prices: pd.DataFrame,
    signal_fn: Callable[[pd.DataFrame], pd.Series],
    cfg: BacktestConfig,
    symbol: str | None = None,
) -> dict:
    """Run a single-asset backtest.

    `signal_fn(df)` must return a Series of positions in [-1, 1]
    aligned to `df.index`. df has columns
    ['open','high','low','close','volume'].

    For multi-asset strategies use `run_portfolio_backtest`.
    """
    df = prices.copy().dropna()
    if len(df) < 30:
        raise ValueError(f"Not enough data: {len(df)} bars")

    # Signal -> desired position, shift by 1 (execute next bar)
    pos = signal_fn(df).reindex(df.index).fillna(0.0).clip(-1, 1)
    pos_exec = pos.shift(1).fillna(0.0)

    # Returns of the asset
    rets = df["close"].pct_change().fillna(0.0)

    # Turnover cost applied on position changes
    trade = pos_exec.diff().abs().fillna(abs(pos_exec.iloc[0]))
    cost_rate = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
    strat_ret = pos_exec * rets - trade * cost_rate

    equity = cfg.initial_equity * (1 + strat_ret).cumprod()
    trades = int((pos_exec.diff().abs() > 1e-9).sum())
    wins = int((strat_ret[strat_ret != 0] > 0).sum())
    total_nz = int((strat_ret != 0).sum())
    win_rate = wins / total_nz if total_nz else 0.0

    sr = sharpe_ratio(strat_ret)
    return {
        "symbol": symbol,
        "equity": equity,
        "returns": strat_ret,
        "sharpe": sr,
        "sortino": sortino_ratio(strat_ret),
        # With 1 trial, DSR ≈ P(SR > 0)
        "deflated_sharpe": deflated_sharpe(sr, strat_ret, n_trials=1),
        "max_drawdown": max_drawdown(equity),
        "cagr": cagr(equity),
        "win_rate": win_rate,
        "n_trades": trades,
        "final_equity": float(equity.iloc[-1]) if len(equity) else cfg.initial_equity,
        "var_99": historical_var(strat_ret, 0.99),
    }


def run_portfolio_backtest(
    prices: dict[str, pd.DataFrame],
    signal_fn: Callable[[pd.DataFrame], pd.Series],
    cfg: BacktestConfig,
    n_trials: int = 1,
) -> dict:
    """Run the signal on each symbol and equal-weight combine.

    `prices` is a dict symbol -> OHLCV DataFrame.
    """
    per_symbol_ret = []
    syms = list(prices.keys())
    aligned_idx = None
    for sym in syms:
        df = prices[sym].dropna()
        pos = signal_fn(df).reindex(df.index).fillna(0.0).clip(-1, 1).shift(1).fillna(0.0)
        rets = df["close"].pct_change().fillna(0.0)
        trade = pos.diff().abs().fillna(abs(pos.iloc[0]))
        cost = (cfg.fee_bps + cfg.slippage_bps) / 10_000.0
        r = pos * rets - trade * cost
        r.name = sym
        per_symbol_ret.append(r)
        aligned_idx = r.index if aligned_idx is None else aligned_idx.intersection(r.index)

    mat = pd.concat(
        [r.reindex(aligned_idx).fillna(0.0) for r in per_symbol_ret], axis=1
    )
    port_ret = mat.mean(axis=1)  # equal weight
    equity = cfg.initial_equity * (1 + port_ret).cumprod()
    sr = sharpe_ratio(port_ret)

    # Trade counting: union across assets
    trades = int(sum((r.diff().abs() > 1e-9).sum() for r in per_symbol_ret))

    return {
        "symbols": syms,
        "equity": equity,
        "returns": port_ret,
        "sharpe": sr,
        "sortino": sortino_ratio(port_ret),
        "deflated_sharpe": deflated_sharpe(sr, port_ret, n_trials=n_trials),
        "max_drawdown": max_drawdown(equity),
        "cagr": cagr(equity),
        "win_rate": float((port_ret > 0).sum() / max(1, (port_ret != 0).sum())),
        "n_trades": trades,
        "final_equity": float(equity.iloc[-1]) if len(equity) else cfg.initial_equity,
        "var_99": historical_var(port_ret, 0.99),
    }


# ---------------- signal compilation ----------------

_SIGNAL_NAMES = {"df", "np", "abs", "min", "max", "len", "float", "int"}
_SIGNAL_CALL_NAMES = {"abs", "min", "max", "len", "float", "int"}
_SIGNAL_METHODS = {
    "abs",
    "astype",
    "between",
    "clip",
    "diff",
    "ewm",
    "expanding",
    "fillna",
    "max",
    "mean",
    "median",
    "min",
    "pct_change",
    "quantile",
    "rank",
    "rolling",
    "shift",
    "std",
    "sum",
    "var",
    "where",
}
_NUMPY_FUNCTIONS = {
    "abs",
    "clip",
    "exp",
    "isfinite",
    "isnan",
    "log",
    "log1p",
    "maximum",
    "minimum",
    "sign",
    "sqrt",
    "tanh",
    "where",
}


def _validate_signal_expression(signal_code: str) -> ast.Expression:
    if not signal_code.strip() or len(signal_code) > 4_000:
        raise ValueError("signal_code must be a non-empty expression under 4,000 characters")
    try:
        tree = ast.parse(signal_code, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"signal_code is not a valid expression: {exc.msg}") from exc
    nodes = list(ast.walk(tree))
    if len(nodes) > 200:
        raise ValueError("signal_code is too complex")

    structural_nodes = (
        ast.Expression,
        ast.Constant,
        ast.Name,
        ast.Load,
        ast.Subscript,
        ast.Slice,
        ast.List,
        ast.Tuple,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.IfExp,
        ast.Call,
        ast.Attribute,
        ast.keyword,
        ast.operator,
        ast.unaryop,
        ast.boolop,
        ast.cmpop,
        ast.expr_context,
    )
    for node in nodes:
        if not isinstance(node, structural_nodes):
            raise ValueError(f"signal_code syntax {type(node).__name__} is not allowed")
        if isinstance(node, ast.Name) and node.id not in _SIGNAL_NAMES:
            raise ValueError(f"signal_code name {node.id!r} is not allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("_"):
                raise ValueError("private and dunder attributes are not allowed")
            if isinstance(node.value, ast.Name) and node.value.id == "np":
                if node.attr not in _NUMPY_FUNCTIONS:
                    raise ValueError(f"NumPy attribute {node.attr!r} is not allowed")
            elif node.attr not in _SIGNAL_METHODS:
                raise ValueError(f"signal method {node.attr!r} is not allowed")
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                if node.func.id not in _SIGNAL_CALL_NAMES:
                    raise ValueError(f"signal function {node.func.id!r} is not allowed")
            elif not isinstance(node.func, ast.Attribute):
                raise ValueError("indirect signal function calls are not allowed")
    return tree

def compile_signal(signal_code: str) -> Callable[[pd.DataFrame], pd.Series]:
    """Compile a StrategySpec.signal_code string into a callable.

    The string must be a single Python EXPRESSION over the symbol `df`
    (DataFrame with OHLCV) that returns a Series in [-1, 1].

    Example: "(df['close'] > df['close'].rolling(50).mean()).astype(float) - 0.5"
    """
    import numpy as _np

    tree = _validate_signal_expression(signal_code)
    compiled = compile(tree, "<strategy-signal>", "eval")
    safe_globals = {
        "__builtins__": {},
        "np": _np,
        "abs": abs,
        "min": min,
        "max": max,
        "len": len,
        "float": float,
        "int": int,
    }

    def _fn(df: pd.DataFrame) -> pd.Series:
        locs = {"df": df}
        sig = eval(compiled, safe_globals, locs)  # noqa: S307 - validated expression DSL
        if isinstance(sig, (int, float)):
            sig = pd.Series(float(sig), index=df.index)
        elif not isinstance(sig, pd.Series):
            sig = pd.Series(sig, index=df.index[: len(sig)])
        return sig.astype(float)
    return _fn


# ---------------- market data helper ----------------

# Moved to tools.data - import from there
