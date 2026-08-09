"""Composition root for the quant research and IBKR agent team."""
from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any, Callable

from broker.ibkr import (
    IBKRBrokerClient,
    IBKRSettings,
    IBKRTraderAgent,
    TradeRiskPolicy,
)
from rag.hybrid import LiteHybridRAG
from tools.data import MultiSourceFetcher

from .llm import HostedLLM
from .quant_team import (
    ModelRiskThresholds,
    QuantModelAgent,
    QuantResearchAgent,
    QuantRiskAgent,
    QuantTeamOrchestrator,
    RAGAgent,
    TradeProposalAgent,
)


def create_quant_team(
    cfg: dict[str, Any],
    llm: HostedLLM,
    rag: LiteHybridRAG,
    backtest: Callable,
    *,
    broker=None,
    price_provider: Callable[[str], Decimal] | None = None,
) -> QuantTeamOrchestrator:
    ibkr_cfg = cfg.get("ibkr", {})
    settings = IBKRSettings.from_mapping(ibkr_cfg)
    policy = TradeRiskPolicy(
        max_order_notional=Decimal(str(ibkr_cfg.get("max_order_notional", 10000))),
        max_daily_notional=Decimal(str(ibkr_cfg.get("max_daily_notional", 25000))),
        allowed_order_types=frozenset(ibkr_cfg.get("allowed_order_types", ["LMT"])),
        allow_market_orders=bool(ibkr_cfg.get("allow_market_orders", False)),
    )
    trader = IBKRTraderAgent(
        broker=broker or IBKRBrokerClient(settings),
        settings=settings,
        policy=policy,
    )
    risk_cfg = cfg.get("quant_team", {}).get("model_risk", {})
    thresholds = ModelRiskThresholds(
        min_sharpe=float(risk_cfg.get("min_sharpe", 0.0)),
        min_deflated_sharpe=float(risk_cfg.get("min_deflated_sharpe", 0.50)),
        max_drawdown=float(risk_cfg.get("max_drawdown", 0.25)),
        max_var_99=float(risk_cfg.get("max_var_99", 0.05)),
    )
    return QuantTeamOrchestrator(
        rag_agent=RAGAgent(rag, llm),
        research_agent=QuantResearchAgent(llm),
        model_agent=QuantModelAgent(llm, backtest),
        risk_agent=QuantRiskAgent(thresholds),
        proposal_agent=TradeProposalAgent(llm),
        trader_agent=trader,
        price_provider=price_provider or _latest_close,
    )


def _latest_close(symbol: str) -> Decimal:
    end = dt.date.today() + dt.timedelta(days=1)
    start = end - dt.timedelta(days=14)
    frames = MultiSourceFetcher().fetch([symbol], start.isoformat(), end.isoformat())
    frame = frames.get(symbol)
    if frame is None or frame.empty:
        raise ValueError(f"No recent market price available for {symbol}")
    return Decimal(str(float(frame["close"].dropna().iloc[-1])))
