"""Hosted multi-agent quant research and IBKR paper-trading team."""
from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN
from typing import Any, Callable
from uuid import uuid4

from pydantic import BaseModel, Field

from broker.ibkr import IBKRTraderAgent, RiskDecision, TradeIntent
from rag.hybrid import LiteHybridRAG

from . import roles
from .llm import HostedLLM
from .schemas import StrategySpec


class ResearchBrief(BaseModel):
    thesis: str
    evidence: list[str] = Field(default_factory=list)
    counter_evidence: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    data_requirements: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class ModelRiskAssessment(BaseModel):
    approved_for_order_proposal: bool
    reasons: list[str] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class TradeRecommendation(BaseModel):
    symbol: str
    action: str
    target_notional: Decimal = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=10)


class AgentTrace(BaseModel):
    agent: str
    status: str
    summary: str


class QuantTeamRun(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid4()))
    task: str
    research: ResearchBrief | None = None
    strategy: StrategySpec | None = None
    backtest: dict[str, Any] | None = None
    model_risk: ModelRiskAssessment | None = None
    trade_intent: TradeIntent | None = None
    execution_risk: RiskDecision | None = None
    trace: list[AgentTrace] = Field(default_factory=list)


@dataclass(frozen=True)
class ModelRiskThresholds:
    min_sharpe: float = 0.0
    min_deflated_sharpe: float = 0.50
    max_drawdown: float = 0.25
    max_var_99: float = 0.05


class RAGAgent:
    def __init__(self, rag: LiteHybridRAG, llm: HostedLLM) -> None:
        self.rag = rag
        self.llm = llm

    def run(self, task: str, k: int = 8) -> list[dict[str, Any]]:
        return self.rag.retrieve(task, k=k, llm=self.llm)


class QuantResearchAgent:
    _PROMPT = """You are the Quant Research Agent in a controlled research team.
Use only the supplied RAG evidence. Separate evidence from assumptions and
include the exact source IDs you relied on. Do not recommend an order.
Return JSON matching the supplied schema."""

    def __init__(self, llm: HostedLLM) -> None:
        self.llm = llm

    def run(self, task: str, docs: list[dict[str, Any]]) -> ResearchBrief:
        context = "\n\n".join(
            f"[source:{doc.get('id', '?')}] {doc.get('text', '')}" for doc in docs
        )
        messages = [
            {"role": "system", "content": self._PROMPT},
            {"role": "system", "content": f"RAG evidence:\n{context}"},
            {"role": "user", "content": task},
        ]
        raw = self.llm.chat_json(
            messages,
            schema_hint=json.dumps(ResearchBrief.model_json_schema()),
            role="research",
        )
        brief = ResearchBrief.model_validate(raw)
        available_ids = {str(doc.get("id", "")) for doc in docs}
        brief.source_ids = [source for source in brief.source_ids if source in available_ids]
        return brief


class QuantModelAgent:
    def __init__(
        self,
        llm: HostedLLM,
        backtest: Callable[[StrategySpec], dict[str, Any]],
    ) -> None:
        self.llm = llm
        self.backtest = backtest

    def run(
        self,
        task: str,
        brief: ResearchBrief,
        docs: list[dict[str, Any]],
    ) -> tuple[StrategySpec, dict[str, Any]]:
        grounded_task = (
            f"{task}\n\nResearch thesis: {brief.thesis}\n"
            f"Assumptions: {brief.assumptions}\n"
            "Design a testable strategy; this is research and not an order."
        )
        strategy = roles.design_strategy(self.llm, grounded_task, docs)
        result = self.backtest(strategy)
        return strategy, result


class QuantRiskAgent:
    def __init__(self, thresholds: ModelRiskThresholds | None = None) -> None:
        self.thresholds = thresholds or ModelRiskThresholds()

    def run(self, backtest: dict[str, Any] | None) -> ModelRiskAssessment:
        reasons: list[str] = []
        metrics = dict(backtest or {})
        if not backtest:
            reasons.append("Backtest output is missing")
        elif backtest.get("error"):
            reasons.append(f"Backtest failed: {backtest['error']}")
        else:
            self._minimum(backtest, "sharpe", self.thresholds.min_sharpe, reasons)
            self._minimum(
                backtest,
                "deflated_sharpe",
                self.thresholds.min_deflated_sharpe,
                reasons,
            )
            self._maximum(backtest, "max_drawdown", self.thresholds.max_drawdown, reasons)
            self._maximum(backtest, "var_99", self.thresholds.max_var_99, reasons)
        return ModelRiskAssessment(
            approved_for_order_proposal=not reasons,
            reasons=reasons,
            metrics=metrics,
        )

    @staticmethod
    def _minimum(values: dict[str, Any], key: str, limit: float, reasons: list[str]) -> None:
        value = values.get(key)
        if value is None or float(value) < limit:
            reasons.append(f"{key}={value!r} is below required {limit}")

    @staticmethod
    def _maximum(values: dict[str, Any], key: str, limit: float, reasons: list[str]) -> None:
        value = values.get(key)
        if value is None or float(value) > limit:
            reasons.append(f"{key}={value!r} exceeds allowed {limit}")


class TradeProposalAgent:
    _PROMPT = """You are the Trade Proposal Agent. Produce a single conservative
paper-trading recommendation from the validated research and backtest. Choose
only a symbol in the supplied universe, BUY or SELL, and a target notional.
Do not call a broker. Return JSON matching the supplied schema."""

    def __init__(self, llm: HostedLLM) -> None:
        self.llm = llm

    def run(
        self,
        task: str,
        strategy: StrategySpec,
        backtest: dict[str, Any],
        max_notional: Decimal,
    ) -> TradeRecommendation:
        messages = [
            {"role": "system", "content": self._PROMPT},
            {
                "role": "user",
                "content": (
                    f"Task: {task}\nUniverse: {strategy.universe}\n"
                    f"Strategy: {strategy.model_dump_json()}\n"
                    f"Backtest: {backtest}\nMaximum target notional: {max_notional}"
                ),
            },
        ]
        raw = self.llm.chat_json(
            messages,
            schema_hint=json.dumps(TradeRecommendation.model_json_schema()),
            role="trader",
        )
        recommendation = TradeRecommendation.model_validate(raw)
        recommendation.symbol = recommendation.symbol.upper()
        recommendation.action = recommendation.action.upper()
        if recommendation.symbol not in {symbol.upper() for symbol in strategy.universe}:
            raise ValueError("Trade agent selected a symbol outside the strategy universe")
        if recommendation.action not in {"BUY", "SELL"}:
            raise ValueError("Trade agent action must be BUY or SELL")
        if recommendation.target_notional > max_notional:
            recommendation.target_notional = max_notional
        return recommendation


class QuantTeamOrchestrator:
    """Runs RAG -> research -> model/backtest -> risk -> paper-order proposal."""

    def __init__(
        self,
        rag_agent: RAGAgent,
        research_agent: QuantResearchAgent,
        model_agent: QuantModelAgent,
        risk_agent: QuantRiskAgent,
        proposal_agent: TradeProposalAgent,
        trader_agent: IBKRTraderAgent,
        price_provider: Callable[[str], Decimal],
    ) -> None:
        self.rag_agent = rag_agent
        self.research_agent = research_agent
        self.model_agent = model_agent
        self.risk_agent = risk_agent
        self.proposal_agent = proposal_agent
        self.trader_agent = trader_agent
        self.price_provider = price_provider

    def run(self, task: str) -> QuantTeamRun:
        state = QuantTeamRun(task=task)
        docs = self.rag_agent.run(task)
        state.trace.append(AgentTrace(agent="rag", status="ok", summary=f"Retrieved {len(docs)} chunks"))

        state.research = self.research_agent.run(task, docs)
        state.trace.append(AgentTrace(agent="research", status="ok", summary=state.research.thesis[:200]))

        state.strategy, state.backtest = self.model_agent.run(task, state.research, docs)
        state.trace.append(AgentTrace(agent="model", status="ok", summary=state.strategy.name))

        state.model_risk = self.risk_agent.run(state.backtest)
        state.trace.append(AgentTrace(
            agent="risk",
            status="ok" if state.model_risk.approved_for_order_proposal else "rejected",
            summary="; ".join(state.model_risk.reasons) or "Backtest gates passed",
        ))
        if not state.model_risk.approved_for_order_proposal:
            return state

        recommendation = self.proposal_agent.run(
            task,
            state.strategy,
            state.backtest,
            self.trader_agent.policy.max_order_notional,
        )
        price = self.price_provider(recommendation.symbol)
        if price <= 0:
            raise ValueError("Price provider returned a non-positive price")
        quantity = (recommendation.target_notional / price).to_integral_value(rounding=ROUND_DOWN)
        if quantity <= 0:
            raise ValueError("Target notional is too small for one unit at the current price")
        state.trade_intent = TradeIntent(
            symbol=recommendation.symbol,
            action=recommendation.action,
            quantity=quantity,
            order_type="LMT",
            limit_price=price,
            strategy_name=state.strategy.name,
            research_run_id=state.run_id,
            rationale=recommendation.rationale,
        )
        state.execution_risk = self.trader_agent.review(state.trade_intent)
        state.trace.append(AgentTrace(
            agent="ibkr_trader",
            status="approval_required" if state.execution_risk.approved else "rejected",
            summary="; ".join(state.execution_risk.reasons) or "One-time human approval required",
        ))
        return state

    def execute(self, run: QuantTeamRun, approval_token: str):
        if run.trade_intent is None or run.execution_risk is None:
            raise ValueError("This run has no approved trade intent")
        return self.trader_agent.execute(run.trade_intent, approval_token)
