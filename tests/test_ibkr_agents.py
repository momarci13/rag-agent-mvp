from decimal import Decimal
from concurrent.futures import ThreadPoolExecutor

import pytest

from agents.quant_team import (
    QuantModelAgent,
    QuantResearchAgent,
    QuantRiskAgent,
    QuantTeamOrchestrator,
    RAGAgent,
    TradeProposalAgent,
)
from broker.ibkr import (
    IBKRBrokerClient,
    IBKROrderReceipt,
    IBKRSettings,
    IBKRTraderAgent,
    TradeIntent,
    TradeRiskPolicy,
)


class FakeBroker:
    def __init__(self):
        self.orders = []

    def place_order(self, intent):
        self.orders.append(intent)
        return IBKROrderReceipt(
            order_id=42,
            status="HeldInTWS",
            transmitted=False,
            account="DU123",
            symbol=intent.symbol,
        )


def _intent():
    return TradeIntent(
        symbol="SPY",
        action="BUY",
        quantity=Decimal("10"),
        order_type="LMT",
        limit_price=Decimal("500"),
        strategy_name="momentum",
        research_run_id="run-1",
        rationale="Validated momentum signal with bounded paper risk.",
    )


def test_trader_requires_bound_one_time_approval():
    broker = FakeBroker()
    settings = IBKRSettings(enabled=True, account="DU123", transmit=False)
    trader = IBKRTraderAgent(
        broker,
        settings,
        TradeRiskPolicy(
            max_order_notional=Decimal("10000"),
            max_daily_notional=Decimal("20000"),
        ),
    )
    intent = _intent()
    decision = trader.review(intent)
    assert decision.approved
    receipt = trader.execute(intent, decision.approval_token or "")
    assert receipt.order_id == 42
    assert len(broker.orders) == 1

    with pytest.raises(PermissionError, match="already used"):
        trader.execute(intent, decision.approval_token or "")


def test_trader_rejects_non_paper_account_in_paper_mode():
    trader = IBKRTraderAgent(
        FakeBroker(),
        IBKRSettings(enabled=True, account="U123", paper_only=True),
        TradeRiskPolicy(),
    )
    decision = trader.review(_intent())
    assert not decision.approved
    assert any("paper account" in reason for reason in decision.reasons)


def test_final_daily_notional_gate_is_atomic_across_concurrent_approvals():
    broker = FakeBroker()
    trader = IBKRTraderAgent(
        broker,
        IBKRSettings(enabled=True, account="DU123", transmit=False),
        TradeRiskPolicy(
            max_order_notional=Decimal("5000"),
            max_daily_notional=Decimal("5000"),
        ),
    )
    first = _intent()
    second = first.model_copy(update={"research_run_id": "run-2"})
    first_decision = trader.review(first)
    second_decision = trader.review(second)
    assert first_decision.approved and second_decision.approved

    def submit(intent, token):
        try:
            return trader.execute(intent, token)
        except PermissionError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(
            pool.map(
                lambda args: submit(*args),
                [
                    (first, first_decision.approval_token or ""),
                    (second, second_decision.approval_token or ""),
                ],
            )
        )

    assert sum(isinstance(item, IBKROrderReceipt) for item in outcomes) == 1
    assert sum(isinstance(item, PermissionError) for item in outcomes) == 1
    assert len(broker.orders) == 1


def test_official_ibapi_adapter_builds_expected_contract_and_order(monkeypatch):
    class FakeApp:
        next_order_id = 100

        def __init__(self):
            self.call = None

        def placeOrder(self, order_id, contract, order):
            self.call = (order_id, contract, order)

    app = FakeApp()
    settings = IBKRSettings(enabled=True, account="DU123", transmit=False)
    client = IBKRBrokerClient(settings)
    monkeypatch.setattr(client, "_connect", lambda: app)

    receipt = client.place_order(_intent())
    order_id, contract, order = app.call
    assert order_id == 100
    assert app.next_order_id == 101
    assert contract.symbol == "SPY"
    assert contract.secType == "STK"
    assert order.account == "DU123"
    assert order.orderType == "LMT"
    assert order.transmit is False
    assert receipt.status == "HeldInTWS"


class FakeRAG:
    def retrieve(self, *_args, **_kwargs):
        return [{"id": "paper-1", "text": "Momentum evidence with transaction costs."}]


class FakeLLM:
    def chat_json(self, _messages, _schema_hint="", *, role=None, **_kwargs):
        if role == "research":
            return {
                "thesis": "Momentum may persist after conservative costs.",
                "evidence": ["paper-1 reports persistence"],
                "counter_evidence": ["turnover can erase returns"],
                "assumptions": ["liquid ETF fills near the limit"],
                "data_requirements": ["adjusted OHLCV"],
                "source_ids": ["paper-1", "fabricated-id"],
            }
        if role == "quant":
            return {
                "name": "SPY momentum",
                "universe": ["SPY"],
                "frequency": "daily",
                "lookback_days": 252,
                "signal": "Price above 50-day average",
                "signal_code": "(df['close'] > df['close'].rolling(50).mean()).astype(float)",
                "position_sizing": "equal_weight",
                "rebalance_days": 5,
            }
        if role == "trader":
            return {
                "symbol": "SPY",
                "action": "BUY",
                "target_notional": "5000",
                "confidence": 0.6,
                "rationale": "Paper proposal follows the validated strategy and risk gates.",
            }
        raise AssertionError(f"unexpected role {role}")


def test_end_to_end_quant_team_prepares_but_does_not_auto_submit():
    llm = FakeLLM()
    broker = FakeBroker()
    settings = IBKRSettings(enabled=True, account="DU123", transmit=False)
    trader = IBKRTraderAgent(
        broker,
        settings,
        TradeRiskPolicy(
            max_order_notional=Decimal("10000"),
            max_daily_notional=Decimal("20000"),
        ),
    )
    def backtest(_spec):
        return {
            "sharpe": 1.1,
            "deflated_sharpe": 0.7,
            "max_drawdown": 0.12,
            "var_99": 0.02,
        }
    team = QuantTeamOrchestrator(
        RAGAgent(FakeRAG(), llm),
        QuantResearchAgent(llm),
        QuantModelAgent(llm, backtest),
        QuantRiskAgent(),
        TradeProposalAgent(llm),
        trader,
        price_provider=lambda _symbol: Decimal("500"),
    )

    run = team.run("Research a conservative SPY momentum strategy")
    assert run.research.source_ids == ["paper-1"]
    assert run.trade_intent.quantity == Decimal("10")
    assert run.execution_risk.approved
    assert broker.orders == []

    receipt = team.execute(run, run.execution_risk.approval_token or "")
    assert receipt.order_id == 42
    assert len(broker.orders) == 1
