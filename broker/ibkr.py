"""IBKR paper-trading adapter and deterministic execution safeguards.

The LLM never receives direct access to the broker client. It can propose a
typed :class:`TradeIntent`; deterministic code performs risk checks, issues a
short-lived one-time approval token, and only then calls the official TWS API.
"""
from __future__ import annotations

import hashlib
import os
import secrets
import threading
import time
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field, model_validator


class TradeIntent(BaseModel):
    """Auditable order proposal produced by the quant/model agents."""

    symbol: str = Field(min_length=1, max_length=24)
    action: Literal["BUY", "SELL"]
    quantity: Decimal = Field(gt=0)
    order_type: Literal["LMT", "MKT"] = "LMT"
    limit_price: Decimal | None = Field(default=None, gt=0)
    tif: Literal["DAY", "GTC", "IOC"] = "DAY"
    exchange: str = "SMART"
    primary_exchange: str | None = None
    currency: str = "USD"
    strategy_name: str
    research_run_id: str
    rationale: str = Field(min_length=10)

    @model_validator(mode="after")
    def require_limit_price(self) -> "TradeIntent":
        if self.order_type == "LMT" and self.limit_price is None:
            raise ValueError("limit_price is required for LMT orders")
        return self

    @property
    def estimated_notional(self) -> Decimal | None:
        if self.limit_price is None:
            return None
        return self.quantity * self.limit_price


class RiskDecision(BaseModel):
    approved: bool
    reasons: list[str] = Field(default_factory=list)
    estimated_notional: Decimal | None = None
    approval_token: str | None = None
    expires_at_epoch: float | None = None


class IBKROrderReceipt(BaseModel):
    order_id: int
    status: str
    transmitted: bool
    account: str
    symbol: str
    submitted_at_epoch: float = Field(default_factory=time.time)


@dataclass(frozen=True)
class TradeRiskPolicy:
    max_order_notional: Decimal = Decimal("10000")
    max_daily_notional: Decimal = Decimal("25000")
    allowed_order_types: frozenset[str] = frozenset({"LMT"})
    allow_market_orders: bool = False


@dataclass
class IBKRSettings:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 7497
    client_id: int = 71
    account: str = ""
    paper_only: bool = True
    transmit: bool = False
    require_approval: bool = True
    connect_timeout_s: float = 10.0

    @classmethod
    def from_mapping(cls, values: dict[str, Any]) -> "IBKRSettings":
        return cls(
            enabled=_env_bool("IBKR_ENABLED", bool(values.get("enabled", False))),
            host=os.getenv("IBKR_HOST", str(values.get("host", "127.0.0.1"))),
            port=int(os.getenv("IBKR_PORT", str(values.get("port", 7497)))),
            client_id=int(os.getenv("IBKR_CLIENT_ID", str(values.get("client_id", 71)))),
            account=os.getenv("IBKR_ACCOUNT", str(values.get("account", ""))),
            paper_only=_env_bool("IBKR_PAPER_ONLY", bool(values.get("paper_only", True))),
            transmit=_env_bool("IBKR_TRANSMIT", bool(values.get("transmit", False))),
            require_approval=bool(values.get("require_approval", True)),
            connect_timeout_s=float(values.get("connect_timeout_s", 10.0)),
        )

    def validate_for_execution(self) -> None:
        if not self.enabled:
            raise ValueError("IBKR execution is disabled; set IBKR_ENABLED=1 after configuring paper TWS")
        if not self.account:
            raise ValueError("IBKR_ACCOUNT is required before order execution")
        if self.paper_only and not self.account.upper().startswith("DU"):
            raise ValueError("Paper-only mode requires an IBKR paper account (normally DU*)")
        if not self.paper_only and not _env_bool("ALLOW_LIVE_TRADING", False):
            raise ValueError("Live IBKR execution requires ALLOW_LIVE_TRADING=1")


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class _Approval:
    intent_fingerprint: str
    token_hash: str
    expires_at: float
    consumed: bool = False


class ApprovalStore:
    """In-memory, one-time, expiring approval tokens bound to exact intents."""

    def __init__(self, ttl_s: int = 300) -> None:
        self.ttl_s = ttl_s
        self._items: dict[str, _Approval] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _fingerprint(intent: TradeIntent) -> str:
        payload = intent.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def issue(self, intent: TradeIntent) -> tuple[str, float]:
        token = secrets.token_urlsafe(32)
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        expires_at = time.time() + self.ttl_s
        with self._lock:
            self._items[token_hash] = _Approval(
                intent_fingerprint=self._fingerprint(intent),
                token_hash=token_hash,
                expires_at=expires_at,
            )
        return token, expires_at

    def consume(self, intent: TradeIntent, token: str) -> None:
        token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
        with self._lock:
            approval = self._items.get(token_hash)
            if approval is None:
                raise PermissionError("Unknown approval token")
            if approval.consumed:
                raise PermissionError("Approval token was already used")
            if time.time() > approval.expires_at:
                raise PermissionError("Approval token expired")
            if approval.intent_fingerprint != self._fingerprint(intent):
                raise PermissionError("Approval token does not match this order")
            approval.consumed = True


class BrokerClient(Protocol):
    def place_order(self, intent: TradeIntent) -> IBKROrderReceipt: ...


class IBKRBrokerClient:
    """Thin official ``ibapi`` adapter.

    The dependency is imported lazily so research-only deployments do not need
    TWS API installed. The adapter waits for ``nextValidId`` before submitting,
    as required by IBKR's order protocol.
    """

    def __init__(self, settings: IBKRSettings) -> None:
        self.settings = settings
        self._app: Any | None = None
        self._thread: threading.Thread | None = None

    def _connect(self) -> Any:
        self.settings.validate_for_execution()
        if self._app is not None and self._app.isConnected():
            return self._app
        try:
            from ibapi.client import EClient
            from ibapi.wrapper import EWrapper
        except ImportError as exc:
            raise RuntimeError(
                "IBKR TWS Python API is not installed. Install the official API package from IBKR."
            ) from exc

        class App(EWrapper, EClient):
            def __init__(self) -> None:
                EClient.__init__(self, self)
                self.next_order_id: int | None = None
                self.ready = threading.Event()
                self.errors: list[tuple[int, int, str]] = []

            def nextValidId(self, orderId: int) -> None:  # noqa: N802 - IBKR callback name
                self.next_order_id = orderId
                self.ready.set()

            def error(self, reqId: int, errorCode: int, errorString: str, *_: Any) -> None:
                self.errors.append((reqId, errorCode, errorString))

        app = App()
        app.connect(self.settings.host, self.settings.port, self.settings.client_id)
        thread = threading.Thread(target=app.run, name="ibkr-api", daemon=True)
        thread.start()
        if not app.ready.wait(self.settings.connect_timeout_s):
            app.disconnect()
            raise TimeoutError("Timed out waiting for IBKR nextValidId")
        self._app = app
        self._thread = thread
        return app

    def place_order(self, intent: TradeIntent) -> IBKROrderReceipt:
        app = self._connect()
        from ibapi.contract import Contract
        from ibapi.order import Order

        contract = Contract()
        contract.symbol = intent.symbol.upper()
        contract.secType = "STK"
        contract.exchange = intent.exchange
        contract.currency = intent.currency
        if intent.primary_exchange:
            contract.primaryExchange = intent.primary_exchange

        order = Order()
        order.account = self.settings.account
        order.action = intent.action
        order.orderType = intent.order_type
        order.totalQuantity = intent.quantity
        order.tif = intent.tif
        order.transmit = self.settings.transmit
        if intent.limit_price is not None:
            order.lmtPrice = float(intent.limit_price)

        order_id = int(app.next_order_id)
        app.next_order_id += 1
        app.placeOrder(order_id, contract, order)
        return IBKROrderReceipt(
            order_id=order_id,
            status="Submitted" if self.settings.transmit else "HeldInTWS",
            transmitted=self.settings.transmit,
            account=self.settings.account,
            symbol=intent.symbol.upper(),
        )

    def disconnect(self) -> None:
        if self._app is not None:
            self._app.disconnect()
            self._app = None


class IBKRTraderAgent:
    """Deterministic risk/approval/execution agent for IBKR orders."""

    def __init__(
        self,
        broker: BrokerClient,
        settings: IBKRSettings,
        policy: TradeRiskPolicy,
        approvals: ApprovalStore | None = None,
    ) -> None:
        self.broker = broker
        self.settings = settings
        self.policy = policy
        self.approvals = approvals or ApprovalStore()
        self._daily_notional = Decimal("0")
        self._lock = threading.Lock()

    def review(self, intent: TradeIntent) -> RiskDecision:
        reasons: list[str] = []
        notional = intent.estimated_notional
        if intent.order_type not in self.policy.allowed_order_types:
            reasons.append(f"Order type {intent.order_type} is not allowed")
        if intent.order_type == "MKT" and not self.policy.allow_market_orders:
            reasons.append("Market orders are disabled")
        if notional is None:
            reasons.append("A limit price is required for deterministic notional checks")
        else:
            if notional > self.policy.max_order_notional:
                reasons.append(
                    f"Order notional {notional} exceeds cap {self.policy.max_order_notional}"
                )
            if self._daily_notional + notional > self.policy.max_daily_notional:
                reasons.append("Daily submitted notional cap would be exceeded")
        try:
            self.settings.validate_for_execution()
        except ValueError as exc:
            reasons.append(str(exc))

        if reasons:
            return RiskDecision(approved=False, reasons=reasons, estimated_notional=notional)
        token, expires = self.approvals.issue(intent)
        return RiskDecision(
            approved=True,
            estimated_notional=notional,
            approval_token=token,
            expires_at_epoch=expires,
        )

    def execute(self, intent: TradeIntent, approval_token: str) -> IBKROrderReceipt:
        if self.settings.require_approval:
            self.approvals.consume(intent, approval_token)
        # The preview check is advisory; this final check and reservation are
        # atomic so concurrent approvals cannot race the daily notional cap.
        # Keep the conservative reservation if broker submission raises because
        # acknowledgement failures can leave the actual IBKR state ambiguous.
        with self._lock:
            review = self._review_without_issuing(intent)
            if not review.approved:
                raise PermissionError(
                    "Order failed final risk check: " + "; ".join(review.reasons)
                )
            if intent.estimated_notional is not None:
                self._daily_notional += intent.estimated_notional
        return self.broker.place_order(intent)

    def _review_without_issuing(self, intent: TradeIntent) -> RiskDecision:
        notional = intent.estimated_notional
        reasons: list[str] = []
        if intent.order_type not in self.policy.allowed_order_types:
            reasons.append(f"Order type {intent.order_type} is not allowed")
        if intent.order_type == "MKT" and not self.policy.allow_market_orders:
            reasons.append("Market orders are disabled")
        if notional is None or notional > self.policy.max_order_notional:
            reasons.append("Order notional is unavailable or above the configured cap")
        if notional is not None and self._daily_notional + notional > self.policy.max_daily_notional:
            reasons.append("Daily submitted notional cap would be exceeded")
        try:
            self.settings.validate_for_execution()
        except ValueError as exc:
            reasons.append(str(exc))
        return RiskDecision(approved=not reasons, reasons=reasons, estimated_notional=notional)
