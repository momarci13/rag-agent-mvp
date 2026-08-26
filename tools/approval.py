"""Domain-neutral, one-time, expiring human-approval token store.

Used to gate any side-effecting action on an explicit, per-item human
sign-off: broker order execution (broker/ibkr.py::IBKRTraderAgent) and
risk-validation report finalisation
(agents/risk_validation_team.py::RiskValidationOrchestrator) both issue a
token bound to the exact reviewed object (via a SHA-256 fingerprint of its
serialized form) and require that same token, unmodified, to proceed.
"""
from __future__ import annotations

import hashlib
import secrets
import threading
import time
from dataclasses import dataclass

from pydantic import BaseModel


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
    def _fingerprint(intent: BaseModel) -> str:
        payload = intent.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def issue(self, intent: BaseModel) -> tuple[str, float]:
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

    def consume(self, intent: BaseModel, token: str) -> None:
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
