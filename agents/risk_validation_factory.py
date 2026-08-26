"""Composition root for the bank risk-validation agent team."""
from __future__ import annotations

from typing import Any

from rag.hybrid import LiteHybridRAG
from tools.approval import ApprovalStore

from .llm import HostedLLM
from .quant_team import RAGAgent
from .risk_validation_team import (
    CreditRiskValidationAgent,
    ModelRiskValidationAgent,
    NonCreditRiskValidationAgent,
    ReportComposerAgent,
    RiskValidationOrchestrator,
    ValidationGateAgent,
    ValidationThresholds,
)


def create_risk_validation_team(
    cfg: dict[str, Any],
    llm: HostedLLM,
    rag: LiteHybridRAG,
    *,
    approvals: ApprovalStore | None = None,
) -> RiskValidationOrchestrator:
    rv_cfg = cfg.get("risk_validation", {})
    credit_cfg = rv_cfg.get("credit_risk", {})
    non_credit_cfg = rv_cfg.get("non_credit_risk", {})
    market_cfg = non_credit_cfg.get("market", {})
    operational_cfg = non_credit_cfg.get("operational", {})
    liquidity_cfg = non_credit_cfg.get("liquidity", {})
    model_risk_cfg = rv_cfg.get("model_risk", {})

    thresholds = ValidationThresholds(
        credit_max_psi=float(credit_cfg.get("max_psi", 0.25)),
        credit_min_gini=float(credit_cfg.get("min_gini", 0.40)),
        credit_max_backtesting_exception_rate=float(
            credit_cfg.get("max_backtesting_exception_rate", 0.05)
        ),
        credit_max_override_rate_pct=float(credit_cfg.get("max_override_rate_pct", 0.10)),
        market_max_var_exceptions_250d=int(market_cfg.get("max_var_backtesting_exceptions_250d", 4)),
        market_max_var_exceptions_red_250d=int(
            market_cfg.get("max_var_backtesting_exceptions_red_250d", 9)
        ),
        operational_material_loss_threshold=float(
            operational_cfg.get("material_loss_threshold_amount", 100000.0)
        ),
        liquidity_min_lcr_pct=float(liquidity_cfg.get("min_lcr_pct", 100.0)),
        liquidity_min_nsfr_pct=float(liquidity_cfg.get("min_nsfr_pct", 100.0)),
        model_max_psi=float(model_risk_cfg.get("max_psi", 0.25)),
        model_min_gini=float(model_risk_cfg.get("min_gini", 0.40)),
    )

    approval_ttl_s = int(rv_cfg.get("approval_token_ttl_s", 1800))

    return RiskValidationOrchestrator(
        rag_agent=RAGAgent(rag, llm),
        credit_agent=CreditRiskValidationAgent(llm),
        noncredit_agent=NonCreditRiskValidationAgent(llm),
        model_agent=ModelRiskValidationAgent(llm),
        gate_agent=ValidationGateAgent(thresholds),
        composer_agent=ReportComposerAgent(llm),
        approvals=approvals or ApprovalStore(ttl_s=approval_ttl_s),
    )
