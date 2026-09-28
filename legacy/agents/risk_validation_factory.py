"""Composition root for the bank risk-validation agent team."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from rag.hybrid import LiteHybridRAG
from tools.risk_storage import RiskStorage

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
    storage: RiskStorage | None = None,
) -> RiskValidationOrchestrator:
    rv_cfg = cfg.get("risk_validation", {})
    credit_cfg = rv_cfg.get("credit_risk", {})
    non_credit_cfg = rv_cfg.get("non_credit_risk", {})
    market_cfg = non_credit_cfg.get("market", {})
    operational_cfg = non_credit_cfg.get("operational", {})
    liquidity_cfg = non_credit_cfg.get("liquidity", {})
    model_risk_cfg = rv_cfg.get("model_risk", {})

    default_deadlines = {"critical": 30, "high": 90, "medium": 180, "low": 365, "observation": 365}
    remediation_deadlines = {
        **default_deadlines,
        **{str(k): int(v) for k, v in rv_cfg.get("remediation_deadlines", {}).items()},
    }
    default_tier_days = {
        "tier_1_high_materiality": int(model_risk_cfg.get("tier_1_revalidation_days", 365)),
        "tier_2_medium_materiality": int(model_risk_cfg.get("tier_2_revalidation_days", 730)),
        "tier_3_low_materiality": int(model_risk_cfg.get("tier_3_revalidation_days", 1095)),
    }

    thresholds = ValidationThresholds(
        credit_max_psi=float(credit_cfg.get("max_psi", 0.25)),
        credit_min_gini=float(credit_cfg.get("min_gini", 0.40)),
        credit_max_backtesting_exception_rate=float(
            credit_cfg.get("max_backtesting_exception_rate", 0.05)
        ),
        credit_backtesting_significance_level=float(
            credit_cfg.get("backtesting_significance_level", 0.05)
        ),
        credit_pd_calibration_significance=float(credit_cfg.get("pd_calibration_significance", 0.05)),
        credit_lgd_backtesting_significance=float(credit_cfg.get("lgd_backtesting_significance", 0.05)),
        credit_lgd_backtesting_tolerance=float(credit_cfg.get("lgd_backtesting_tolerance", 0.10)),
        credit_min_moc_pct=float(credit_cfg.get("min_moc_pct", 0.0)),
        credit_min_data_completeness_pct=float(credit_cfg.get("min_data_completeness_pct", 95.0)),
        credit_min_data_accuracy_pct=float(credit_cfg.get("min_data_accuracy_pct", 98.0)),
        credit_min_observation_period_years=float(credit_cfg.get("min_observation_period_years", 5.0)),
        credit_revalidation_days=int(credit_cfg.get("revalidation_days", 365)),
        credit_revalidation_warning_days=int(credit_cfg.get("revalidation_warning_days", 60)),
        credit_max_override_rate_pct=float(credit_cfg.get("max_override_rate_pct", 0.10)),
        credit_max_single_name_concentration_pct=float(
            credit_cfg.get("max_single_name_concentration_pct", 5.0)
        ),
        market_max_var_exceptions_250d=int(market_cfg.get("max_var_backtesting_exceptions_250d", 4)),
        market_max_var_exceptions_red_250d=int(
            market_cfg.get("max_var_backtesting_exceptions_red_250d", 9)
        ),
        market_kupiec_significance_level=float(market_cfg.get("kupiec_significance_level", 0.05)),
        market_es_confidence_level=float(market_cfg.get("es_confidence_level", 0.975)),
        market_require_expected_shortfall=bool(market_cfg.get("require_expected_shortfall", True)),
        market_pla_spearman_green=float(market_cfg.get("pla_spearman_green", 0.80)),
        market_pla_spearman_amber=float(market_cfg.get("pla_spearman_amber", 0.70)),
        market_pla_ks_green=float(market_cfg.get("pla_ks_green", 0.09)),
        market_pla_ks_amber=float(market_cfg.get("pla_ks_amber", 0.12)),
        operational_material_loss_threshold=float(
            operational_cfg.get("material_loss_threshold_amount", 100000.0)
        ),
        liquidity_min_lcr_pct=float(liquidity_cfg.get("min_lcr_pct", 100.0)),
        liquidity_min_nsfr_pct=float(liquidity_cfg.get("min_nsfr_pct", 100.0)),
        liquidity_max_funding_concentration_pct=float(
            liquidity_cfg.get("max_funding_concentration_pct", 25.0)
        ),
        model_max_psi=float(model_risk_cfg.get("max_psi", 0.25)),
        model_min_gini=float(model_risk_cfg.get("min_gini", 0.40)),
        model_revalidation_warning_days=int(model_risk_cfg.get("revalidation_warning_days", 60)),
        model_tier_revalidation_days=default_tier_days,
        remediation_deadline_days=remediation_deadlines,
        credit_jurisdiction_overrides=dict(credit_cfg.get("jurisdictions", {})),
        model_tier_overrides=dict(model_risk_cfg.get("tiers", {})),
    )

    approval_ttl_s = int(rv_cfg.get("approval_token_ttl_s", 1800))
    required_signoffs = int(rv_cfg.get("required_signoffs", 1))
    require_distinct_signoffs = bool(rv_cfg.get("require_distinct_signoffs", True))
    tier_required_signoffs = {
        str(tier): int(overrides["required_signoffs"])
        for tier, overrides in model_risk_cfg.get("tiers", {}).items()
        if isinstance(overrides, dict) and "required_signoffs" in overrides
    }

    if storage is None:
        storage_cfg = rv_cfg.get("storage", {})
        db_path = storage_cfg.get("db_path", "output/risk_validation/risk_validation.db")
        storage = RiskStorage(Path(db_path))

    return RiskValidationOrchestrator(
        llm=llm,
        rag_agent=RAGAgent(rag, llm),
        credit_agent=CreditRiskValidationAgent(llm),
        noncredit_agent=NonCreditRiskValidationAgent(llm),
        model_agent=ModelRiskValidationAgent(llm),
        gate_agent=ValidationGateAgent(thresholds),
        composer_agent=ReportComposerAgent(llm),
        storage=storage,
        required_signoffs=required_signoffs,
        approval_ttl_s=approval_ttl_s,
        require_distinct_signoffs=require_distinct_signoffs,
        tier_required_signoffs=tier_required_signoffs,
    )
