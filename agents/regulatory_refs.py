"""Controlled vocabulary of regulatory instruments referenced by the bank
risk-validation gates, plus a mapping from validation-area name to the
instrument that most directly speaks to it.

DRAFT / SUPPORT TOOL ONLY -- see agents/risk_schemas.py's module docstring.
The citation strings below are canonical *titles* of the instruments, not
verbatim quotations of their text, and the area -> instrument mapping is an
illustrative drafting aid, not a verified legal cross-reference. A human
validator must confirm every citation against the source document before any
report leaves the drafting stage.

This module has no import-time dependency on the gate/orchestrator code so it
can be reused by prompts, report builders, and tests without a cycle.
"""
from __future__ import annotations

from enum import Enum

from .risk_schemas import RegulatoryReference


class RegulatoryInstrument(str, Enum):
    """Stable identifiers for the instruments the validation gates cite."""

    CRR_IRB = "CRR_IRB"
    CRR_DEFAULT = "CRR_DEFAULT"
    CRR_MARKET_RISK = "CRR_MARKET_RISK"
    CRR_OPERATIONAL_RISK = "CRR_OPERATIONAL_RISK"
    CRR_NSFR = "CRR_NSFR"
    EBA_GL_2017_16 = "EBA_GL_2017_16"
    EBA_GL_2017_11 = "EBA_GL_2017_11"
    EBA_GL_2019_03 = "EBA_GL_2019_03"
    EBA_GL_2017_06 = "EBA_GL_2017_06"
    DELEGATED_REG_529_2014 = "DELEGATED_REG_529_2014"
    ECB_EGIM_GENERAL = "ECB_EGIM_GENERAL"
    ECB_EGIM_CREDIT = "ECB_EGIM_CREDIT"
    ECB_EGIM_MARKET = "ECB_EGIM_MARKET"
    ECB_EGIM_CCR = "ECB_EGIM_CCR"
    ECB_TRIM = "ECB_TRIM"
    BCBS_D457_FRTB = "BCBS_D457_FRTB"
    BCBS_BACKTESTING = "BCBS_BACKTESTING"
    BCBS_239 = "BCBS_239"
    LCR_DELEGATED_REG_2015_61 = "LCR_DELEGATED_REG_2015_61"
    ECB_GUIDE_ICAAP = "ECB_GUIDE_ICAAP"
    ECB_GUIDE_ILAAP = "ECB_GUIDE_ILAAP"
    EBA_ITS_REPORTING = "EBA_ITS_REPORTING"
    IFRS9 = "IFRS9"
    MNB_RECOMMENDATION = "MNB_RECOMMENDATION"


# Canonical instrument titles. NOT verbatim text -- a drafting label only.
CITATIONS: dict[RegulatoryInstrument, str] = {
    RegulatoryInstrument.CRR_IRB: (
        "Regulation (EU) No 575/2013 (CRR), Part Three Title II Chapter 3, "
        "Articles 174-191 (internal ratings based approach)"
    ),
    RegulatoryInstrument.CRR_DEFAULT: (
        "Regulation (EU) No 575/2013 (CRR), Article 178 (default of an obligor)"
    ),
    RegulatoryInstrument.CRR_MARKET_RISK: (
        "Regulation (EU) No 575/2013 (CRR), Part Three Title IV (own funds "
        "requirements for market risk), Articles 325-377"
    ),
    RegulatoryInstrument.CRR_OPERATIONAL_RISK: (
        "Regulation (EU) No 575/2013 (CRR), Part Three Title III (own funds "
        "requirements for operational risk)"
    ),
    RegulatoryInstrument.CRR_NSFR: (
        "Regulation (EU) No 575/2013 (CRR), Part Six Title IV, Article 428b "
        "(net stable funding ratio)"
    ),
    RegulatoryInstrument.EBA_GL_2017_16: (
        "EBA/GL/2017/16 - Guidelines on PD estimation, LGD estimation and the "
        "treatment of defaulted exposures"
    ),
    RegulatoryInstrument.EBA_GL_2017_11: (
        "EBA/GL/2017/11 - Guidelines on the application of the definition of "
        "default under Article 178 of Regulation (EU) No 575/2013"
    ),
    RegulatoryInstrument.EBA_GL_2019_03: (
        "EBA/GL/2019/03 - Guidelines for the estimation of LGD appropriate for "
        "an economic downturn ('downturn LGD estimation')"
    ),
    RegulatoryInstrument.EBA_GL_2017_06: (
        "EBA/GL/2017/06 - Guidelines on credit institutions' credit risk "
        "management practices and accounting for expected credit losses"
    ),
    RegulatoryInstrument.DELEGATED_REG_529_2014: (
        "Commission Delegated Regulation (EU) No 529/2014 - assessment of the "
        "materiality of extensions and changes of the IRB approach and the "
        "Advanced Measurement Approach"
    ),
    RegulatoryInstrument.ECB_EGIM_GENERAL: (
        "ECB guide to internal models (2024 consolidated version), General "
        "topics chapter (governance, internal validation, use test, data "
        "quality, third-party involvement)"
    ),
    RegulatoryInstrument.ECB_EGIM_CREDIT: (
        "ECB guide to internal models (2024 consolidated version), Credit risk "
        "chapter"
    ),
    RegulatoryInstrument.ECB_EGIM_MARKET: (
        "ECB guide to internal models (2024 consolidated version), Market risk "
        "chapter"
    ),
    RegulatoryInstrument.ECB_EGIM_CCR: (
        "ECB guide to internal models (2024 consolidated version), Counterparty "
        "credit risk chapter"
    ),
    RegulatoryInstrument.ECB_TRIM: (
        "ECB guide for the Targeted Review of Internal Models (TRIM), 2017-2021 "
        "- superseded as living guidance by the ECB guide to internal models"
    ),
    RegulatoryInstrument.BCBS_D457_FRTB: (
        "BCBS d457 - Minimum capital requirements for market risk (Fundamental "
        "Review of the Trading Book), January 2019"
    ),
    RegulatoryInstrument.BCBS_BACKTESTING: (
        "BCBS - Supervisory framework for the use of 'backtesting' in "
        "conjunction with the internal models approach to market risk capital "
        "requirements (traffic-light approach)"
    ),
    RegulatoryInstrument.BCBS_239: (
        "BCBS 239 - Principles for effective risk data aggregation and risk "
        "reporting"
    ),
    RegulatoryInstrument.LCR_DELEGATED_REG_2015_61: (
        "Commission Delegated Regulation (EU) 2015/61 - liquidity coverage "
        "requirement for credit institutions"
    ),
    RegulatoryInstrument.ECB_GUIDE_ICAAP: (
        "ECB Guide to the internal capital adequacy assessment process (ICAAP), "
        "November 2018"
    ),
    RegulatoryInstrument.ECB_GUIDE_ILAAP: (
        "ECB Guide to the internal liquidity adequacy assessment process "
        "(ILAAP), November 2018"
    ),
    RegulatoryInstrument.EBA_ITS_REPORTING: (
        "Commission Implementing Regulation (EU) 2021/451 - implementing "
        "technical standards on supervisory reporting (COREP/FINREP)"
    ),
    RegulatoryInstrument.IFRS9: (
        "IFRS 9 Financial Instruments - expected credit loss impairment model "
        "and significant-increase-in-credit-risk assessment"
    ),
    RegulatoryInstrument.MNB_RECOMMENDATION: (
        "MNB (Magyar Nemzeti Bank) recommendation (ajanlas) - national "
        "supervisory expectation layered on EU-level guidance (placeholder)"
    ),
}


# Validation-area name -> primary instrument. Areas are the exact strings the
# ValidationGateAgent and CREDIT_VALIDATION_AREAS use, so a lookup miss here is
# a signal that a new area needs a mapping entry.
AREA_TO_REFERENCE: dict[str, RegulatoryInstrument] = {
    # --- EBA GL 2017/16-style credit checklist ---
    "Conceptual soundness": RegulatoryInstrument.ECB_EGIM_GENERAL,
    "Data quality": RegulatoryInstrument.BCBS_239,
    "Discriminatory power": RegulatoryInstrument.EBA_GL_2017_16,
    "Calibration and back-testing": RegulatoryInstrument.EBA_GL_2017_16,
    "Override analysis": RegulatoryInstrument.EBA_GL_2017_16,
    "IT implementation": RegulatoryInstrument.ECB_EGIM_GENERAL,
    "Use test": RegulatoryInstrument.CRR_IRB,
    "Ongoing monitoring": RegulatoryInstrument.ECB_EGIM_GENERAL,
    "Margin of Conservatism": RegulatoryInstrument.EBA_GL_2017_16,
    "Downturn LGD estimation": RegulatoryInstrument.EBA_GL_2019_03,
    "Model change management": RegulatoryInstrument.DELEGATED_REG_529_2014,
    # --- other credit gate areas ---
    "Portfolio concentration risk": RegulatoryInstrument.ECB_EGIM_CREDIT,
    "IFRS9 staging consistency": RegulatoryInstrument.EBA_GL_2017_06,
    # --- non-credit (market / operational / liquidity) ---
    "VaR backtesting": RegulatoryInstrument.BCBS_BACKTESTING,
    "P&L attribution": RegulatoryInstrument.BCBS_D457_FRTB,
    "Expected shortfall": RegulatoryInstrument.BCBS_D457_FRTB,
    "Operational loss event review": RegulatoryInstrument.CRR_OPERATIONAL_RISK,
    "Liquidity Coverage Ratio": RegulatoryInstrument.LCR_DELEGATED_REG_2015_61,
    "Net Stable Funding Ratio": RegulatoryInstrument.CRR_NSFR,
    "Funding concentration": RegulatoryInstrument.ECB_GUIDE_ILAAP,
    # --- model risk (ECB EGIM / TRIM assessment areas) ---
    "Stability testing": RegulatoryInstrument.ECB_EGIM_GENERAL,
    "Outcomes analysis": RegulatoryInstrument.ECB_EGIM_GENERAL,
    "Benchmarking": RegulatoryInstrument.ECB_EGIM_GENERAL,
    "Ongoing monitoring review": RegulatoryInstrument.ECB_EGIM_GENERAL,
}

_MAPPING_NOTE = (
    "Illustrative mapping of validation area to regulatory instrument; a "
    "drafting aid, not a verified citation. Confirm against the source "
    "document before use."
)


def citation_text(area: str) -> str:
    """Free-text citation string for ``ValidationFinding.regulatory_reference``.
    Empty string when the area has no mapping (keeps the field's existing
    'unset' convention)."""
    instrument = AREA_TO_REFERENCE.get(area)
    return CITATIONS[instrument] if instrument is not None else ""


def reference_for(area: str, domain: str | None = None) -> RegulatoryReference:
    """Structured citation for ``ValidationFinding.regulatory_reference_structured``.

    ``domain`` is accepted for call-site symmetry and future disambiguation
    (an area name shared across domains could map differently) but is not used
    yet -- the area name is currently unique enough.
    """
    instrument = AREA_TO_REFERENCE.get(area)
    if instrument is None:
        return RegulatoryReference(
            note="No structured regulatory mapping for this validation area (illustrative).",
        )
    return RegulatoryReference(paragraph=CITATIONS[instrument], note=_MAPPING_NOTE)
