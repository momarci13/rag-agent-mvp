"""Built-in requirement checklists the Validator applies by default.

The references below point to the public legal texts by article; they are a
starting checklist, not a legal interpretation. Attached regulations and
concept papers add document-specific requirements on top (see
validator.RequirementsStep). Verify every reference against the official
text and your institution's policy before relying on it.
"""
from __future__ import annotations

from .schemas import Citation, Requirement

FRAMEWORK_LABELS: dict[str, str] = {
    "sound_practice": "Generic model development and validation practice",
    "eu_banking": "EU banking: CRR, EBA guidelines, ECB guide to internal models, MNB",
    "eu_ai_act": "EU AI Act (Regulation (EU) 2024/1689)",
}

# Folder in data/regulatory -> framework key, for the built-in reference corpus.
CORPUS_FOLDER_FRAMEWORK: dict[str, str] = {
    "eba_gl_2017_11": "eu_banking",
    "eba_gl_2017_16_pd_lgd": "eu_banking",
    "eba_gl_2019_03_downturn_lgd": "eu_banking",
    "ecb_egim": "eu_banking",
    "ecb_trim": "eu_banking",
    "frtb_market_risk": "eu_banking",
    "bcbs_239_risk_data": "eu_banking",
    "corep_finrep_templates": "eu_banking",
    "mnb_circulars": "eu_banking",
    "model_change_529_2014": "eu_banking",
    "eu_ai_act": "eu_ai_act",
}

_CRR = "Regulation (EU) No 575/2013 (CRR)"
_EBA_PDLGD = "EBA/GL/2017/16 (PD and LGD estimation)"
_AIA = "Regulation (EU) 2024/1689 (EU AI Act)"
_CREDIT = ["credit_risk"]
_MARKET = ["market_risk"]


def _r(req_id: str, framework: str, text: str, source: str = "", locator: str = "",
       applies_to: list[str] | None = None) -> Requirement:
    return Requirement(
        req_id=req_id,
        framework=framework,
        text=text,
        citation=Citation(source=source, locator=locator) if source else None,
        applies_to=applies_to or [],
    )


CHECKLISTS: dict[str, list[Requirement]] = {
    "sound_practice": [
        _r("GEN-01", "sound_practice", "Model purpose, intended use, users and limits of use are documented."),
        _r("GEN-02", "sound_practice", "Data sources, sample definition, data quality checks and data treatments are documented and justified."),
        _r("GEN-03", "sound_practice", "The methodology is conceptually sound for the intended use; alternatives considered and key assumptions are justified."),
        _r("GEN-04", "sound_practice", "The implementation matches the documented methodology (code and document are consistent)."),
        _r("GEN-05", "sound_practice", "Results are reproducible from the delivered code and data with a fixed random seed."),
        _r("GEN-06", "sound_practice", "Performance is measured out of sample against acceptance criteria that were set before testing."),
        _r("GEN-07", "sound_practice", "Sensitivity and stress analysis covers the key parameters and assumptions."),
        _r("GEN-08", "sound_practice", "The model is benchmarked against a simpler or alternative approach."),
        _r("GEN-09", "sound_practice", "An ongoing monitoring plan defines metrics, thresholds, frequency and escalation triggers."),
        _r("GEN-10", "sound_practice", "Limitations, residual model risk and compensating controls are documented."),
        _r("GEN-11", "sound_practice", "The test suite covers every item of the documented test plan and all tests pass or failures are explained."),
    ],
    "eu_banking": [
        _r("EUB-01", "eu_banking", "Internal validation is performed independently of model development and covers all rating systems and estimates.", _CRR, "Art. 185"),
        _r("EUB-02", "eu_banking", "Discriminatory power of the rating system is assessed with appropriate statistics (e.g. AUC/Gini) on development and validation samples.", _EBA_PDLGD, "validation of risk parameters", _CREDIT),
        _r("EUB-03", "eu_banking", "Estimated PDs are compared with realised default rates per grade (calibration back-testing) and deviations are explained.", _CRR, "Art. 185(b)", _CREDIT),
        _r("EUB-04", "eu_banking", "Data used for estimation are representative of the population to which the model is applied.", _CRR, "Art. 179", _CREDIT),
        _r("EUB-05", "eu_banking", "The historical observation period meets the regulatory minimum for the exposure class.", _CRR, "Art. 180", _CREDIT),
        _r("EUB-06", "eu_banking", "A margin of conservatism is quantified for identified data and methodological deficiencies.", _EBA_PDLGD, "margin of conservatism", _CREDIT),
        _r("EUB-07", "eu_banking", "LGD estimates reflect downturn conditions where required.", "EBA/GL/2019/03 (downturn LGD)", "", _CREDIT),
        _r("EUB-08", "eu_banking", "Market risk models are back-tested (VaR exceptions, traffic-light zones) and, where applicable, P&L attribution and expected shortfall are assessed.", _CRR, "Art. 366; BCBS d457 (FRTB)", _MARKET),
        _r("EUB-09", "eu_banking", "Risk data aggregation meets accuracy, completeness and timeliness principles.", "BCBS 239", "Principles 3 to 5"),
        _r("EUB-10", "eu_banking", "Model changes are classified for materiality and notified or approved as required.", "Commission Delegated Regulation (EU) No 529/2014", ""),
        _r("EUB-11", "eu_banking", "Applicable national supervisory expectations (MNB recommendations and circulars for Hungarian institutions) are identified and met.", "MNB", ""),
    ],
    "eu_ai_act": [
        _r("AIA-01", "eu_ai_act", "The AI system is classified for risk; creditworthiness assessment or credit scoring of natural persons is high-risk.", _AIA, "Art. 6 and Annex III point 5(b)"),
        _r("AIA-02", "eu_ai_act", "A risk management system covers the whole lifecycle of a high-risk AI system.", _AIA, "Art. 9"),
        _r("AIA-03", "eu_ai_act", "Training, validation and test data are relevant, representative, as free of errors as possible and examined for possible biases.", _AIA, "Art. 10"),
        _r("AIA-04", "eu_ai_act", "Technical documentation contains the elements of Annex IV before the system is placed on the market or put into service.", _AIA, "Art. 11 and Annex IV"),
        _r("AIA-05", "eu_ai_act", "The system automatically records events (logs) over its lifetime.", _AIA, "Art. 12"),
        _r("AIA-06", "eu_ai_act", "Deployers receive instructions for use, including accuracy levels and known limitations.", _AIA, "Art. 13"),
        _r("AIA-07", "eu_ai_act", "Human oversight measures allow people to understand, monitor, override or stop the system.", _AIA, "Art. 14"),
        _r("AIA-08", "eu_ai_act", "Accuracy, robustness and cybersecurity are appropriate and accuracy metrics are declared.", _AIA, "Art. 15"),
    ],
}


def checklist_for(frameworks: list[str], category: str) -> list[Requirement]:
    """Requirements for the chosen frameworks that apply to ``category``.
    Generic practice is always included."""
    keys = ["sound_practice"] + [f for f in frameworks if f in CHECKLISTS and f != "sound_practice"]
    out: list[Requirement] = []
    for key in keys:
        for req in CHECKLISTS[key]:
            if not req.applies_to or category in req.applies_to:
                out.append(req)
    return out
