"""A deterministic stand-in for the OpenAI client used by the studio tests.

It recognises each agent step from the prompt and returns a canned, schema-
valid response. The canned model code is real: it is executed by the sandbox,
tested by pytest and replicated by the validator.
"""
from __future__ import annotations

import json
import re

API_TEMPLATE = '''
import json, os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

FEATURES = {features}


def load_data(data_dir):
    df = pd.read_csv(os.path.join(data_dir, "loans.csv"))
    return {{"loans": df}}


def split(data):
    df = data["loans"]
    return df[df["year"] < 2022], df[df["year"] >= 2022]


def fit(data, seed=42):
    train, _ = split(data)
    model = LogisticRegression(max_iter=1000, random_state=seed)
    model.fit(train[FEATURES], train["default"])
    return model


def predict(model, df):
    return model.predict_proba(df[FEATURES])[:, 1]


def _psi(a, b, bins=10):
    edges = np.quantile(a, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    pa = np.histogram(a, edges)[0] / len(a) + 1e-6
    pb = np.histogram(b, edges)[0] / len(b) + 1e-6
    return float(np.sum((pa - pb) * np.log(pa / pb)))


def evaluate(model, data):
    train, test = split(data)
    p = predict(model, test)
    auc = roc_auc_score(test["default"], p)
    order = np.argsort(p)
    y = test["default"].to_numpy()[order]
    cum_bad = np.cumsum(y) / y.sum()
    cum_good = np.cumsum(1 - y) / (1 - y).sum()
    return {{
        "auc": float(auc),
        "gini_coefficient": float(2 * auc - 1),
        "ks_statistic": float(np.max(np.abs(cum_bad - cum_good))),
        "brier": float(np.mean((p - test["default"]) ** 2)),
        "mean_pd_test": float(p.mean()),
        "default_rate_test": float(test["default"].mean()),
        "population_stability_index": _psi(predict(model, train), p),
        "data_completeness_pct": float(100 * (1 - data["loans"].isna().mean().mean())),
        "historical_observation_period_years": float(data["loans"]["year"].nunique()),
    }}


def run(data_dir, out_dir, seed=42):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = load_data(data_dir)
    model = fit(data, seed)
    metrics = evaluate(model, data)
    os.makedirs(os.path.join(out_dir, "figures"), exist_ok=True)
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    _, test = split(data)
    plt.figure(figsize=(5, 3))
    plt.hist(predict(model, test), bins=30)
    plt.title("Predicted PD, test sample")
    plt.savefig(os.path.join(out_dir, "figures", "pd_distribution.png"), dpi=100)
    plt.close()
    return metrics
'''

PIPELINE = '''
import argparse
from model.api import run

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    a = p.parse_args()
    print(run(a.data_dir, a.out_dir))
'''

CONFTEST = '''
import os
import pytest
from model.api import load_data, fit, evaluate


@pytest.fixture(scope="session")
def data_dir():
    return os.environ["STUDIO_DATA_DIR"]


@pytest.fixture(scope="session")
def data(data_dir):
    return load_data(data_dir)


@pytest.fixture(scope="session")
def model(data):
    return fit(data, seed=42)


@pytest.fixture(scope="session")
def metrics(model, data):
    return evaluate(model, data)
'''

TESTS = '''
import numpy as np
from model.api import fit, predict, evaluate, split


def test_MT01_no_missing_target(data):
    assert data["loans"]["default"].isna().sum() == 0


def test_MT02_auc_threshold(metrics):
    assert metrics["auc"] >= 0.70, f"AUC {{metrics['auc']:.3f}} < 0.70"


def test_MT03_pd_in_unit_interval(model, data):
    _, test = split(data)
    p = predict(model, test)
    assert ((p >= 0) & (p <= 1)).all()


def test_MT04_reproducible(data):
    a = evaluate(fit(data, 42), data)
    b = evaluate(fit(data, 42), data)
    assert a == b
{extra}
'''

MT05 = '''

def test_MT05_calibration_in_the_large(metrics):
    gap = abs(metrics["mean_pd_test"] - metrics["default_rate_test"])
    assert gap < 0.05, f"calibration gap {gap:.3f}"
'''


def spec(round_no: int) -> dict:
    return {
        "title": "Retail mortgage PD model",
        "category": "credit_risk",
        "subcategory": "logistic regression PD",
        "objective": "Estimate 12-month probability of default for retail mortgages.",
        "intended_use": "IRB PD estimation and origination decisions.",
        "target": "default flag within 12 months",
        "unit_of_analysis": "loan-year",
        "data_description": "Synthetic loan-level panel 2016-2023 with income, debt ratio and age.",
        "data_treatment": ["Out-of-time split: 2016-2021 train, 2022-2023 test."],
        "features": ["income", "debt_ratio", "age"],
        "methodology_summary": "Logistic regression of the default flag on borrower characteristics.",
        "equations": [{"label": "Eq. 1", "latex": r"PD_i = \frac{1}{1 + e^{-(\beta_0 + \beta^\top x_i)}}", "explanation": "Logit link."}],
        "assumptions": ["Relationships are stable over time."],
        "alternatives_considered": ["Gradient boosting (less interpretable)."],
        "acceptance_criteria": [
            {"metric": "auc", "operator": ">=", "threshold": 0.70, "primary": True, "higher_is_better": True, "rationale": "Internal standard"},
            {"metric": "population_stability_index", "operator": "<=", "threshold": 0.25, "higher_is_better": False},
        ],
        "test_plan": [
            {"test_id": "MT-01", "name": "Target completeness", "category": "data_quality", "objective": "No missing target", "method": "count", "acceptance": "0 missing", "severity_if_fail": "high"},
            {"test_id": "MT-02", "name": "Discrimination", "category": "discrimination", "objective": "AUC above threshold", "method": "ROC AUC out of time", "acceptance": "AUC >= 0.70", "severity_if_fail": "high"},
            {"test_id": "MT-03", "name": "PD range", "category": "unit", "objective": "PDs are probabilities", "method": "range check", "acceptance": "0<=PD<=1"},
            {"test_id": "MT-04", "name": "Reproducibility", "category": "reproducibility", "objective": "Same seed same result", "method": "refit", "acceptance": "identical"},
            {"test_id": "MT-05", "name": "Calibration in the large", "category": "calibration", "objective": "Mean PD close to default rate", "method": "difference", "acceptance": "gap < 0.05"},
        ],
        "limitations": ["Synthetic data."],
        "requirement_ids": ["GEN-01", "EUB-02"],
        "monitoring_plan": ["Quarterly AUC and PSI."],
        "regulatory_profile": {"model_type": "PD", "exposure_class": "retail", "portfolio_segment": "mortgages",
                               "estimation_approach": "internal_ratings_based", "jurisdiction": "EU"},
    }


def bundle(round_no: int) -> dict:
    # Round 1 uses a single weak feature and forgets MT-05; round 2 fixes both.
    features = '["age"]' if round_no == 1 else '["income", "debt_ratio", "age"]'
    extra = "" if round_no == 1 else MT05
    return {"files": [
        {"path": "model/__init__.py", "content": ""},
        {"path": "model/api.py", "content": API_TEMPLATE.format(features=features)},
        {"path": "pipeline.py", "content": PIPELINE},
        {"path": "tests/conftest.py", "content": CONFTEST},
        {"path": "tests/test_model.py", "content": TESTS.format(extra=extra).replace("{{", "{").replace("}}", "}")},
    ]}


INDEPENDENT = {
    "plan": [
        {"test_id": "VT-01", "name": "Naive benchmark", "category": "benchmark", "objective": "Beat a constant PD", "method": "Brier vs base rate", "acceptance": "Brier lower than constant", "severity_if_fail": "high"},
        {"test_id": "VT-02", "name": "Stability by year", "category": "stability", "objective": "AUC stable per test year", "method": "AUC per year", "acceptance": "each year AUC >= 0.6", "severity_if_fail": "medium"},
    ],
    "files": [
        {"path": "tests/conftest.py", "content": CONFTEST},
        {"path": "tests/test_independent.py", "content": '''
import numpy as np
from sklearn.metrics import roc_auc_score
from model.api import predict, split


def test_VT01_beats_constant(model, data):
    _, test = split(data)
    p = predict(model, test)
    y = test["default"].to_numpy()
    base = np.mean((y.mean() - y) ** 2)
    brier = np.mean((p - y) ** 2)
    assert brier < base, f"Brier {brier:.4f} not below constant {base:.4f}"


def test_VT02_auc_each_year(model, data):
    _, test = split(data)
    for year, g in test.groupby("year"):
        auc = roc_auc_score(g["default"], predict(model, g))
        assert auc >= 0.6, f"{year}: AUC {auc:.3f}"
'''},
    ],
}

CHALLENGER = {
    "description": "Gradient-boosted trees on the same features.",
    "files": [
        {"path": "model/__init__.py", "content": ""},
        {"path": "pipeline.py", "content": PIPELINE},
        {"path": "model/api.py", "content": '''
import json, os
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
F = ["income", "debt_ratio", "age"]

def load_data(d):
    return {"loans": pd.read_csv(os.path.join(d, "loans.csv"))}

def fit(data, seed=42):
    df = data["loans"]; tr = df[df.year < 2022]
    return GradientBoostingClassifier(n_estimators=60, max_depth=2, random_state=seed).fit(tr[F], tr["default"])

def predict(m, df):
    return m.predict_proba(df[F])[:, 1]

def evaluate(m, data):
    df = data["loans"]; te = df[df.year >= 2022]
    return {"auc": float(roc_auc_score(te["default"], predict(m, te)))}

def run(data_dir, out_dir, seed=42):
    data = load_data(data_dir); m = fit(data, seed); r = evaluate(m, data)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(r, open(os.path.join(out_dir, "metrics.json"), "w"))
    return r
'''},
    ],
}


EXTERNAL_CODE = """
import pandas as pd
from sklearn.linear_model import LogisticRegression
FEATURES = ["income", "debt_ratio", "age"]


def train(df):
    return LogisticRegression(max_iter=1000).fit(df[FEATURES], df["default"])


def score(m, df):
    return m.predict_proba(df[FEATURES])[:, 1]
"""

EXTERNAL_ADAPTER = """
import json, os
import pandas as pd
from sklearn.metrics import roc_auc_score
from pdmodel.core import train, score


def load_data(d):
    return {"loans": pd.read_csv(os.path.join(d, "loans.csv"))}


def split(data):
    df = data["loans"]
    return df[df.year < 2022], df[df.year >= 2022]


def fit(data, seed=42):
    return train(split(data)[0])


def predict(m, df):
    return score(m, df)


def evaluate(m, data):
    te = split(data)[1]
    return {"auc": float(roc_auc_score(te["default"], predict(m, te)))}


def run(data_dir, out_dir, seed=42):
    data = load_data(data_dir)
    r = evaluate(fit(data, seed), data)
    os.makedirs(out_dir, exist_ok=True)
    json.dump(r, open(os.path.join(out_dir, "metrics.json"), "w"))
    return r
"""


class ScriptedLLM:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.round = 1

    def chat_json(self, messages, schema_hint="", *, temperature=0.0, max_retries=2,
                  task_complexity=None, role=None, strict=False):
        user = next(m["content"] for m in reversed(messages) if m["role"] == "user")
        def step(name):
            self.calls.append(name)
        if "Write the model specification" in user:
            step("design"); return spec(1)
        if "Respond to every finding" in user:
            step("remediation"); self.round = 2
            ids = re.findall(r'"finding_id": "([^"]+)"', user)
            return {"responses": [{"finding_id": i, "action": "fixed", "explanation": "Added all borrower features and MT-05.", "changes": ["model/api.py"]} for i in dict.fromkeys(ids)],
                    "spec": spec(2)}
        if "Return the complete package" in user:
            step("implement"); return bundle(self.round)
        if "narrative parts of the modelling document" in user:
            step("narrative")
            return {"executive_summary": "A logistic PD model.\n\nResults are in section 7.",
                    "sections": [{"heading": h, "body": "Text.\n\n- point one\n- point two"} for h in
                                 ("Methodology rationale", "Interpretation of results", "Model risk and limitations assessment", "Guidance for use")]}
        if "List the concrete, checkable requirements" in user:
            step("requirements")
            chunk = re.search(r"\[(D[0-9a-f]{10}:\d+:\d+)\]", user)
            reqs = [{"req_id": "ATT-01", "framework": "Concept paper", "text": "Use an out-of-time test sample.",
                     "citation": {"source": "", "locator": "", "chunk_id": chunk.group(1) if chunk else "missing"}},
                    {"req_id": "ATT-02", "framework": "Concept paper", "text": "Fabricated requirement.",
                     "citation": {"source": "x", "locator": "p. 99", "chunk_id": "DOESNOTEXIST"}}]
            return {"requirements": reqs}
        if "describe the model exactly as the developer documented it" in user:
            step("external-intake")
            sp = spec(2)
            sp["test_plan"] = sp["test_plan"][:1]
            return {"spec": sp, "documented_metrics": {"auc": 0.80},
                    "adapter_files": [
                        {"path": "studio_adapter/__init__.py", "content": ""},
                        {"path": "studio_adapter/api.py", "content": EXTERNAL_ADAPTER},
                        {"path": "studio_pipeline.py", "content": PIPELINE.replace("model.api", "studio_adapter.api")},
                    ]}
        if "Design and implement 6 to 12 independent validation tests" in user:
            step("independent")
            mod = re.search(r"importable as `([\w.]+)`", user).group(1)
            return json.loads(json.dumps(INDEPENDENT).replace("model.api", mod))
        if "Build an independent challenger model" in user:
            step("challenger"); return CHALLENGER
        if "requirement_assessments" in user:
            step("review")
            reqs = re.findall(r'"req_id": "([A-Z]+-\d+)"', user)
            prev = re.findall(r'"finding_id": "(F-Q-\d+)"', user)
            out = {"requirement_assessments": [{"req_id": r, "status": "met", "evidence": "Section 4"} for r in dict.fromkeys(reqs)],
                   "findings": [], "previous_finding_updates": [{"finding_id": p, "status": "closed", "note": "resolved"} for p in prev]}
            if not prev and self.round == 1:
                out["findings"] = [{"finding_id": "new", "title": "Single-variable specification", "area": "Conceptual soundness",
                                    "severity": "high", "description": "Only age is used; income and leverage are omitted.",
                                    "recommendation": "Add affordability drivers.", "citations": [{"source": "", "chunk_id": "DOESNOTEXIST"}]}]
            return out
        if "Write the conclusion of the validation report" in user:
            step("conclusion"); return {"conclusion": "Conclusion text."}
        if "Return only the files you change" in user or "Return only changed files" in user or "Return only the test files you change" in user:
            step("repair"); return {"files": []}
        raise AssertionError("Unrecognised prompt: " + user[:300])
