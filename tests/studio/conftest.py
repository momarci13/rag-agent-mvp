from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rag.embeddings import DeterministicHashEmbeddings
from studio.config import StudioSettings
from studio.library import Library, make_rag
from studio.orchestrator import Studio
from studio.storage import ProjectStore

from .scripted_llm import ScriptedLLM


def make_loans(n: int = 4000, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    year = rng.integers(2016, 2024, n)
    income = rng.lognormal(10.5, 0.4, n)
    debt_ratio = rng.beta(2, 5, n)
    age = rng.integers(21, 70, n)
    logit = -2.2 + 4.0 * debt_ratio - 0.9 * (np.log(income) - 10.5) - 0.01 * (age - 40)
    default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame({"loan_id": np.arange(n), "year": year, "income": income.round(2),
                         "debt_ratio": debt_ratio.round(4), "age": age, "default": default})


@pytest.fixture()
def loans_csv() -> bytes:
    return make_loans().to_csv(index=False).encode()


@pytest.fixture()
def studio(tmp_path: Path) -> Studio:
    settings = StudioSettings(
        output_dir=tmp_path / "studio", pipeline_timeout_s=300, test_timeout_s=300,
        max_repair_attempts=1, pdf_via="none",
    )
    store = ProjectStore(settings.output_dir)
    rag = make_rag({"rag": {"db_path": str(tmp_path / "chroma")}}, "studio-test", DeterministicHashEmbeddings())
    library = Library(tmp_path / "library", rag)
    library.ensure_builtin(settings.builtin_corpus_dir)
    return Studio({}, ScriptedLLM(), library, store, settings)
