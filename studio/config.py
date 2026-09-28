"""Configuration loading for the Modeler/Validator studio."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from agents.llm import LLMConfig, ModelSelectionStrategy, ModelSpec

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "config.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    if not cfg_path.is_absolute():
        cfg_path = ROOT / cfg_path
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    embedding_override = os.getenv("OPENAI_EMBEDDING_MODEL", "").strip()
    if embedding_override:
        cfg.setdefault("rag", {})["embedding_model"] = embedding_override
    return cfg


def make_llm_config(cfg: dict[str, Any]) -> LLMConfig:
    """OpenAI client config; the API key only ever comes from the environment."""
    llm_cfg = cfg.get("llm", {})
    models: list[ModelSpec] | None = None
    if "models" in llm_cfg:
        models = [ModelSpec(**m) for m in llm_cfg["models"]]

    model_override = os.getenv("OPENAI_MODEL", "").strip()
    selected_model = model_override or llm_cfg.get("model", "gpt-5.1-mini")
    if model_override:
        models = [ModelSpec(name=model_override, priority=0)] + [
            m for m in (models or []) if m.name != model_override
        ]

    strategy = ModelSelectionStrategy(llm_cfg.get("selection_strategy", "priority"))
    role_models = dict(llm_cfg.get("role_models", {}))
    if model_override:
        role_models = {role: model_override for role in role_models}

    return LLMConfig(
        model=selected_model,
        base_url=os.getenv("OPENAI_BASE_URL") or llm_cfg.get("base_url") or None,
        api_key=os.getenv("OPENAI_API_KEY", ""),
        provider_name=llm_cfg.get("provider_name", "OpenAI"),
        require_api_key=bool(llm_cfg.get("require_api_key", True)),
        temperature=llm_cfg.get("temperature", 0.2),
        timeout_s=llm_cfg.get("timeout_s", 240),
        max_output_tokens=llm_cfg.get("max_output_tokens", 16000),
        models=models,
        selection_strategy=strategy,
        fallback_timeout_s=llm_cfg.get("fallback_timeout_s", 90),
        role_models=role_models,
    )


@dataclass(frozen=True)
class StudioSettings:
    output_dir: Path
    max_rounds: int = 3
    pipeline_timeout_s: int = 900
    test_timeout_s: int = 900
    sandbox_mem_mb: int = 4096
    max_repair_attempts: int = 2
    replication_rel_tol: float = 1e-6
    replication_abs_tol: float = 1e-9
    challenger_material_gap: float = 0.02
    pdf_via: str = "libreoffice"
    data_sample_rows: int = 5
    library_collection: str = "studio-library-v1"
    builtin_corpus_dir: Path = ROOT / "data" / "regulatory"

    @classmethod
    def from_config(cls, cfg: dict[str, Any]) -> "StudioSettings":
        s = cfg.get("studio", {})
        out = Path(s.get("output_dir", "output/studio"))
        corpus = Path(s.get("builtin_corpus_dir", "data/regulatory"))
        return cls(
            output_dir=out if out.is_absolute() else ROOT / out,
            max_rounds=int(s.get("max_rounds", 3)),
            pipeline_timeout_s=int(s.get("pipeline_timeout_s", 900)),
            test_timeout_s=int(s.get("test_timeout_s", 900)),
            sandbox_mem_mb=int(s.get("sandbox_mem_mb", 4096)),
            max_repair_attempts=int(s.get("max_repair_attempts", 2)),
            replication_rel_tol=float(s.get("replication_rel_tol", 1e-6)),
            replication_abs_tol=float(s.get("replication_abs_tol", 1e-9)),
            challenger_material_gap=float(s.get("challenger_material_gap", 0.02)),
            pdf_via=str(s.get("pdf_via", "libreoffice")),
            data_sample_rows=int(s.get("data_sample_rows", 5)),
            library_collection=str(s.get("library_collection", "studio-library-v1")),
            builtin_corpus_dir=corpus if corpus.is_absolute() else ROOT / corpus,
        )
