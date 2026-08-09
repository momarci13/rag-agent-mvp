"""Tests for agent roles and graph logic."""

import json

import httpx
import pytest
from agents.graph import _extract_code
from agents.llm import HostedLLM, LLMConfig, ModelSpec, ModelSelectionStrategy
from run import load_config, make_llm_config


def test_extract_code_python():
    md = """Here's some code:
```python
print("hello")
```
More text."""
    lang, code = _extract_code(md)
    assert lang == "python"
    assert code == 'print("hello")'


def test_extract_code_r():
    md = """R code example:
```r
print("hello from R")
```
End."""
    lang, code = _extract_code(md)
    assert lang == "r"
    assert code == 'print("hello from R")'


def test_extract_code_unknown():
    md = """Generic code:
```
some code
```
"""
    lang, code = _extract_code(md)
    assert lang == "unknown"
    assert code == "some code"


def test_extract_code_no_block():
    md = "No code block here."
    lang, code = _extract_code(md)
    assert lang == "unknown"
    assert code == "No code block here."


def test_model_selection():
    """Test model selection logic."""
    models = [
        ModelSpec(name="anthropic/claude-sonnet-5", priority=1, capabilities=["reasoning"]),
        ModelSpec(name="openrouter/free", priority=2, capabilities=["fast"]),
    ]
    cfg = LLMConfig(
        model="openrouter/free",
        models=models,
        selection_strategy=ModelSelectionStrategy.COMPLEXITY_BASED
    )
    llm = HostedLLM(cfg)

    # Test simple task
    selected = llm.select_models("simple")
    assert len(selected) == 2
    assert selected[0].name == "openrouter/free"

    # Test complex task
    selected = llm.select_models("complex")
    assert len(selected) == 2
    assert selected[0].name == "anthropic/claude-sonnet-5"


def test_task_complexity_estimation():
    """Test task complexity estimation."""
    cfg = LLMConfig(model="test")
    llm = HostedLLM(cfg)

    assert llm.estimate_task_complexity("Compute mean") == "simple"
    assert llm.estimate_task_complexity("Analyze complex strategy with optimization") == "complex"
    assert llm.estimate_task_complexity("This is a longer task with enough words to make medium complexity") == "medium"


def test_openrouter_chat_uses_openai_contract_auth_and_attribution():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        captured["title"] = request.headers.get("x-title")
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "pong"}}],
        })

    llm = HostedLLM(
        LLMConfig(
            model="openrouter/free",
            base_url="https://openrouter.ai/api/v1",
            api_key="sk-or-test",
            extra_headers={"X-Title": "Quant Research RAG Agents"},
        ),
        transport=httpx.MockTransport(handler),
    )
    assert llm.chat([{"role": "user", "content": "ping"}]) == "pong"
    assert captured["path"] == "/api/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-or-test"
    assert captured["title"] == "Quant Research RAG Agents"
    assert captured["body"]["model"] == "openrouter/free"


def test_openrouter_health_requires_api_key():
    llm = HostedLLM(LLMConfig(api_key="", require_api_key=True))
    assert llm.health() is False


def test_openrouter_health_validates_current_key_endpoint():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(200, json={"data": {"is_free_tier": True}})

    llm = HostedLLM(
        LLMConfig(api_key="sk-or-test"),
        transport=httpx.MockTransport(handler),
    )
    assert llm.health() is True
    assert captured == {"path": "/api/v1/key", "auth": "Bearer sk-or-test"}


def test_default_config_uses_only_hosted_free_routes(monkeypatch):
    monkeypatch.delenv("OPENROUTER_MODEL", raising=False)
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENROUTER_EMBEDDING_MODEL", raising=False)
    config = load_config()
    llm_config = make_llm_config(config)
    assert llm_config.base_url == "https://openrouter.ai/api/v1"
    assert llm_config.model == "openrouter/free"
    assert set(llm_config.role_models.values()) == {"openrouter/free"}
    assert config["rag"]["embedding_model"].endswith(":free")


def test_openrouter_model_override_applies_to_every_role(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-5")
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-or-test")
    monkeypatch.setenv("ALLOW_PAID_INFERENCE", "1")
    config = make_llm_config(load_config())
    assert config.model == "anthropic/claude-sonnet-5"
    assert set(config.role_models.values()) == {"anthropic/claude-sonnet-5"}
    assert config.models[0].name == "anthropic/claude-sonnet-5"


def test_paid_model_override_requires_explicit_cost_gate(monkeypatch):
    monkeypatch.setenv("OPENROUTER_MODEL", "anthropic/claude-sonnet-5")
    monkeypatch.delenv("ALLOW_PAID_INFERENCE", raising=False)
    with pytest.raises(ValueError, match="ALLOW_PAID_INFERENCE=1"):
        make_llm_config(load_config())


def test_paid_embedding_override_requires_explicit_cost_gate(monkeypatch):
    monkeypatch.setenv("OPENROUTER_EMBEDDING_MODEL", "openai/text-embedding-3-small")
    monkeypatch.delenv("ALLOW_PAID_INFERENCE", raising=False)
    with pytest.raises(ValueError, match="ALLOW_PAID_INFERENCE=1"):
        load_config()
