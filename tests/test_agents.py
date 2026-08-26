"""Tests for agent roles and graph logic."""

import json

import httpx2
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
        ModelSpec(name="gpt-5.1", priority=1, capabilities=["reasoning"]),
        ModelSpec(name="gpt-5.1-mini", priority=2, capabilities=["fast"]),
    ]
    cfg = LLMConfig(
        model="gpt-5.1-mini",
        models=models,
        selection_strategy=ModelSelectionStrategy.COMPLEXITY_BASED
    )
    llm = HostedLLM(cfg)

    # Test simple task
    selected = llm.select_models("simple")
    assert len(selected) == 2
    assert selected[0].name == "gpt-5.1-mini"

    # Test complex task
    selected = llm.select_models("complex")
    assert len(selected) == 2
    assert selected[0].name == "gpt-5.1"


def test_task_complexity_estimation():
    """Test task complexity estimation."""
    cfg = LLMConfig(model="test")
    llm = HostedLLM(cfg)

    assert llm.estimate_task_complexity("Compute mean") == "simple"
    assert llm.estimate_task_complexity("Analyze complex strategy with optimization") == "complex"
    assert llm.estimate_task_complexity("This is a longer task with enough words to make medium complexity") == "medium"


def test_openai_chat_uses_openai_contract_and_auth():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json={
            "choices": [{"message": {"content": "pong"}}],
        })

    llm = HostedLLM(
        LLMConfig(
            model="gpt-5.1-mini",
            api_key="sk-test",
        ),
        transport=httpx2.MockTransport(handler),
    )
    assert llm.chat([{"role": "user", "content": "ping"}]) == "pong"
    assert captured["path"] == "/v1/chat/completions"
    assert captured["auth"] == "Bearer sk-test"
    assert captured["body"]["model"] == "gpt-5.1-mini"


def test_openai_chat_json_mode_sets_response_format():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return httpx2.Response(200, json={
            "choices": [{"message": {"content": '{"ok": true}'}}],
        })

    llm = HostedLLM(
        LLMConfig(model="gpt-5.1-mini", api_key="sk-test"),
        transport=httpx2.MockTransport(handler),
    )
    result = llm.chat_json([{"role": "user", "content": "ping"}])
    assert result == {"ok": True}
    assert captured["body"]["response_format"] == {"type": "json_object"}


def test_openai_chat_with_fallback_falls_through_tier_chain():
    attempts = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        body = json.loads(request.content)
        attempts.append(body["model"])
        if body["model"] == "gpt-5.1":
            return httpx2.Response(429, json={"error": {"message": "rate limited"}})
        return httpx2.Response(200, json={"choices": [{"message": {"content": "pong"}}]})

    llm = HostedLLM(
        LLMConfig(
            model="gpt-5.1",
            api_key="sk-test",
            models=[
                ModelSpec(name="gpt-5.1", priority=1),
                ModelSpec(name="gpt-5.1-mini", priority=2),
            ],
        ),
        transport=httpx2.MockTransport(handler),
    )
    result = llm.chat_with_fallback([{"role": "user", "content": "ping"}])
    assert result == "pong"
    assert attempts == ["gpt-5.1", "gpt-5.1-mini"]


def test_openai_health_requires_api_key():
    llm = HostedLLM(LLMConfig(api_key="", require_api_key=True))
    assert llm.health() is False


def test_openai_health_validates_key_via_models_list():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["path"] = request.url.path
        captured["auth"] = request.headers.get("authorization")
        return httpx2.Response(200, json={"object": "list", "data": []})

    llm = HostedLLM(
        LLMConfig(api_key="sk-test"),
        transport=httpx2.MockTransport(handler),
    )
    assert llm.health() is True
    assert captured["path"] == "/v1/models"
    assert captured["auth"] == "Bearer sk-test"


def test_openai_health_returns_false_on_error():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(401, json={"error": {"message": "invalid key"}})

    llm = HostedLLM(
        LLMConfig(api_key="sk-bad"),
        transport=httpx2.MockTransport(handler),
    )
    assert llm.health() is False


def test_role_models_route_to_configured_model():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["model"] = json.loads(request.content)["model"]
        return httpx2.Response(200, json={"choices": [{"message": {"content": "ok"}}]})

    llm = HostedLLM(
        LLMConfig(
            model="gpt-5.1-mini",
            api_key="sk-test",
            role_models={"credit_validator": "gpt-5.1"},
        ),
        transport=httpx2.MockTransport(handler),
    )
    llm.chat([{"role": "user", "content": "ping"}], role="credit_validator")
    assert captured["model"] == "gpt-5.1"


def test_default_config_uses_openai_model_defaults(monkeypatch):
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_EMBEDDING_MODEL", raising=False)
    config = load_config()
    llm_config = make_llm_config(config)
    assert llm_config.base_url is None
    assert llm_config.model == "gpt-5.1-mini"
    assert config["rag"]["embedding_model"] == "text-embedding-3-large"


def test_openai_model_override_applies_to_every_role(monkeypatch):
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.1")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    config = make_llm_config(load_config())
    assert config.model == "gpt-5.1"
    assert set(config.role_models.values()) == {"gpt-5.1"}
    assert config.models[0].name == "gpt-5.1"
