import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from agent.tools import ensemble as ens_mod
from agent.tools.ensemble import (
    _llm_for_provider,
    _provider_has_key,
    aggregate_estimates,
    get_ensemble_llms,
    run_ensemble_estimate,
)
from models import ResearchEstimate


@pytest.fixture
def _settings_clear():
    from config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _make_settings(**overrides):
    from config import Settings

    defaults = {
        "groq_api_key": None,
        "llm_api_key": "changeme",
        "llm_base_url": "https://example.com/v1",
        "llm_model": "test-model",
    }
    defaults.update(overrides)
    return Settings(**defaults)


# ── _provider_has_key ───────────────────────────────────────────────────────

def test_provider_has_key_groq_set():
    s = _make_settings(groq_api_key="gsk_abc")
    assert _provider_has_key(s, "groq") is True


def test_provider_has_key_groq_changeme():
    s = _make_settings(groq_api_key="changeme")
    assert _provider_has_key(s, "groq") is False


def test_provider_has_key_groq_none():
    s = _make_settings(groq_api_key=None)
    assert _provider_has_key(s, "groq") is False


def test_provider_has_key_nvidia_set():
    s = _make_settings(llm_api_key="nvapi-xyz")
    assert _provider_has_key(s, "nvidia") is True


def test_provider_has_key_nvidia_wrong_prefix():
    s = _make_settings(llm_api_key="sk-xyz")
    assert _provider_has_key(s, "nvidia") is False


def test_provider_has_key_openai_set():
    s = _make_settings(llm_api_key="sk-real")
    assert _provider_has_key(s, "openai") is True


def test_provider_has_key_openai_changeme():
    s = _make_settings(llm_api_key="changeme")
    assert _provider_has_key(s, "openai") is False


def test_provider_has_key_unknown():
    s = _make_settings()
    assert _provider_has_key(s, "anthropic") is False


# ── _llm_for_provider ───────────────────────────────────────────────────────

def test_llm_for_provider_groq(monkeypatch):
    captured = {}

    class FakeGroq:
        def __init__(self, **kw):
            captured.update(kw)

    import langchain_groq

    monkeypatch.setattr(langchain_groq, "ChatGroq", FakeGroq)
    s = _make_settings(groq_api_key="gsk_a")
    out = _llm_for_provider(s, "groq")
    assert isinstance(out, FakeGroq)
    assert captured["api_key"] == "gsk_a"
    assert captured["streaming"] is False


def test_llm_for_provider_nvidia(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            captured.update(kw)

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeOpenAI)
    s = _make_settings(llm_api_key="nvapi-x")
    out = _llm_for_provider(s, "nvidia")
    assert isinstance(out, FakeOpenAI)
    assert "nvidia" in captured["base_url"]


def test_llm_for_provider_openai_default(monkeypatch):
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kw):
            captured.update(kw)

    import langchain_openai

    monkeypatch.setattr(langchain_openai, "ChatOpenAI", FakeOpenAI)
    s = _make_settings(llm_api_key="sk-y", llm_base_url="https://custom.example/v1")
    out = _llm_for_provider(s, "openai")
    assert isinstance(out, FakeOpenAI)
    assert captured["base_url"] == "https://custom.example/v1"
    assert captured["api_key"] == "sk-y"


# ── get_ensemble_llms ───────────────────────────────────────────────────────

def test_get_ensemble_llms_no_keys_fallback_active(monkeypatch, _settings_clear):
    monkeypatch.setenv("GROQ_API_KEY", "")
    monkeypatch.setenv("LLM_API_KEY", "changeme")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    sentinel = MagicMock(name="active_llm")
    monkeypatch.setattr(ens_mod, "get_llm", lambda streaming=False: sentinel)

    out = get_ensemble_llms()
    assert len(out) == 1
    assert out[0][1] is sentinel


def test_get_ensemble_llms_groq_only(monkeypatch, _settings_clear):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_real")
    monkeypatch.setenv("LLM_API_KEY", "changeme")

    fake = MagicMock(name="groq_llm")
    monkeypatch.setattr(ens_mod, "_llm_for_provider", lambda s, name: fake)

    out = get_ensemble_llms()
    names = [n for n, _ in out]
    assert names == ["groq"]
    assert out[0][1] is fake


def test_get_ensemble_llms_multiple_providers(monkeypatch, _settings_clear):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_real")
    monkeypatch.setenv("LLM_API_KEY", "nvapi-real")  # nvidia path qualifies

    built = {}

    def fake_build(s, name):
        m = MagicMock(name=f"llm_{name}")
        built[name] = m
        return m

    monkeypatch.setattr(ens_mod, "_llm_for_provider", fake_build)

    out = get_ensemble_llms()
    names = [n for n, _ in out]
    # groq qualifies; nvidia qualifies (nvapi- prefix). openai also qualifies
    # because llm_api_key is non-changeme.
    assert "groq" in names
    assert "nvidia" in names
    assert "openai" in names


def test_get_ensemble_llms_skips_failed_provider(monkeypatch, _settings_clear):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_real")
    monkeypatch.setenv("LLM_API_KEY", "changeme")

    def fake_build(s, name):
        raise RuntimeError("boom")

    sentinel = MagicMock(name="fallback_active")
    monkeypatch.setattr(ens_mod, "_llm_for_provider", fake_build)
    monkeypatch.setattr(ens_mod, "get_llm", lambda streaming=False: sentinel)

    out = get_ensemble_llms()
    # All real builds fail, falls back to active llm
    assert len(out) == 1
    assert out[0][1] is sentinel


# ── aggregate_estimates edge case ───────────────────────────────────────────

def _est(prob, conf="medium"):
    return ResearchEstimate(
        ai_prob=prob,
        confidence=conf,
        reasoning="r",
        key_evidence=["a"],
        bull_case="b",
        bear_case="c",
    )


def test_aggregate_estimates_empty_raises():
    with pytest.raises(ValueError):
        aggregate_estimates([], disagreement_threshold=0.15)


def test_aggregate_estimates_anchor_uses_closest_to_median():
    estimates = [
        ("p1", _est(0.30)),
        ("p2", _est(0.55)),  # closest to median 0.55
        ("p3", _est(0.80)),
    ]
    agg = aggregate_estimates(estimates, disagreement_threshold=1.0)
    assert agg.ai_prob == 0.55
    assert agg.downgrade_confidence is False
    assert agg.confidence == "medium"


def test_aggregate_estimates_reasoning_includes_providers():
    estimates = [("groq", _est(0.4)), ("openai", _est(0.6))]
    agg = aggregate_estimates(estimates, disagreement_threshold=0.5)
    assert "groq" in agg.reasoning
    assert "openai" in agg.reasoning
    assert "median" in agg.reasoning


# ── run_ensemble_estimate ──────────────────────────────────────────────────

def _patch_llm_with_structured(monkeypatch, result_or_exc):
    """Make get_ensemble_llms return one LLM whose .with_structured_output
    returns an object whose .ainvoke yields result_or_exc."""

    estimator = MagicMock()
    if isinstance(result_or_exc, Exception):
        estimator.ainvoke = AsyncMock(side_effect=result_or_exc)
    else:
        estimator.ainvoke = AsyncMock(return_value=result_or_exc)

    llm = MagicMock()
    llm.with_structured_output = MagicMock(return_value=estimator)
    monkeypatch.setattr(ens_mod, "get_ensemble_llms", lambda: [("groq", llm)])
    return estimator


def test_run_ensemble_estimate_happy(monkeypatch, _settings_clear):
    _patch_llm_with_structured(monkeypatch, _est(0.7))
    result = asyncio.run(
        run_ensemble_estimate("sys", "ctx", disagreement_threshold=0.1)
    )
    assert result.ai_prob == 0.7
    assert result.providers == ["groq"]


def test_run_ensemble_estimate_all_fail_raises(monkeypatch, _settings_clear):
    _patch_llm_with_structured(monkeypatch, RuntimeError("provider down"))
    with pytest.raises(RuntimeError):
        asyncio.run(
            run_ensemble_estimate("sys", "ctx", disagreement_threshold=0.1)
        )


def test_run_ensemble_estimate_uses_default_threshold(monkeypatch, _settings_clear):
    monkeypatch.setenv("ENSEMBLE_DISAGREEMENT_THRESHOLD", "0.5")
    _patch_llm_with_structured(monkeypatch, _est(0.5))
    result = asyncio.run(run_ensemble_estimate("sys", "ctx"))
    assert result.ai_prob == 0.5


def test_run_ensemble_estimate_mixed_success_and_failure(monkeypatch, _settings_clear):
    good = MagicMock()
    good.ainvoke = AsyncMock(return_value=_est(0.6))
    good_llm = MagicMock()
    good_llm.with_structured_output = MagicMock(return_value=good)

    bad = MagicMock()
    bad.ainvoke = AsyncMock(side_effect=RuntimeError("nope"))
    bad_llm = MagicMock()
    bad_llm.with_structured_output = MagicMock(return_value=bad)

    monkeypatch.setattr(
        ens_mod, "get_ensemble_llms",
        lambda: [("groq", good_llm), ("nvidia", bad_llm)],
    )

    result = asyncio.run(
        run_ensemble_estimate("sys", "ctx", disagreement_threshold=0.1)
    )
    # only the good provider survived
    assert result.providers == ["groq"]
    assert result.ai_prob == 0.6
