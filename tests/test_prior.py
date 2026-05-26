import asyncio


def test_build_prior_returns_none_when_empty(monkeypatch):
    from agent.tools import prior as prior_mod

    async def fake_get(_mid):
        return None

    monkeypatch.setattr(
        prior_mod.store, "get_latest_unresolved_pick_for_market", fake_get
    )
    result = asyncio.run(prior_mod.build_prior("m99"))
    assert result is None


def test_build_prior_happy(monkeypatch):
    from agent.tools import prior as prior_mod

    async def fake_get(_mid):
        return {
            "id": 1,
            "ai_prob": 0.62,
            "reasoning_trace": "Prior reasoning " * 20,
        }

    monkeypatch.setattr(
        prior_mod.store, "get_latest_unresolved_pick_for_market", fake_get
    )
    result = asyncio.run(prior_mod.build_prior("m1"))
    assert result["ai_prob"] == 0.62
    assert "Prior reasoning" in result["summary"]
    assert len(result["summary"]) <= 303
