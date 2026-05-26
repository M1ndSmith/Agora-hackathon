import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest


@pytest.fixture
def reset_client(monkeypatch):
    from agent.tools import polymarket

    monkeypatch.setattr(polymarket, "_client", None)
    yield
    monkeypatch.setattr(polymarket, "_client", None)


def _mock_client(payload):
    client = MagicMock()
    client.is_closed = False
    response = MagicMock()
    response.json = MagicMock(return_value=payload)
    response.raise_for_status = MagicMock(return_value=None)
    client.get = AsyncMock(return_value=response)
    client.aclose = AsyncMock(return_value=None)
    return client


def _make_market(**overrides):
    future = (
        (datetime.now(timezone.utc) + timedelta(days=10))
        .isoformat()
        .replace("+00:00", "Z")
    )
    base = {
        "id": "m1",
        "question": "Will X happen?",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.6", "0.4"]',
        "volumeNum": 50_000,
        "endDate": future,
        "slug": "x",
    }
    base.update(overrides)
    return base


def _patch_get_client(monkeypatch, fake):
    from agent.tools import polymarket

    async def fake_get_client():
        return fake

    monkeypatch.setattr(polymarket, "_get_client", fake_get_client)


def test_fetch_markets_happy(monkeypatch, reset_client):
    from agent.tools import polymarket

    _patch_get_client(monkeypatch, _mock_client([_make_market()]))
    markets = asyncio.run(polymarket.fetch_markets(min_volume=1000, limit=10))
    assert len(markets) == 1
    assert markets[0].id == "m1"
    assert markets[0].market_prob == 0.6


def test_fetch_markets_filters_low_volume(monkeypatch, reset_client):
    from agent.tools import polymarket

    _patch_get_client(monkeypatch, _mock_client([_make_market(volumeNum=100)]))
    markets = asyncio.run(polymarket.fetch_markets(min_volume=10_000, limit=10))
    assert markets == []


def test_fetch_markets_filters_non_binary(monkeypatch, reset_client):
    from agent.tools import polymarket

    raw = [_make_market(
        outcomes='["A","B","C"]', outcomePrices='["0.3","0.3","0.4"]'
    )]
    _patch_get_client(monkeypatch, _mock_client(raw))
    markets = asyncio.run(polymarket.fetch_markets(min_volume=100, limit=10))
    assert markets == []


def test_fetch_markets_filters_unparseable_prices(monkeypatch, reset_client):
    from agent.tools import polymarket

    raw = [_make_market(outcomePrices="not-json")]
    _patch_get_client(monkeypatch, _mock_client(raw))
    markets = asyncio.run(polymarket.fetch_markets(min_volume=100, limit=10))
    assert markets == []


def test_fetch_markets_filters_past_end_date(monkeypatch, reset_client):
    from agent.tools import polymarket

    past = (
        (datetime.now(timezone.utc) - timedelta(days=1))
        .isoformat()
        .replace("+00:00", "Z")
    )
    _patch_get_client(monkeypatch, _mock_client([_make_market(endDate=past)]))
    markets = asyncio.run(polymarket.fetch_markets(min_volume=100, limit=10))
    assert markets == []


def test_fetch_markets_filters_far_future_end_date(monkeypatch, reset_client):
    from agent.tools import polymarket

    far = (
        (datetime.now(timezone.utc) + timedelta(days=400))
        .isoformat()
        .replace("+00:00", "Z")
    )
    _patch_get_client(monkeypatch, _mock_client([_make_market(endDate=far)]))
    markets = asyncio.run(polymarket.fetch_markets(min_volume=100, limit=10))
    assert markets == []


def test_fetch_markets_skips_missing_end_date(monkeypatch, reset_client):
    from agent.tools import polymarket

    m = _make_market()
    m.pop("endDate", None)
    _patch_get_client(monkeypatch, _mock_client([m]))
    markets = asyncio.run(polymarket.fetch_markets(min_volume=100, limit=10))
    assert markets == []


def test_fetch_markets_skips_invalid_end_date(monkeypatch, reset_client):
    from agent.tools import polymarket

    _patch_get_client(monkeypatch, _mock_client([_make_market(endDate="not-a-date")]))
    markets = asyncio.run(polymarket.fetch_markets(min_volume=100, limit=10))
    assert markets == []


def test_get_market_history_returns_payload(monkeypatch, reset_client):
    from agent.tools import polymarket

    payload = {"id": "m1", "closed": False}
    _patch_get_client(monkeypatch, _mock_client(payload))
    data = asyncio.run(polymarket.get_market_history("m1"))
    assert data["id"] == "m1"


def test_get_market_history_returns_error_on_exception(monkeypatch, reset_client):
    from agent.tools import polymarket

    fake = MagicMock()
    fake.is_closed = False
    fake.get = AsyncMock(side_effect=RuntimeError("boom"))
    _patch_get_client(monkeypatch, fake)
    data = asyncio.run(polymarket.get_market_history("m99"))
    assert "error" in data
    assert data["market_id"] == "m99"


def test_fetch_resolved_market_still_open(monkeypatch, reset_client):
    from agent.tools import polymarket

    payload = {"closed": False, "outcomePrices": '["0.5","0.5"]'}
    _patch_get_client(monkeypatch, _mock_client(payload))
    resolved, outcome = asyncio.run(polymarket.fetch_resolved_market("m1"))
    assert resolved is False
    assert outcome is None


def test_fetch_resolved_market_yes(monkeypatch, reset_client):
    from agent.tools import polymarket

    payload = {"closed": True, "outcomePrices": '["1.0","0.0"]'}
    _patch_get_client(monkeypatch, _mock_client(payload))
    resolved, outcome = asyncio.run(polymarket.fetch_resolved_market("m1"))
    assert resolved is True
    assert outcome == "yes"


def test_fetch_resolved_market_no_via_list_prices(monkeypatch, reset_client):
    from agent.tools import polymarket

    payload = {"closed": True, "outcomePrices": [0.0, 1.0]}
    _patch_get_client(monkeypatch, _mock_client(payload))
    resolved, outcome = asyncio.run(polymarket.fetch_resolved_market("m1"))
    assert resolved is True
    assert outcome == "no"


def test_fetch_resolved_market_invalid_prices(monkeypatch, reset_client):
    from agent.tools import polymarket

    payload = {"closed": True, "outcomePrices": "not-json"}
    _patch_get_client(monkeypatch, _mock_client(payload))
    resolved, outcome = asyncio.run(polymarket.fetch_resolved_market("m1"))
    assert resolved is False
    assert outcome is None


def test_fetch_resolved_market_wrong_outcome_count(monkeypatch, reset_client):
    from agent.tools import polymarket

    payload = {"closed": True, "outcomePrices": '["0.3","0.3","0.4"]'}
    _patch_get_client(monkeypatch, _mock_client(payload))
    resolved, outcome = asyncio.run(polymarket.fetch_resolved_market("m1"))
    assert resolved is False
    assert outcome is None


def test_fetch_resolved_market_error_payload(monkeypatch, reset_client):
    from agent.tools import polymarket

    async def err_history(_mid):
        return {"error": "down", "market_id": _mid}

    monkeypatch.setattr(polymarket, "get_market_history", err_history)
    resolved, outcome = asyncio.run(polymarket.fetch_resolved_market("m1"))
    assert resolved is False
    assert outcome is None


def test_close_client_clears_singleton(reset_client):
    from agent.tools import polymarket

    fake = MagicMock()
    fake.is_closed = False
    fake.aclose = AsyncMock(return_value=None)
    polymarket._client = fake
    asyncio.run(polymarket.close_client())
    assert polymarket._client is None
