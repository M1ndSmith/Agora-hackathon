import os

import pytest


@pytest.fixture(autouse=True)
def _minimal_env(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test")
    monkeypatch.setenv("TAVILY_API_KEY", "tvly_test")
    monkeypatch.setenv("AGENT_ADDRESS", "0x1234567890123456789012345678901234567890")
    monkeypatch.setenv("X402_PRICE_USDC", "0.01")
    from config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_get_payment_details_keys():
    from onchain.x402 import get_payment_details

    details = get_payment_details()
    for key in ("scheme", "network", "to", "amount_usdc", "currency"):
        assert key in details
    assert details["scheme"] == "exact"
    assert details["network"] == "arc-testnet"


def test_get_explorer_url_arc_tx():
    from onchain.wallet import get_explorer_url

    url = get_explorer_url("0x" + "a" * 64)
    assert "explorer.arc.io" in url
    assert url.endswith("a" * 64)


def test_get_explorer_url_circle_prefix():
    from onchain.wallet import get_explorer_url

    url = get_explorer_url("circle:tx-id-12345")
    assert "app.circle.com" in url
    assert "tx-id-12345" in url
