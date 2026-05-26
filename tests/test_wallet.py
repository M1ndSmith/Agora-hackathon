import asyncio
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def _settings_clear():
    from config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_write_env_vars_fresh(tmp_path):
    from onchain.wallet import _write_env_vars

    env = tmp_path / ".env"
    _write_env_vars(str(env), {"FOO": "bar", "BAZ": "qux"})
    content = env.read_text()
    assert "FOO=bar" in content
    assert "BAZ=qux" in content


def test_write_env_vars_replace_existing(tmp_path):
    from onchain.wallet import _write_env_vars

    env = tmp_path / ".env"
    env.write_text("FOO=old\nOTHER=keep\n")
    _write_env_vars(str(env), {"FOO": "new", "NEW": "added"})
    content = env.read_text()
    assert "FOO=new" in content
    assert "FOO=old" not in content
    assert "OTHER=keep" in content
    assert "NEW=added" in content


def test_init_web3_wallet_creates_keypair(tmp_path):
    from onchain.wallet import _init_web3_wallet

    env = tmp_path / ".env"
    info = _init_web3_wallet(str(env))
    assert info["source"] == "web3"
    assert info["address"].startswith("0x")
    assert len(info["address"]) == 42
    content = env.read_text()
    assert info["address"] in content
    assert "AGENT_PRIVATE_KEY=" in content


def test_init_wallet_uses_web3_when_no_circle(tmp_path, monkeypatch, _settings_clear):
    for k in (
        "CIRCLE_API_KEY",
        "CIRCLE_ENTITY_SECRET",
        "CIRCLE_WALLET_SET_ID",
        "CIRCLE_WALLET_ID",
    ):
        monkeypatch.delenv(k, raising=False)
    from config import get_settings

    get_settings.cache_clear()
    from onchain.wallet import init_wallet

    env = tmp_path / ".env"
    info = init_wallet(str(env))
    assert info["source"] == "web3"
    assert info["address"].startswith("0x")


def test_get_w3_returns_async_web3(_settings_clear):
    from onchain.wallet import _get_w3
    from web3 import AsyncWeb3

    w3 = _get_w3()
    assert isinstance(w3, AsyncWeb3)


def test_get_balance_no_address_returns_zero(monkeypatch, _settings_clear):
    # Empty strings override values pydantic-settings would load from .env
    monkeypatch.setenv("AGENT_ADDRESS", "")
    monkeypatch.setenv("CIRCLE_API_KEY", "")
    monkeypatch.setenv("CIRCLE_WALLET_ID", "")
    from config import get_settings

    get_settings.cache_clear()
    from onchain.wallet import get_balance

    assert asyncio.run(get_balance("")) == 0.0


def test_get_balance_arc_disabled_returns_zero(monkeypatch, _settings_clear):
    monkeypatch.setenv("AGENT_ADDRESS", "0x" + "1" * 40)
    monkeypatch.setenv("ARC_ENABLED", "false")
    monkeypatch.delenv("CIRCLE_API_KEY", raising=False)
    from config import get_settings

    get_settings.cache_clear()
    from onchain.wallet import get_balance

    assert asyncio.run(get_balance()) == 0.0


def test_get_balance_uses_circle_when_positive(monkeypatch, _settings_clear):
    monkeypatch.setenv("CIRCLE_API_KEY", "TEST_API_KEY:xx")
    monkeypatch.setenv("CIRCLE_WALLET_ID", "w1")
    monkeypatch.setenv("AGENT_ADDRESS", "0x" + "1" * 40)
    from config import get_settings

    get_settings.cache_clear()

    from onchain import circle_wallet

    async def fake_balance(*, wallet_id, api_key):
        assert wallet_id == "w1"
        return 42.5

    monkeypatch.setattr(circle_wallet, "get_circle_token_balance", fake_balance)

    from onchain.wallet import get_balance

    assert asyncio.run(get_balance()) == 42.5


def test_get_balance_circle_error_falls_through(monkeypatch, _settings_clear):
    monkeypatch.setenv("CIRCLE_API_KEY", "TEST_API_KEY:xx")
    monkeypatch.setenv("CIRCLE_WALLET_ID", "w1")
    monkeypatch.setenv("AGENT_ADDRESS", "")
    from config import get_settings

    get_settings.cache_clear()

    from onchain import circle_wallet

    async def fake_balance(*, wallet_id, api_key):
        raise RuntimeError("circle down")

    monkeypatch.setattr(circle_wallet, "get_circle_token_balance", fake_balance)

    from onchain.wallet import get_balance

    # No agent_address after fall-through → 0.0
    assert asyncio.run(get_balance("")) == 0.0


def test_send_proof_tx_returns_mock_when_arc_disabled(monkeypatch, _settings_clear):
    monkeypatch.setenv("ARC_ENABLED", "false")
    from config import get_settings

    get_settings.cache_clear()
    from onchain.wallet import MOCK_TX_HASH, send_proof_tx

    assert asyncio.run(send_proof_tx()) == MOCK_TX_HASH


def test_send_proof_tx_returns_mock_when_no_key(monkeypatch, _settings_clear):
    monkeypatch.setenv("ARC_ENABLED", "true")
    monkeypatch.setenv("AGENT_ADDRESS", "0x" + "1" * 40)
    monkeypatch.setenv("AGENT_PRIVATE_KEY", "")  # override .env
    from config import get_settings

    get_settings.cache_clear()
    from onchain.wallet import MOCK_TX_HASH, send_proof_tx

    assert asyncio.run(send_proof_tx()) == MOCK_TX_HASH


def test_send_proof_tx_returns_mock_when_no_address(monkeypatch, _settings_clear):
    monkeypatch.setenv("ARC_ENABLED", "true")
    monkeypatch.setenv("AGENT_PRIVATE_KEY", "")  # override .env
    monkeypatch.setenv("AGENT_ADDRESS", "")
    from config import get_settings

    get_settings.cache_clear()
    from onchain.wallet import MOCK_TX_HASH, send_proof_tx

    assert asyncio.run(send_proof_tx()) == MOCK_TX_HASH
