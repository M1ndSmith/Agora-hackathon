import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture
def _settings_clear():
    from config import get_settings

    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# Use an all-1s address so EIP-55 checksum is unambiguous lowercase.
AGENT_ADDR = "0x" + "1" * 40


def test_verify_payment_no_agent_address(monkeypatch, _settings_clear):
    monkeypatch.setenv("AGENT_ADDRESS", "")  # override any .env value
    from config import get_settings

    get_settings.cache_clear()
    from onchain.x402 import verify_payment

    assert asyncio.run(verify_payment("0x" + "a" * 40)) is None


def test_verify_payment_arc_disabled(monkeypatch, _settings_clear):
    monkeypatch.setenv("AGENT_ADDRESS", AGENT_ADDR)
    monkeypatch.setenv("ARC_ENABLED", "false")
    from config import get_settings

    get_settings.cache_clear()
    from onchain.x402 import verify_payment

    assert asyncio.run(verify_payment("0x" + "a" * 40)) is None


def test_verify_payment_by_hash_no_agent_address(monkeypatch, _settings_clear):
    monkeypatch.setenv("AGENT_ADDRESS", "")  # override any .env value
    from config import get_settings

    get_settings.cache_clear()
    from onchain.x402 import verify_payment_by_hash

    assert asyncio.run(verify_payment_by_hash("0x" + "b" * 64)) is False


def test_verify_payment_by_hash_arc_disabled(monkeypatch, _settings_clear):
    monkeypatch.setenv("AGENT_ADDRESS", AGENT_ADDR)
    monkeypatch.setenv("ARC_ENABLED", "false")
    from config import get_settings

    get_settings.cache_clear()
    from onchain.x402 import verify_payment_by_hash

    assert asyncio.run(verify_payment_by_hash("0x" + "b" * 64)) is False


def _patched_async_web3(monkeypatch, fake_w3):
    """Patch onchain.x402.AsyncWeb3 so AsyncWeb3() returns fake_w3 but classmethods work."""
    from web3 import AsyncWeb3 as Real
    from onchain import x402

    mock_cls = MagicMock(return_value=fake_w3)
    mock_cls.to_checksum_address = Real.to_checksum_address
    monkeypatch.setattr(x402, "AsyncWeb3", mock_cls)
    return mock_cls


def _fake_w3_skeleton():
    fake = MagicMock()
    fake.middleware_onion = MagicMock()
    fake.middleware_onion.inject = MagicMock(return_value=None)
    fake.provider = MagicMock()
    fake.provider.disconnect = AsyncMock(return_value=None)
    fake.eth = MagicMock()
    return fake


def test_verify_payment_by_hash_no_receipt(monkeypatch, _settings_clear):
    monkeypatch.setenv("AGENT_ADDRESS", AGENT_ADDR)
    monkeypatch.setenv("ARC_ENABLED", "true")
    from config import get_settings

    get_settings.cache_clear()

    from onchain import x402

    fake = _fake_w3_skeleton()
    fake.eth.get_transaction_receipt = AsyncMock(return_value=None)
    _patched_async_web3(monkeypatch, fake)

    # Hash without 0x prefix exercises normalisation path
    result = asyncio.run(x402.verify_payment_by_hash("deadbeef" + "0" * 56))
    assert result is False
    # Confirm prefix was added before lookup
    called_with = fake.eth.get_transaction_receipt.call_args.args[0]
    assert called_with.startswith("0x")


def test_verify_payment_by_hash_failed_status(monkeypatch, _settings_clear):
    monkeypatch.setenv("AGENT_ADDRESS", AGENT_ADDR)
    from config import get_settings

    get_settings.cache_clear()

    from onchain import x402

    fake = _fake_w3_skeleton()
    fake.eth.get_transaction_receipt = AsyncMock(return_value={"status": 0})
    _patched_async_web3(monkeypatch, fake)

    assert asyncio.run(x402.verify_payment_by_hash("0x" + "c" * 64)) is False


def test_verify_payment_by_hash_success(monkeypatch, _settings_clear):
    monkeypatch.setenv("AGENT_ADDRESS", AGENT_ADDR)
    monkeypatch.setenv("X402_PRICE_USDC", "0.01")
    from config import get_settings

    get_settings.cache_clear()

    from onchain import x402

    fake = _fake_w3_skeleton()
    fake.eth.get_transaction_receipt = AsyncMock(
        return_value={"status": 1, "logs": []}
    )

    decimals_call = MagicMock()
    decimals_call.call = AsyncMock(return_value=6)
    decimals_fn = MagicMock(return_value=decimals_call)

    transfer_event = MagicMock()
    transfer_event.process_receipt = MagicMock(
        return_value=[
            # Matches AGENT_ADDR; value 10_000 == 0.01 * 10^6
            {"args": {"to": AGENT_ADDR, "value": 10_000}}
        ]
    )
    transfer_callable = MagicMock(return_value=transfer_event)

    contract = MagicMock()
    contract.functions = MagicMock()
    contract.functions.decimals = decimals_fn
    contract.events = MagicMock()
    contract.events.Transfer = transfer_callable
    fake.eth.contract = MagicMock(return_value=contract)

    _patched_async_web3(monkeypatch, fake)

    assert asyncio.run(x402.verify_payment_by_hash("0x" + "d" * 64)) is True


def test_verify_payment_by_hash_no_matching_log(monkeypatch, _settings_clear):
    monkeypatch.setenv("AGENT_ADDRESS", AGENT_ADDR)
    from config import get_settings

    get_settings.cache_clear()

    from onchain import x402

    fake = _fake_w3_skeleton()
    fake.eth.get_transaction_receipt = AsyncMock(
        return_value={"status": 1, "logs": []}
    )

    decimals_call = MagicMock()
    decimals_call.call = AsyncMock(return_value=6)
    contract = MagicMock()
    contract.functions = MagicMock()
    contract.functions.decimals = MagicMock(return_value=decimals_call)
    transfer_event = MagicMock()
    transfer_event.process_receipt = MagicMock(return_value=[])  # no logs
    contract.events = MagicMock()
    contract.events.Transfer = MagicMock(return_value=transfer_event)
    fake.eth.contract = MagicMock(return_value=contract)

    _patched_async_web3(monkeypatch, fake)

    assert asyncio.run(x402.verify_payment_by_hash("0x" + "e" * 64)) is False


def test_verify_payment_no_logs_returns_none(monkeypatch, _settings_clear):
    monkeypatch.setenv("AGENT_ADDRESS", AGENT_ADDR)
    monkeypatch.setenv("X402_PRICE_USDC", "0.01")
    monkeypatch.setenv("X402_SCAN_BLOCKS", "10")
    monkeypatch.setenv("X402_SCAN_CHUNK", "5")
    from config import get_settings

    get_settings.cache_clear()

    from onchain import x402

    fake = _fake_w3_skeleton()

    async def _block():
        return 50

    fake.eth.block_number = _block()
    fake.eth.get_logs = AsyncMock(return_value=[])

    decimals_call = MagicMock()
    decimals_call.call = AsyncMock(return_value=6)
    contract = MagicMock()
    contract.functions = MagicMock()
    contract.functions.decimals = MagicMock(return_value=decimals_call)
    fake.eth.contract = MagicMock(return_value=contract)

    keccak_result = MagicMock()
    keccak_result.hex = MagicMock(return_value="abc123")
    fake.keccak = MagicMock(return_value=keccak_result)

    _patched_async_web3(monkeypatch, fake)

    assert asyncio.run(x402.verify_payment("0x" + "f" * 40)) is None
    fake.eth.get_logs.assert_awaited()


def test_verify_payment_get_logs_error_continues(monkeypatch, _settings_clear):
    """Chunk error should be logged and scan continues; returns None when no luck."""
    monkeypatch.setenv("AGENT_ADDRESS", AGENT_ADDR)
    monkeypatch.setenv("X402_PRICE_USDC", "0.01")
    monkeypatch.setenv("X402_SCAN_BLOCKS", "10")
    monkeypatch.setenv("X402_SCAN_CHUNK", "5")
    from config import get_settings

    get_settings.cache_clear()

    from onchain import x402

    fake = _fake_w3_skeleton()

    async def _block():
        return 50

    fake.eth.block_number = _block()
    fake.eth.get_logs = AsyncMock(side_effect=RuntimeError("rpc down"))

    decimals_call = MagicMock()
    decimals_call.call = AsyncMock(return_value=6)
    contract = MagicMock()
    contract.functions = MagicMock()
    contract.functions.decimals = MagicMock(return_value=decimals_call)
    fake.eth.contract = MagicMock(return_value=contract)

    keccak_result = MagicMock()
    keccak_result.hex = MagicMock(return_value="abc123")
    fake.keccak = MagicMock(return_value=keccak_result)

    _patched_async_web3(monkeypatch, fake)

    assert asyncio.run(x402.verify_payment("0x" + "f" * 40)) is None
