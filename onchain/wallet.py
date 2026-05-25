"""
Arc testnet wallet integration using web3.py AsyncWeb3.

Sends a minimal USDC transfer as onchain proof of each agent pick.
Feature-flagged: if ARC_ENABLED=false or Arc is unreachable, returns a
mock tx hash so the rest of the pipeline still functions.
"""
import logging
import os
from pathlib import Path
from typing import Optional

from web3 import AsyncWeb3, AsyncHTTPProvider
from web3.middleware import ExtraDataToPOAMiddleware

from config import get_settings

logger = logging.getLogger(__name__)

# Minimal ERC-20 ABI — only balanceOf + transfer needed
ERC20_ABI = [
    {
        "constant": True,
        "inputs": [{"name": "_owner", "type": "address"}],
        "name": "balanceOf",
        "outputs": [{"name": "balance", "type": "uint256"}],
        "type": "function",
    },
    {
        "constant": False,
        "inputs": [
            {"name": "_to", "type": "address"},
            {"name": "_value", "type": "uint256"},
        ],
        "name": "transfer",
        "outputs": [{"name": "", "type": "bool"}],
        "type": "function",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
]

MOCK_TX_HASH = "0x" + "0" * 64
ARC_EXPLORER_BASE = "https://explorer.arc.io/tx"


def _get_w3() -> AsyncWeb3:
    settings = get_settings()
    w3 = AsyncWeb3(AsyncHTTPProvider(settings.arc_rpc))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    return w3


CIRCLE_DASHBOARD_URL = "https://app.circle.com/w3s/transactions"

def get_explorer_url(tx_hash: str) -> str:
    if tx_hash.startswith("circle:"):
        # Circle transaction — link to Circle dashboard instead of Arc explorer
        circle_id = tx_hash[len("circle:"):]
        return f"{CIRCLE_DASHBOARD_URL}/{circle_id}"
    return f"{ARC_EXPLORER_BASE}/{tx_hash}"


async def get_balance(address: Optional[str] = None) -> float:
    """
    Return Arc testnet USDC balance for the agent wallet.

    If Circle is configured, tries Circle's balance API first (no RPC needed).
    Falls back to direct web3 query. Returns 0.0 on any error.
    """
    settings = get_settings()

    # Circle path — no RPC call needed, balance comes from Circle API
    if settings.circle_api_key and settings.circle_wallet_id:
        try:
            from onchain.circle_wallet import get_circle_token_balance
            balance = await get_circle_token_balance(
                wallet_id=settings.circle_wallet_id,
                api_key=settings.circle_api_key,
            )
            if balance > 0:
                return balance
            # Circle returned 0 — may be Arc testnet not listed there yet,
            # fall through to direct web3 query below
        except Exception as e:
            logger.debug(f"Circle balance lookup failed, using web3: {e}")

    address = address or settings.agent_address
    if not address or not settings.arc_enabled:
        return 0.0

    w3 = _get_w3()
    try:
        checksum_addr = AsyncWeb3.to_checksum_address(address)
        usdc = w3.eth.contract(
            address=AsyncWeb3.to_checksum_address(settings.arc_usdc_contract),
            abi=ERC20_ABI,
        )
        decimals = await usdc.functions.decimals().call()
        raw = await usdc.functions.balanceOf(checksum_addr).call()
        return raw / (10**decimals)
    except Exception:
        return 0.0
    finally:
        await w3.provider.disconnect()


async def send_proof_tx(
    amount_usdc: float = 0.01,
    to_address: Optional[str] = None,
) -> str:
    """
    Send a minimal USDC transfer on Arc testnet as proof of agent action.

    Uses web3.py + AGENT_PRIVATE_KEY to sign directly on Arc testnet.
    Circle Programmable Wallets are not used here because Circle does not
    support Arc testnet as a chain for on-chain transfers.

    Falls back to MOCK_TX_HASH if ARC_ENABLED=false, key is missing, or
    the RPC is unreachable.
    """
    settings = get_settings()

    if not settings.arc_enabled:
        return MOCK_TX_HASH

    recipient = to_address or settings.agent_address

    if not settings.agent_private_key or not settings.agent_address:
        return MOCK_TX_HASH

    w3 = _get_w3()
    try:
        account = w3.eth.account.from_key(settings.agent_private_key)
        checksum_from = AsyncWeb3.to_checksum_address(account.address)
        checksum_to = AsyncWeb3.to_checksum_address(recipient)

        usdc = w3.eth.contract(
            address=AsyncWeb3.to_checksum_address(settings.arc_usdc_contract),
            abi=ERC20_ABI,
        )

        decimals = await usdc.functions.decimals().call()
        amount_raw = int(amount_usdc * (10**decimals))

        nonce = await w3.eth.get_transaction_count(checksum_from)
        gas_price = await w3.eth.gas_price

        tx = await usdc.functions.transfer(checksum_to, amount_raw).build_transaction(
            {
                "from": checksum_from,
                "nonce": nonce,
                "gasPrice": gas_price,
            }
        )

        signed = account.sign_transaction(tx)
        tx_hash_bytes = await w3.eth.send_raw_transaction(signed.raw_transaction)
        return tx_hash_bytes.hex()

    except Exception as e:
        logger.warning(f"web3 proof tx failed: {e}")
        return MOCK_TX_HASH
    finally:
        await w3.provider.disconnect()


def init_wallet(env_path: str = ".env") -> dict:
    """
    Generate or provision the agent wallet.

    Priority:
    1. Circle Programmable Wallets API — if CIRCLE_API_KEY + CIRCLE_WALLET_SET_ID are set
       Creates a Circle developer-controlled wallet (key managed by Circle HSM).
       Writes AGENT_ADDRESS + CIRCLE_WALLET_ID to .env.

    2. Raw web3 keypair — fallback when Circle is not configured.
       Writes AGENT_ADDRESS + AGENT_PRIVATE_KEY to .env.

    Returns {"address": ..., "source": "circle"|"web3", ...}.
    """
    settings = get_settings()

    if (
        settings.circle_api_key
        and settings.circle_wallet_set_id
        and settings.circle_entity_secret
    ):
        return _init_circle_wallet(env_path, settings)

    return _init_web3_wallet(env_path)


def _init_circle_wallet(env_path: str, settings) -> dict:
    """Create a Circle developer-controlled wallet and write address to .env."""
    import asyncio
    from onchain.circle_wallet import create_circle_wallet, CircleAPIError, CircleNotConfiguredError

    try:
        wallet_info = asyncio.run(
            create_circle_wallet(
                api_key=settings.circle_api_key,
                entity_secret=settings.circle_entity_secret,
                wallet_set_id=settings.circle_wallet_set_id,
                name="agora-agent",
            )
        )
        address = wallet_info["address"]
        wallet_id = wallet_info["wallet_id"]

        _write_env_vars(env_path, {
            "AGENT_ADDRESS": address,
            "CIRCLE_WALLET_ID": wallet_id,
        })

        return {
            "address": address,
            "wallet_id": wallet_id,
            "source": "circle",
        }
    except Exception as e:
        logger.warning(f"Circle wallet creation failed ({e}), falling back to web3")
        return _init_web3_wallet(env_path)


def _init_web3_wallet(env_path: str) -> dict:
    """Generate a raw EVM keypair and write to .env."""
    from web3 import Web3

    account = Web3().eth.account.create()
    address = account.address
    private_key = account.key.hex()

    _write_env_vars(env_path, {
        "AGENT_ADDRESS": address,
        "AGENT_PRIVATE_KEY": private_key,
    })

    return {"address": address, "private_key": private_key, "source": "web3"}


def _write_env_vars(env_path: str, vars_dict: dict) -> None:
    """Write or update key=value pairs in an .env file."""
    env_file = Path(env_path)
    if env_file.exists():
        lines = env_file.read_text().splitlines()
        replaced = set()
        new_lines = []
        for line in lines:
            key = line.split("=")[0] if "=" in line else None
            if key and key in vars_dict:
                new_lines.append(f"{key}={vars_dict[key]}")
                replaced.add(key)
            else:
                new_lines.append(line)
        for key, value in vars_dict.items():
            if key not in replaced:
                new_lines.append(f"{key}={value}")
        env_file.write_text("\n".join(new_lines) + "\n")
    else:
        with Path(env_path).open("w") as f:
            for key, value in vars_dict.items():
                f.write(f"{key}={value}\n")
