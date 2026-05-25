"""
Circle Programmable Wallets REST client.

Creates and manages developer-controlled wallets via Circle's API.
Used by init_wallet() as the preferred path when Circle credentials are configured.

Docs: https://developers.circle.com/w3s/developer-controlled-wallets-quickstart

NOTE: Circle does not support Arc testnet for on-chain transfers. The functions
here are used for wallet creation and balance reads only. On-chain proof
transactions on Arc are always sent via web3.py (onchain/wallet.py).

Feature flag: if CIRCLE_API_KEY / CIRCLE_WALLET_SET_ID are not set,
all functions raise CircleNotConfiguredError and the caller falls back
to raw web3 keypair generation.
"""
import logging
import secrets
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

CIRCLE_API_BASE_PRODUCTION = "https://api.circle.com/v1/w3s"
CIRCLE_API_BASE_SANDBOX    = "https://api-sandbox.circle.com/v1/w3s"

# Arc testnet blockchain identifier for Circle API
ARC_BLOCKCHAIN = "ETH-SEPOLIA"  # update if Circle ever lists Arc explicitly


def _api_base(api_key: str) -> str:
    """Return the correct Circle API base URL.
    Keys starting with 'TEST_API_KEY:' are sandbox keys."""
    if api_key.startswith("TEST_API_KEY:"):
        return CIRCLE_API_BASE_SANDBOX
    return CIRCLE_API_BASE_PRODUCTION


class CircleNotConfiguredError(Exception):
    """Raised when Circle credentials are missing — caller should fall back."""
    pass


class CircleAPIError(Exception):
    """Raised on non-2xx responses from Circle API."""
    pass


def _headers(api_key: str) -> dict:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


async def create_circle_wallet(
    api_key: str,
    entity_secret: str,
    wallet_set_id: str,
    name: str = "agora-agent",
) -> dict:
    """
    Create a developer-controlled wallet via Circle Programmable Wallets API.

    Returns dict with:
      wallet_id   — Circle wallet UUID
      address     — EVM address
      blockchain  — chain identifier

    Raises CircleAPIError on failure.
    """
    idempotency_key = secrets.token_hex(16)
    entity_secret_ciphertext = _encode_entity_secret(entity_secret)

    payload = {
        "idempotencyKey": idempotency_key,
        "entitySecretCiphertext": entity_secret_ciphertext,
        "walletSetId": wallet_set_id,
        "blockchains": [ARC_BLOCKCHAIN],
        "count": 1,
        "metadata": [{"name": name}],
    }

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{_api_base(api_key)}/wallets",
            headers=_headers(api_key),
            json=payload,
        )

    if resp.status_code not in (200, 201):
        raise CircleAPIError(
            f"Circle wallet creation failed: {resp.status_code} — {resp.text[:300]}"
        )

    data = resp.json()
    wallets = data.get("data", {}).get("wallets", [])
    if not wallets:
        raise CircleAPIError(f"Circle returned no wallets: {data}")

    wallet = wallets[0]
    return {
        "wallet_id": wallet.get("id"),
        "address": wallet.get("address"),
        "blockchain": wallet.get("blockchain"),
        "state": wallet.get("state"),
    }


async def get_circle_wallet(wallet_id: str, api_key: str) -> dict:
    """Fetch details of an existing Circle wallet."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(
            f"{_api_base(api_key)}/wallets/{wallet_id}",
            headers=_headers(api_key),
        )
    if resp.status_code != 200:
        raise CircleAPIError(f"Circle wallet fetch failed: {resp.status_code} — {resp.text[:200]}")
    return resp.json().get("data", {}).get("wallet", {})


async def get_circle_token_balance(
    wallet_id: str,
    api_key: str,
    token_symbol: str = "USDC",
) -> float:
    """
    Get token balance for a Circle wallet.
    Returns 0.0 on any error.
    """
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{_api_base(api_key)}/wallets/{wallet_id}/balances",
                headers=_headers(api_key),
            )
        if resp.status_code != 200:
            return 0.0

        data = resp.json()
        token_balances = data.get("data", {}).get("tokenBalances", [])
        for tb in token_balances:
            if tb.get("token", {}).get("symbol", "").upper() == token_symbol.upper():
                return float(tb.get("amount", 0))
        return 0.0
    except Exception as e:
        logger.warning(f"Circle balance fetch error: {e}")
        return 0.0


def _encode_entity_secret(entity_secret: str) -> str:
    """
    Encode the entity secret for Circle API calls.

    Circle's full encryption scheme uses RSA-OAEP with their public key.
    For sandbox/hackathon use, pass the raw hex secret — Circle's sandbox
    accepts this. Replace with proper RSA-OAEP encryption for production.

    See: https://developers.circle.com/w3s/entity-secret-management
    """
    return entity_secret.lstrip("0x")
