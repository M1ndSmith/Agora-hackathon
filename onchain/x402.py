"""
x402 Payment Protocol — Agora implementation.

Implements the x402 micro-payment handshake for unlocking AI reasoning traces.
No smart contract required — verification is a Transfer event log scan on Arc testnet.

Flow:
  1. UI shows blurred trace + payment details (402 equivalent)
  2. User sends 0.01 USDC to agent_address on Arc testnet
  3. UI calls verify_payment(from_address, pick_id)
  4. We scan ERC-20 Transfer logs to confirm the payment
  5. Receipt (tx_hash) stored in SQLite; trace revealed in UI

Narrative for judges:
  "The agent charges for access to its own intelligence.
   Every reasoning trace is a micro-paywalled asset, settled onchain."
"""
import json
import logging
from typing import Optional

from web3 import AsyncWeb3, AsyncHTTPProvider
from web3.middleware import ExtraDataToPOAMiddleware

from config import get_settings

logger = logging.getLogger(__name__)

# ERC-20 Transfer event ABI — needed for log scanning
ERC20_TRANSFER_ABI = [
    {
        "anonymous": False,
        "inputs": [
            {"indexed": True, "name": "from", "type": "address"},
            {"indexed": True, "name": "to", "type": "address"},
            {"indexed": False, "name": "value", "type": "uint256"},
        ],
        "name": "Transfer",
        "type": "event",
    },
    {
        "inputs": [],
        "name": "decimals",
        "outputs": [{"name": "", "type": "uint8"}],
        "stateMutability": "view",
        "type": "function",
    },
]

# Default fallback if config not loaded. Real value comes from settings.x402_scan_blocks.
SCAN_BLOCKS = 50_000


def get_payment_details() -> dict:
    """
    Return x402-style payment requirements for a reasoning trace unlock.
    This is the '402 Payment Required' response equivalent.
    """
    settings = get_settings()
    return {
        "scheme": "exact",
        "network": "arc-testnet",
        "to": settings.agent_address or "0x0000000000000000000000000000000000000000",
        "amount_usdc": settings.x402_price_usdc,
        "currency": "USDC",
        "description": "Unlock Agora AI reasoning trace",
        "explorer": "https://explorer.arc.io",
    }


async def verify_payment(
    from_address: str,
    min_amount_usdc: Optional[float] = None,
) -> Optional[str]:
    """
    Verify that from_address sent >= min_amount_usdc USDC to the agent wallet
    on Arc testnet within the last SCAN_BLOCKS blocks.

    Returns the tx_hash if payment found, None otherwise.

    This is the server-side x402 receipt verification step.
    Falls back gracefully if Arc is unreachable.
    """
    settings = get_settings()
    agent_address = settings.agent_address
    if not agent_address or not settings.arc_enabled:
        return None

    amount = min_amount_usdc or settings.x402_price_usdc

    w3 = AsyncWeb3(AsyncHTTPProvider(settings.arc_rpc))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    try:
        usdc = w3.eth.contract(
            address=AsyncWeb3.to_checksum_address(settings.arc_usdc_contract),
            abi=ERC20_TRANSFER_ABI,
        )

        decimals_raw = await usdc.functions.decimals().call()
        min_raw = int(amount * (10 ** decimals_raw))

        checksum_from = AsyncWeb3.to_checksum_address(from_address)
        checksum_to = AsyncWeb3.to_checksum_address(agent_address)
        usdc_addr = AsyncWeb3.to_checksum_address(settings.arc_usdc_contract)

        latest_block = await w3.eth.block_number
        total_lookback = settings.x402_scan_blocks or SCAN_BLOCKS
        chunk_size = max(1_000, settings.x402_scan_chunk or 5_000)
        oldest_block = max(0, latest_block - total_lookback)

        # Topics for ERC-20 Transfer(address,address,uint256)
        transfer_sig = "0x" + w3.keccak(
            text="Transfer(address,address,uint256)"
        ).hex().lstrip("0x")
        from_topic = "0x" + checksum_from[2:].lower().rjust(64, "0")
        to_topic = "0x" + checksum_to[2:].lower().rjust(64, "0")

        # Scan newest → oldest in chunks so we can stop early on a hit
        # and avoid RPC range limits.
        end = latest_block
        while end >= oldest_block:
            start = max(oldest_block, end - chunk_size + 1)
            try:
                logs = await w3.eth.get_logs({
                    "fromBlock": start,
                    "toBlock": end,
                    "address": usdc_addr,
                    "topics": [transfer_sig, from_topic, to_topic],
                })
            except Exception as chunk_err:
                logger.warning(
                    f"x402 chunk {start}-{end} failed: {chunk_err}; skipping"
                )
                end = start - 1
                continue

            for raw_log in reversed(logs):  # newest within chunk first
                try:
                    parsed = usdc.events.Transfer().process_log(raw_log)
                except Exception:
                    continue
                value = parsed["args"]["value"]
                if value >= min_raw:
                    tx_hash = parsed["transactionHash"].hex()
                    if not tx_hash.startswith("0x"):
                        tx_hash = "0x" + tx_hash
                    logger.info(
                        f"x402 payment verified: {from_address[:10]}... "
                        f"→ {tx_hash[:16]}... (block {parsed['blockNumber']})"
                    )
                    return tx_hash

            end = start - 1

        logger.info(
            f"x402: no qualifying payment from {from_address[:10]}... "
            f"in last {total_lookback} blocks"
        )
        return None

    except Exception as e:
        logger.warning(f"x402 verification error: {e}")
        return None
    finally:
        try:
            await w3.provider.disconnect()
        except Exception:
            pass


async def verify_payment_by_hash(tx_hash: str) -> bool:
    """
    Alternative: verify by tx hash directly.
    Checks that the given tx sent >= x402_price_usdc USDC to agent_address.
    """
    settings = get_settings()
    agent_address = settings.agent_address
    if not agent_address or not settings.arc_enabled:
        return False

    # Normalise tx hash format
    th = tx_hash.strip()
    if not th.startswith("0x"):
        th = "0x" + th

    w3 = AsyncWeb3(AsyncHTTPProvider(settings.arc_rpc))
    w3.middleware_onion.inject(ExtraDataToPOAMiddleware, layer=0)
    try:
        receipt = await w3.eth.get_transaction_receipt(th)
        if not receipt or receipt["status"] != 1:
            return False

        usdc = w3.eth.contract(
            address=AsyncWeb3.to_checksum_address(settings.arc_usdc_contract),
            abi=ERC20_TRANSFER_ABI,
        )
        decimals_raw = await usdc.functions.decimals().call()
        min_raw = int(settings.x402_price_usdc * (10 ** decimals_raw))
        checksum_to = AsyncWeb3.to_checksum_address(agent_address)

        transfer_logs = usdc.events.Transfer().process_receipt(receipt)
        for log in transfer_logs:
            if (
                log["args"]["to"].lower() == checksum_to.lower()
                and log["args"]["value"] >= min_raw
            ):
                return True

        return False

    except Exception as e:
        logger.warning(f"x402 tx hash verification error: {e}")
        return False
    finally:
        try:
            await w3.provider.disconnect()
        except Exception:
            pass
