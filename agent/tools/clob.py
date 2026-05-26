"""
Polymarket CLOB order book microstructure signals.

Fetches bid/ask spread and depth; never breaks the pipeline on failure.
"""
import logging
from typing import Any, Dict, List, Optional

import httpx

from config import get_settings
from models import MicrostructureSignal

logger = logging.getLogger(__name__)


def _parse_price_level(level: Any) -> Optional[float]:
    if isinstance(level, dict):
        for key in ("price", "p"):
            if key in level:
                try:
                    return float(level[key])
                except (TypeError, ValueError):
                    return None
        return None
    try:
        return float(level)
    except (TypeError, ValueError):
        return None


def _parse_size(level: Any) -> float:
    if isinstance(level, dict):
        for key in ("size", "s", "amount"):
            if key in level:
                try:
                    return float(level[key])
                except (TypeError, ValueError):
                    return 0.0
    return 0.0


def parse_orderbook(raw: Dict[str, Any]) -> Optional[MicrostructureSignal]:
    """
    Parse CLOB /book response into spread and depth summary.

    Returns None if book is empty or malformed.
    """
    if not raw or not isinstance(raw, dict):
        return None

    bids_raw: List = raw.get("bids") or []
    asks_raw: List = raw.get("asks") or []

    bid_prices = []
    for b in bids_raw:
        p = _parse_price_level(b)
        if p is not None and 0 <= p <= 1:
            bid_prices.append((p, _parse_size(b)))

    ask_prices = []
    for a in asks_raw:
        p = _parse_price_level(a)
        if p is not None and 0 <= p <= 1:
            ask_prices.append((p, _parse_size(a)))

    if not bid_prices and not ask_prices:
        return None

    best_bid = max((p for p, _ in bid_prices), default=0.0)
    best_ask = min((p for p, _ in ask_prices), default=1.0)

    if bid_prices and ask_prices and best_ask < best_bid:
        best_ask = best_bid

    spread = max(0.0, best_ask - best_bid) if (bid_prices and ask_prices) else 0.0

    depth_usd = 0.0
    for p, sz in bid_prices[:5] + ask_prices[:5]:
        depth_usd += p * sz

    return MicrostructureSignal(
        best_bid=round(best_bid, 4),
        best_ask=round(best_ask, 4),
        spread=round(spread, 4),
        depth_usd=round(depth_usd, 2),
    )


async def fetch_microstructure(clob_token_id: str) -> Optional[MicrostructureSignal]:
    """
    Fetch order book for a YES token from Polymarket CLOB API.

    Returns None on any error or when CLOB is disabled.
    """
    settings = get_settings()
    if not settings.clob_enabled or not clob_token_id:
        return None

    url = f"{settings.clob_base_url.rstrip('/')}/book"
    params = {"token_id": clob_token_id}

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
        return parse_orderbook(data if isinstance(data, dict) else {})
    except Exception as e:
        logger.debug(f"CLOB fetch failed for {clob_token_id[:12]}...: {e}")
        return None
