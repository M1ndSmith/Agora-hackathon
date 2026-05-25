"""
Polymarket Gamma API client.

Live schema verified from:
  GET https://gamma-api.polymarket.com/markets?active=true&closed=false&limit=2&order=volume&ascending=false

Key fields used:
  id, question, outcomePrices (JSON string), volumeNum (float), endDate, slug

A module-level httpx.AsyncClient is reused across calls so we avoid the
TLS handshake / connection-pool setup on every request.
"""
import asyncio
import json
from datetime import datetime, timezone, timedelta
from typing import List, Optional

import httpx

from config import get_settings
from models import PolymarketMarket

GAMMA_API = "https://gamma-api.polymarket.com/markets"
POLYMARKET_BASE = "https://polymarket.com/market"
BUILDER_REF = "https://polymarket.com/market/{slug}?ref=agora-agent"

# Module-level shared httpx client (created lazily on first use).
# Reusing the same client keeps connections in the pool warm.
_client: Optional[httpx.AsyncClient] = None
_client_lock = asyncio.Lock()


async def _get_client() -> httpx.AsyncClient:
    """Return the shared AsyncClient, creating it on first call."""
    global _client
    if _client is None or _client.is_closed:
        async with _client_lock:
            if _client is None or _client.is_closed:
                _client = httpx.AsyncClient(
                    timeout=httpx.Timeout(30.0, connect=10.0),
                    limits=httpx.Limits(
                        max_connections=20,
                        max_keepalive_connections=10,
                    ),
                )
    return _client


def _parse_binary_prob(outcome_prices: str) -> Optional[float]:
    """Parse outcomePrices JSON string → YES probability. Returns None if not binary."""
    try:
        prices = json.loads(outcome_prices)
        if len(prices) != 2:
            return None
        return float(prices[0])
    except (json.JSONDecodeError, ValueError, IndexError):
        return None


def _is_binary(outcomes: str) -> bool:
    """Check outcomes field is a two-outcome binary market."""
    try:
        parsed = json.loads(outcomes)
        return len(parsed) == 2
    except Exception:
        return False


async def fetch_markets(
    min_volume: Optional[float] = None,
    limit: int = 50,
    max_days_to_end: int = 60,
) -> List[PolymarketMarket]:
    """
    Fetch active binary markets from Polymarket Gamma API, sorted by volume desc.
    Filters by min_volume and end_date within max_days_to_end days.
    """
    settings = get_settings()
    if min_volume is None:
        min_volume = settings.min_volume

    params = {
        "active": "true",
        "closed": "false",
        "limit": limit,
        "order": "volume",
        "ascending": "false",
    }

    client = await _get_client()
    resp = await client.get(GAMMA_API, params=params)
    resp.raise_for_status()
    raw_markets = resp.json()

    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(days=max_days_to_end)
    results: List[PolymarketMarket] = []

    for m in raw_markets:
        # Volume filter — use volumeNum (float field confirmed from live API)
        volume = m.get("volumeNum", 0.0)
        if volume < min_volume:
            continue

        # Binary market check
        outcomes_raw = m.get("outcomes", "[]")
        if not _is_binary(outcomes_raw):
            continue

        # Parse YES probability
        outcome_prices_raw = m.get("outcomePrices", "[]")
        market_prob = _parse_binary_prob(outcome_prices_raw)
        if market_prob is None:
            continue

        # End date filter
        end_date_str = m.get("endDate") or m.get("endDateIso")
        if not end_date_str:
            continue
        try:
            end_date = datetime.fromisoformat(end_date_str.replace("Z", "+00:00"))
        except ValueError:
            continue
        if end_date < now or end_date > cutoff:
            continue

        slug = m.get("slug", m.get("id", ""))
        market = PolymarketMarket(
            id=str(m["id"]),
            question=m["question"],
            market_prob=market_prob,
            volume=volume,
            end_date=end_date,
            url=f"{POLYMARKET_BASE}/{slug}",
            builder_code_url=BUILDER_REF.format(slug=slug),
        )
        results.append(market)

    return results


async def get_market_history(market_id: str) -> dict:
    """Fetch price history for a specific market."""
    url = f"{GAMMA_API}/{market_id}"
    try:
        client = await _get_client()
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        return {"error": str(e), "market_id": market_id}


async def close_client() -> None:
    """Close the shared httpx client. Optional — primarily for clean shutdown."""
    global _client
    if _client is not None and not _client.is_closed:
        await _client.aclose()
        _client = None
