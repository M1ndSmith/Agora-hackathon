"""
Bayesian prior injection for re-scanned markets.

Loads the most recent unresolved pick for a market_id from SQLite.
"""
import logging
from typing import Optional

from db import store

logger = logging.getLogger(__name__)


async def build_prior(market_id: str) -> Optional[dict]:
    """
    Return prior estimate dict for re-scan, or None if no prior exists.

    Keys: ai_prob (float), summary (str truncated reasoning)
    """
    if not market_id:
        return None

    try:
        prior_row = await store.get_latest_unresolved_pick_for_market(market_id)
    except Exception as e:
        logger.debug(f"Prior lookup failed for {market_id}: {e}")
        return None

    if not prior_row:
        return None

    try:
        ai_prob = float(prior_row.get("ai_prob", 0))
    except (TypeError, ValueError):
        return None

    trace = (prior_row.get("reasoning_trace") or "").strip()
    summary = trace[:300] + ("..." if len(trace) > 300 else "")

    return {
        "ai_prob": ai_prob,
        "summary": summary or "(no prior reasoning)",
        "pick_id": prior_row.get("id"),
    }
