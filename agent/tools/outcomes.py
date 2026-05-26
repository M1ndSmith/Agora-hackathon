"""
Resolve open picks against Polymarket Gamma API outcomes.
"""
import asyncio
import logging
from typing import List, Optional, Tuple

from agent.tools.polymarket import fetch_resolved_market
from db import store

logger = logging.getLogger(__name__)

_MAX_CONCURRENT = 5


async def _resolve_one(pick: dict, sem: asyncio.Semaphore) -> Tuple[int, bool, Optional[str]]:
    pick_id = int(pick["id"])
    market_id = str(pick.get("market_id") or "")
    async with sem:
        resolved, outcome = await fetch_resolved_market(market_id)
    if resolved and outcome:
        await store.update_pick_outcome(pick_id, outcome)
        return pick_id, True, outcome
    return pick_id, False, None


async def resolve_open_picks() -> dict:
    """
    Poll Polymarket for all unresolved picks and update SQLite outcomes.

    Returns summary dict with keys: checked, newly_resolved, still_open.
    """
    await store.init_db()
    open_picks = await store.get_unresolved_picks()
    if not open_picks:
        return {"checked": 0, "newly_resolved": 0, "still_open": 0}

    sem = asyncio.Semaphore(_MAX_CONCURRENT)
    results = await asyncio.gather(
        *[_resolve_one(p, sem) for p in open_picks],
        return_exceptions=True,
    )

    newly = 0
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"resolve pick error: {r}")
            continue
        _, ok, _ = r
        if ok:
            newly += 1

    still_open = len(open_picks) - newly
    return {
        "checked": len(open_picks),
        "newly_resolved": newly,
        "still_open": still_open,
    }
