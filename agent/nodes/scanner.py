"""
Scanner node — Stage 1 of the Agora LangGraph.

Fetches active Polymarket markets, runs 2 broad web searches concurrently,
then makes ONE structured LLM call to shortlist mispriced markets.

Outputs: state["candidates"] — list of dicts ready for the researcher.
"""
import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage, SystemMessage

from agent.tools.ev import calculate_ev, confidence_level
from agent.tools.polymarket import fetch_markets
from agent.tools.search import search_compact
from config import get_llm, get_settings

logger = logging.getLogger(__name__)

# Per-search token budget (chars). 2 searches × 600 chars = ~300 tokens context.
_SEARCH_MAX_CHARS = 600

SCANNER_SYSTEM = """You are Agora's market scanner — a fast analytical AI that identifies mispriced prediction markets on Polymarket.

You receive a list of binary YES/NO markets with current prices and recent web context.

For each market, estimate the TRUE probability of YES using your knowledge and the context.
Flag markets where |EV| = |(your_prob - market_prob) / market_prob| > {min_ev}.

OUTPUT FORMAT — strictly a JSON array, nothing else. No prose, no markdown fences.
[
  {{"market_id": "...", "question": "...", "market_prob": 0.XX, "ai_prob": 0.XX}}
]

If nothing qualifies, output: []
"""


def _parse_candidates(content: str) -> List[dict]:
    """Extract a JSON array from the LLM response, tolerating wrappers."""
    content = (content or "").strip()
    try:
        result = json.loads(content)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass

    fenced = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
    if fenced:
        try:
            return json.loads(fenced.group(1))
        except json.JSONDecodeError:
            pass

    bracket = re.search(r"(\[.*\])", content, re.DOTALL)
    if bracket:
        try:
            return json.loads(bracket.group(1))
        except json.JSONDecodeError:
            pass

    return []


async def scanner_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: scan Polymarket markets and shortlist EV candidates.

    1. Fetch markets directly from Polymarket
    2. Run 2 broad context searches concurrently (compact Tavily helper)
    3. ONE structured LLM call over the compact market list
    4. Validate + filter by EV threshold
    """
    settings = get_settings()
    scan_config = state.get("scan_config", {})
    min_ev = scan_config.get("min_ev", settings.min_ev_threshold)
    min_volume = scan_config.get("min_volume", settings.min_volume)

    # ── Step 1: Fetch markets ────────────────────────────────────────────────
    try:
        markets = await fetch_markets(min_volume=min_volume, limit=50)
        logger.info(f"Fetched {len(markets)} markets from Polymarket")
    except Exception as e:
        logger.error(f"Failed to fetch markets: {e}")
        return {**state, "candidates": [], "error": str(e)}

    if not markets:
        logger.warning("No markets returned from Polymarket API")
        return {**state, "candidates": [], "error": "No markets available"}

    # ── Gate 1: drop extreme-price markets (crowd has resolved these) ───────
    pre_filter_count = len(markets)
    markets = [
        m for m in markets
        if settings.extreme_low <= m.market_prob <= settings.extreme_high
    ]
    dropped = pre_filter_count - len(markets)
    if dropped:
        logger.info(
            f"Pre-filtered {dropped} extreme markets "
            f"(p < {settings.extreme_low} or p > {settings.extreme_high}); "
            f"{len(markets)} remain"
        )

    if not markets:
        logger.warning("All markets filtered out by extreme-price gate")
        return {**state, "candidates": [], "error": "All markets at extreme prices"}

    # ── Step 2: Parallel context searches ────────────────────────────────────
    queries = [
        "biggest political and economic events this week 2026",
        "major sports geopolitics breaking news 2026",
    ]
    try:
        search_results = await asyncio.gather(
            *[
                search_compact(q, max_results=2, max_chars=_SEARCH_MAX_CHARS)
                for q in queries
            ],
            return_exceptions=True,
        )
        search_blocks = [s for s in search_results if isinstance(s, str) and s]
        search_context = "\n\n".join(search_blocks)
    except Exception as e:
        logger.warning(f"Web search failed (continuing): {e}")
        search_context = ""

    # ── Step 3: Build compact market list ────────────────────────────────────
    now = datetime.now(timezone.utc)
    market_list = []
    for m in markets:
        days_to_end = (
            int((m.end_date - now).total_seconds() // 86400) if m.end_date else -1
        )
        market_list.append({
            "id": m.id,
            "q": m.question[:200],  # truncate very long questions
            "p": round(m.market_prob, 4),
            "v": int(m.volume),
            "d": days_to_end,
        })

    # Compact JSON, no indent — saves ~30% chars
    market_json = json.dumps(market_list, separators=(",", ":"))

    prompt = f"""MARKETS (compact: id,q=question,p=market_prob_YES,v=volume_usd,d=days_to_end):
{market_json}

CONTEXT (recent news, may help calibrate):
{search_context[:1500] if search_context else "(none)"}

Flag markets where |EV| > {min_ev}. Output JSON array only."""

    # ── Step 4: Single structured LLM call ───────────────────────────────────
    try:
        llm = get_llm(streaming=False)
        system_prompt = SCANNER_SYSTEM.format(min_ev=min_ev)

        response = await llm.ainvoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt),
        ])

        content = response.content if hasattr(response, "content") else str(response)
        raw_candidates = _parse_candidates(content)

    except Exception as e:
        logger.error(f"Scanner LLM call failed: {e}")
        return {**state, "candidates": [], "error": str(e)}

    # ── Step 5: Validate + enrich + attach Polymarket URLs ───────────────────
    market_by_id = {m.id: m for m in markets}

    enriched = []
    for c in raw_candidates:
        try:
            mid = str(c.get("market_id") or c.get("id") or "")
            if not mid or mid not in market_by_id:
                continue
            mp = float(c.get("market_prob") if "market_prob" in c else c.get("p"))
            ap = float(c["ai_prob"])
            if not (0 < mp < 1 and 0 < ap < 1):
                continue
            ev = calculate_ev(mp, ap)
            if abs(ev) < min_ev:
                continue

            source_market = market_by_id[mid]
            enriched.append({
                "market_id": mid,
                "question": source_market.question,
                "market_prob": mp,
                "ai_prob": ap,
                "ev": round(ev, 4),
                "confidence": confidence_level(ev, ap, mp),
                "url": source_market.url,
            })
        except Exception:
            continue

    logger.info(
        f"Scanner: {len(markets)} markets → {len(enriched)} candidates (min_ev={min_ev})"
    )

    return {
        **state,
        "candidates": enriched,
        "markets": [m.id for m in markets],
        "error": None,
    }
