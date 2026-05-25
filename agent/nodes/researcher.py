"""
Researcher node — Stage 2 of the Agora LangGraph.

Deterministic, token-efficient pipeline per candidate:
  1. Fetch market history (one HTTP call, no LLM)
  2. Run 2 compact Tavily searches in parallel (no LLM)
  3. Pack everything into one small context block (capped ~2k chars)
  4. ONE structured LLM call with .with_structured_output(ResearchEstimate)

All candidates are researched concurrently with asyncio.gather (bounded by
an asyncio.Semaphore to respect provider rate limits).

Outputs: state["picks"] — list of Pick objects with structured traces.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agent.tools.ev import (
    calculate_ev,
    confidence_level,
    kelly_fraction,
    logit_distance,
)
from agent.tools.polymarket import get_market_history
from agent.tools.search import search_compact
from config import get_llm, get_settings
from models import Pick, ResearchEstimate

logger = logging.getLogger(__name__)

# Parallelism cap — keeps within Groq's free-tier RPM/TPM ceilings.
_MAX_CONCURRENT_CANDIDATES = 4

# Per-search context truncation. 2 searches × 700 chars = ~1.4k chars context.
_SEARCH_MAX_CHARS = 700

# Maximum total context characters sent to the estimator LLM.
_CONTEXT_BUDGET_CHARS = 2_500

ESTIMATION_PROMPT = """You are a calibrated probability forecaster analyzing a Polymarket prediction market.

You will receive:
- The market question
- Current market probability (what bettors collectively think)
- Compact market metadata (volume, liquidity, spread, competitiveness)
- 2 short web research snippets

Produce a calibrated YES probability estimate. Be honest — not overconfident.

Consider:
- Base rates for similar events
- Recent developments visible in the research
- Market efficiency (the market price already embeds most public information)
- Quality and recency of the evidence

Output the structured ResearchEstimate. Reasoning should be 3-5 sentences citing
specific evidence. Key evidence: 3-5 bullets. Bull/bear: strongest single argument each.
"""


async def _fetch_history_compact(market_id: str) -> str:
    """
    Fetch market metadata directly (no LLM tool). Returns a small string
    with only history-relevant fields: volume, liquidity, spread, competitive.
    """
    try:
        data = await get_market_history(market_id)
    except Exception as e:
        return f"(market history unavailable: {e})"

    if not isinstance(data, dict) or "error" in data:
        return "(market history unavailable)"

    parts = []
    if (v := data.get("volumeNum")) is not None:
        parts.append(f"volume_usd={float(v):.0f}")
    if (liq := data.get("liquidityNum")) is not None:
        parts.append(f"liquidity_usd={float(liq):.0f}")
    if (sp := data.get("spread")) is not None:
        parts.append(f"spread={sp}")
    if (comp := data.get("competitive")) is not None:
        parts.append(f"competitive={comp}")
    if (end := data.get("endDate")):
        parts.append(f"ends={end}")

    return " | ".join(parts) if parts else "(no metadata)"


def _build_search_queries(question: str) -> List[str]:
    """
    Build two complementary search queries for a market question.
    Kept simple and deterministic so we don't burn an LLM call to write them.
    """
    q = (question or "").strip()
    return [
        f"{q} latest news 2026",
        f"{q} expert analysis odds probability",
    ]


def _pack_context(
    candidate: dict,
    history: str,
    searches: List[str],
) -> str:
    """Assemble a compact context block, respecting the global char budget."""
    header = (
        f"QUESTION: {candidate.get('question', '')[:250]}\n"
        f"CURRENT MARKET PROBABILITY (YES): {candidate.get('market_prob')}\n"
        f"SCANNER ESTIMATE (YES): {candidate.get('ai_prob')}\n"
        f"SCANNER EV: {candidate.get('ev')}\n"
        f"MARKET METADATA: {history}\n"
        f"\nWEB RESEARCH:\n"
    )

    body_parts = []
    for i, s in enumerate(searches, 1):
        snippet = (s or "").strip()
        if snippet:
            body_parts.append(f"[Search {i}]\n{snippet[:_SEARCH_MAX_CHARS]}")
    body = "\n\n".join(body_parts) if body_parts else "(no web results)"

    full = header + body
    return full[:_CONTEXT_BUDGET_CHARS]


async def _invoke_with_retry(runnable, payload, max_retries: int = 3):
    """Exponential backoff on rate-limit / oversized-payload errors."""
    for attempt in range(max_retries):
        try:
            return await runnable.ainvoke(payload)
        except Exception as e:
            err = str(e)
            is_rate_limit = (
                "rate_limit" in err.lower()
                or "429" in err
                or "413" in err
                or "too large" in err.lower()
            )
            if is_rate_limit and attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                logger.warning(
                    f"Rate/size limit hit, retrying in {wait}s (attempt {attempt+1})"
                )
                await asyncio.sleep(wait)
            else:
                raise


async def _research_single_candidate(
    candidate: dict,
    settings,
    semaphore: asyncio.Semaphore,
) -> Optional[Pick]:
    """
    Deterministic per-candidate research pipeline (no ReAct loop).

    Steps:
    1. fetch_history_compact (parallel with searches)
    2. search_compact × 2 (parallel)
    3. pack into one compact context block
    4. single structured LLM call
    """
    async with semaphore:
        market_id = candidate.get("market_id", "")
        question = candidate.get("question", "")
        queries = _build_search_queries(question)

        # ── Parallel I/O: history + both searches at once ────────────────────
        try:
            history, search1, search2 = await asyncio.gather(
                _fetch_history_compact(market_id),
                search_compact(queries[0], max_results=2, max_chars=_SEARCH_MAX_CHARS),
                search_compact(queries[1], max_results=2, max_chars=_SEARCH_MAX_CHARS),
                return_exceptions=False,
            )
        except Exception as e:
            logger.error(f"Evidence-gather error for {market_id}: {e}")
            return None

        context = _pack_context(candidate, history, [search1, search2])

        # ── Single structured LLM call ───────────────────────────────────────
        try:
            estimator = get_llm(streaming=False).with_structured_output(ResearchEstimate)
            estimate: ResearchEstimate = await _invoke_with_retry(
                estimator,
                [
                    SystemMessage(content=ESTIMATION_PROMPT),
                    HumanMessage(content=context),
                ],
            )
        except Exception as e:
            logger.error(f"Estimator error for {market_id}: {e}")
            return None

        # ── Build Pick ───────────────────────────────────────────────────────
        try:
            ai_prob = max(0.01, min(0.99, float(estimate.ai_prob)))
        except (TypeError, ValueError):
            return None
        market_prob = float(candidate.get("market_prob", 0.5))
        ev = calculate_ev(market_prob, ai_prob)
        conf = estimate.confidence or confidence_level(ev, ai_prob, market_prob)

        # ── Sanity gates: reject statistically implausible picks ────────────
        abs_edge = abs(ai_prob - market_prob)
        logit_dist = logit_distance(ai_prob, market_prob)

        if abs(ev) < settings.min_ev_threshold:
            logger.info(
                f"Market {market_id} killed: EV {ev:+.3f} < threshold "
                f"{settings.min_ev_threshold}"
            )
            return None

        if abs_edge < settings.min_abs_edge:
            logger.info(
                f"Market {market_id} killed: abs_edge {abs_edge:.3f} < "
                f"{settings.min_abs_edge} (tiny denominator inflating EV)"
            )
            return None

        if logit_dist > settings.max_logit_distance:
            logger.info(
                f"Market {market_id} killed: logit_distance {logit_dist:.2f} > "
                f"{settings.max_logit_distance} (AI vs market disagree too strongly)"
            )
            return None

        k_frac = kelly_fraction(market_prob, ai_prob, bankroll=100.0)

        return Pick(
            market_id=str(market_id),
            question=question,
            market_prob=market_prob,
            ai_prob=ai_prob,
            ev=round(ev, 4),
            kelly_fraction=round(k_frac, 4),
            confidence=conf,
            reasoning_trace=estimate.reasoning or "",
            key_evidence=estimate.key_evidence or [],
            bull_case=estimate.bull_case or "",
            bear_case=estimate.bear_case or "",
            builder_url=candidate.get("url", ""),
        )


async def researcher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: research all scanner candidates concurrently.
    Updates state["picks"] with confirmed high-EV picks.
    """
    settings = get_settings()
    candidates = state.get("candidates", [])

    if not candidates:
        return {**state, "picks": []}

    top_candidates = candidates[: settings.top_n_picks]
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_CANDIDATES)

    logger.info(
        f"Researching {len(top_candidates)} candidates "
        f"(concurrency={_MAX_CONCURRENT_CANDIDATES})"
    )

    results = await asyncio.gather(
        *[
            _research_single_candidate(c, settings, semaphore)
            for c in top_candidates
        ],
        return_exceptions=True,
    )

    picks: List[Pick] = []
    for r in results:
        if isinstance(r, Exception):
            logger.error(f"Candidate research raised: {r}")
            continue
        if r is not None:
            picks.append(r)

    logger.info(
        f"Researcher confirmed {len(picks)} picks from {len(top_candidates)} candidates"
    )
    return {**state, "picks": picks}
