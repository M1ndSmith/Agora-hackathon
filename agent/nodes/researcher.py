"""
Researcher node — Stage 2 of the Agora LangGraph.

Deterministic, token-efficient pipeline per candidate:
  1. Fetch market history + CLOB + social + prior (parallel, no LLM)
  2. Run 2 credibility-weighted Tavily searches in parallel
  3. Pack everything into one small context block (capped ~2.5k chars)
  4. ONE structured LLM call (or ensemble when enabled)

Tier 2: domain routing, Bayesian prior, CLOB microstructure, source credibility.
"""
import asyncio
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agent.tools.clob import fetch_microstructure
from agent.tools.domain import classify_domain, get_estimation_prompt
from agent.tools.ensemble import run_ensemble_estimate
from agent.tools.ev import (
    calculate_ev,
    confidence_level,
    kelly_fraction,
    logit_distance,
)
from agent.tools.polymarket import get_market_history
from agent.tools.prior import build_prior
from agent.tools.search import search_compact, search_weighted
from agent.tools.social import fetch_social_signal
from config import get_llm, get_settings
from models import Pick, ResearchEstimate

logger = logging.getLogger(__name__)

_MAX_CONCURRENT_CANDIDATES = 4
_SEARCH_MAX_CHARS = 700
_CONTEXT_BUDGET_CHARS = 2_500


async def _fetch_history_compact(market_id: str) -> str:
    """Fetch market metadata directly (no LLM tool)."""
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
    q = (question or "").strip()
    return [
        f"{q} latest news 2026",
        f"{q} expert analysis odds probability",
    ]


def _pack_context(
    candidate: dict,
    history: str,
    searches: List[str],
    prior: Optional[dict] = None,
    microstructure: Optional[Any] = None,
    social: Optional[dict] = None,
) -> str:
    """Assemble a compact context block, respecting the global char budget."""
    header = (
        f"QUESTION: {candidate.get('question', '')[:250]}\n"
        f"CURRENT MARKET PROBABILITY (YES): {candidate.get('market_prob')}\n"
        f"SCANNER ESTIMATE (YES): {candidate.get('ai_prob')}\n"
        f"SCANNER EV: {candidate.get('ev')}\n"
        f"MARKET METADATA: {history}\n"
    )

    if prior:
        header += (
            f"PRIOR ESTIMATE (previous scan): YES={prior.get('ai_prob')}\n"
            f"PRIOR SUMMARY: {(prior.get('summary') or '')[:200]}\n"
            "Update this prior with new evidence; do not ignore it.\n"
        )

    if microstructure is not None:
        header += (
            f"CLOB: bid={microstructure.best_bid:.3f} ask={microstructure.best_ask:.3f} "
            f"spread={microstructure.spread:.3f} depth_usd={microstructure.depth_usd:.0f}\n"
        )

    if social:
        header += (
            f"SOCIAL: vol={social.get('mention_volume', 'n/a')} "
            f"sentiment={social.get('sentiment', 'n/a')}\n"
        )

    header += "\nWEB RESEARCH:\n"

    body_parts = []
    for i, s in enumerate(searches, 1):
        snippet = (s or "").strip()
        if snippet:
            body_parts.append(f"[Search {i}]\n{snippet[:_SEARCH_MAX_CHARS]}")
    body = "\n\n".join(body_parts) if body_parts else "(no web results)"

    full = header + body
    return full[:_CONTEXT_BUDGET_CHARS]


async def _invoke_with_retry(runnable, payload, max_retries: int = 3):
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


async def _run_searches(question: str, settings) -> tuple:
    """Return (search_texts, top_source, top_score)."""
    queries = _build_search_queries(question)
    use_weighted = True

    if use_weighted:
        r1, r2 = await asyncio.gather(
            search_weighted(queries[0], max_results=2, max_chars=_SEARCH_MAX_CHARS),
            search_weighted(queries[1], max_results=2, max_chars=_SEARCH_MAX_CHARS),
            return_exceptions=True,
        )
        texts = []
        top_source = ""
        top_score = 0.0
        for r in (r1, r2):
            if isinstance(r, Exception):
                continue
            texts.append(r.get("text", ""))
            if r.get("top_score", 0) > top_score:
                top_score = r.get("top_score", 0)
                top_source = r.get("top_source", "")
        return texts, top_source, top_score

    s1, s2 = await asyncio.gather(
        search_compact(queries[0], max_results=2, max_chars=_SEARCH_MAX_CHARS),
        search_compact(queries[1], max_results=2, max_chars=_SEARCH_MAX_CHARS),
    )
    return [s1, s2], "", 0.0


async def _noop() -> None:
    return None


async def _estimate_probability(
    system_prompt: str,
    context: str,
    settings,
) -> tuple:
    """Return (ResearchEstimate, extra_signals dict)."""
    if settings.ensemble_enabled:
        ens = await run_ensemble_estimate(system_prompt, context)
        extra = {
            "ensemble": True,
            "ensemble_spread": ens.spread,
            "ensemble_providers": ens.providers,
        }
        return (
            ResearchEstimate(
                ai_prob=ens.ai_prob,
                confidence=ens.confidence,
                reasoning=ens.reasoning,
                key_evidence=ens.key_evidence,
                bull_case=ens.bull_case,
                bear_case=ens.bear_case,
            ),
            extra,
        )

    estimator = get_llm(streaming=False).with_structured_output(ResearchEstimate)
    est = await _invoke_with_retry(
        estimator,
        [
            SystemMessage(content=system_prompt),
            HumanMessage(content=context),
        ],
    )
    return est, {}


async def _research_single_candidate(
    candidate: dict,
    settings,
    semaphore: asyncio.Semaphore,
) -> Optional[Pick]:
    async with semaphore:
        market_id = candidate.get("market_id", "")
        question = candidate.get("question", "")
        clob_token_id = candidate.get("clob_token_id")

        domain = (
            classify_domain(question)
            if settings.domain_routing_enabled
            else "general"
        )
        system_prompt = get_estimation_prompt(domain)

        # ── Parallel I/O ─────────────────────────────────────────────────────
        try:
            history_coro = _fetch_history_compact(market_id)
            prior_coro = build_prior(market_id)
            clob_coro = (
                fetch_microstructure(clob_token_id)
                if clob_token_id
                else _noop()
            )
            social_coro = (
                fetch_social_signal(question)
                if settings.social_enabled
                else _noop()
            )

            history, prior, micro, social = await asyncio.gather(
                history_coro,
                prior_coro,
                clob_coro,
                social_coro,
                return_exceptions=True,
            )
            if isinstance(history, Exception):
                history = "(market history unavailable)"
            if isinstance(prior, Exception):
                prior = None
            if isinstance(micro, Exception):
                micro = None
            if isinstance(social, Exception):
                social = None

            search_texts, top_source, top_score = await _run_searches(question, settings)

        except Exception as e:
            logger.error(f"Evidence-gather error for {market_id}: {e}")
            return None

        context = _pack_context(
            candidate, history, search_texts, prior=prior,
            microstructure=micro, social=social,
        )

        # ── Structured LLM (single or ensemble) ────────────────────────────
        try:
            estimate, extra_signals = await _estimate_probability(
                system_prompt, context, settings
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

        abs_edge = abs(ai_prob - market_prob)
        logit_dist = logit_distance(ai_prob, market_prob)

        if abs(ev) < settings.min_ev_threshold:
            logger.info(f"Market {market_id} killed: EV {ev:+.3f} < threshold")
            return None
        if abs_edge < settings.min_abs_edge:
            logger.info(f"Market {market_id} killed: abs_edge {abs_edge:.3f}")
            return None
        if logit_dist > settings.max_logit_distance:
            logger.info(f"Market {market_id} killed: logit_distance {logit_dist:.2f}")
            return None

        from agent.tools.ev import kelly_fraction_two_sided
        k_frac = kelly_fraction_two_sided(market_prob, ai_prob, bankroll=100.0)

        signals: Dict[str, Any] = {
            "top_source": top_source,
            "top_score": top_score,
        }
        if micro is not None:
            signals["clob_spread"] = micro.spread
            signals["clob_depth_usd"] = micro.depth_usd
            signals["clob_bid"] = micro.best_bid
            signals["clob_ask"] = micro.best_ask
        if prior:
            signals["prior_ai_prob"] = prior.get("ai_prob")
            signals["prior_updated"] = True
        signals.update(extra_signals)

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
            domain=domain,
            signals=signals,
        )


async def researcher_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node: research all scanner candidates concurrently."""
    settings = get_settings()
    candidates = state.get("candidates", [])

    if not candidates:
        return {**state, "picks": []}

    top_candidates = candidates[: settings.top_n_picks]
    semaphore = asyncio.Semaphore(_MAX_CONCURRENT_CANDIDATES)

    logger.info(
        f"Researching {len(top_candidates)} candidates "
        f"(concurrency={_MAX_CONCURRENT_CANDIDATES}, "
        f"ensemble={settings.ensemble_enabled})"
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
