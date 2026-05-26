"""
Portfolio node — Stage 2.5 of the Agora LangGraph.

Applies risk caps, slippage-aware sizing, dry-run CLOB order tickets,
and advisory hedge / early-close recommendations.
"""
import logging
from typing import Any, Dict, List, Optional

from agent.tools.ev import slippage_aware_kelly
from agent.tools.orders import build_order_ticket, ticket_to_dict
from agent.tools.risk import (
    assess_risk,
    recommend_early_close,
    recommend_hedge,
)
from agent.tools.polymarket import get_market_history
from config import get_settings
from models import MicrostructureSignal, Pick

logger = logging.getLogger(__name__)


async def _current_market_prob(market_id: str, fallback: float) -> float:
    try:
        data = await get_market_history(market_id)
        if not isinstance(data, dict) or data.get("error"):
            return fallback
        raw = data.get("outcomePrices", "[]")
        if isinstance(raw, str):
            import json
            prices = json.loads(raw)
        elif isinstance(raw, list):
            prices = raw
        else:
            return fallback
        if len(prices) >= 1:
            return float(prices[0])
    except Exception as e:
        logger.debug(f"Live prob fetch failed for {market_id}: {e}")
    return fallback


def _micro_from_signals(signals: dict) -> Optional[MicrostructureSignal]:
    if not signals.get("clob_bid") and not signals.get("clob_ask"):
        return None
    try:
        return MicrostructureSignal(
            best_bid=float(signals.get("clob_bid", 0)),
            best_ask=float(signals.get("clob_ask", 1)),
            spread=float(signals.get("clob_spread", 0)),
            depth_usd=float(signals.get("clob_depth_usd", 0)),
        )
    except (TypeError, ValueError):
        return None


async def portfolio_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: portfolio-aware sizing and dry-run execution tickets.
    """
    settings = get_settings()
    picks: List[Pick] = state.get("picks", [])
    if not picks:
        return {
            **state,
            "portfolio": None,
            "risk_summary": {"total_exposure": 0.0, "paused": False},
            "arbitrage_signals": state.get("arbitrage_signals", []),
        }

    bankroll = float(state.get("wallet_balance") or 0.0)
    if bankroll <= 0:
        try:
            from onchain.wallet import get_balance
            bankroll = await get_balance()
        except Exception as e:
            logger.debug(f"Wallet balance fetch failed: {e}")
    if bankroll <= 0:
        bankroll = settings.fallback_bankroll

    history: List[dict] = []
    try:
        from db import store
        await store.init_db()
        history = await store.get_pick_history()
    except Exception as e:
        logger.debug(f"History load for drawdown check: {e}")

    portfolio = assess_risk(picks, history, bankroll, settings)
    assessment_by_id = {a.market_id: a for a in portfolio.assessments}

    updated: List[Pick] = []
    for pick in picks:
        a = assessment_by_id.get(pick.market_id)
        size = a.adjusted_size_usdc if a else float(pick.kelly_fraction or 0)

        signals = dict(pick.signals or {})
        micro = _micro_from_signals(signals)
        depth = micro.depth_usd if micro else 0.0

        if not portfolio.drawdown_paused:
            # Start from the risk-capped size from assess_risk, then only
            # cap further by usable CLOB depth (if known).
            if depth > 0:
                depth_cap = depth * settings.slippage_depth_fraction
                size = round(min(size, depth_cap), 4)

        current_prob = await _current_market_prob(pick.market_id, pick.market_prob)
        hedge = recommend_hedge(pick.model_dump(), current_prob, settings.hedge_move_threshold)
        early = recommend_early_close(
            pick.model_dump(),
            current_prob,
            settings.early_close_profit_threshold,
            settings.early_close_loss_threshold,
        )

        ticket = build_order_ticket(pick, size, microstructure=micro, dry_run=settings.execution_dry_run)

        signals.update({
            "portfolio_size_usdc": size,
            "theme_key": a.theme_key if a else "",
            "risk_warnings": a.warnings if a else [],
            "risk_capped": a.capped if a else False,
            "order_ticket": ticket_to_dict(ticket),
            "dry_run": settings.execution_dry_run,
            "hedge": hedge.model_dump(),
            "early_close": early.model_dump(),
            "current_market_prob": current_prob,
        })

        updated.append(
            pick.model_copy(
                update={
                    "kelly_fraction": round(size, 4),
                    "signals": signals,
                }
            )
        )

    risk_summary = {
        "bankroll": bankroll,
        "total_exposure": portfolio.total_exposure_usdc,
        "pick_count": portfolio.pick_count,
        "paused": portfolio.drawdown_paused,
        "theme_groups": portfolio.theme_groups,
    }

    logger.info(
        f"Portfolio: {len(updated)} picks, exposure=${portfolio.total_exposure_usdc:.2f}, "
        f"paused={portfolio.drawdown_paused}"
    )

    return {
        **state,
        "picks": updated,
        "wallet_balance": bankroll,
        "portfolio": portfolio.model_dump(),
        "risk_summary": risk_summary,
    }
