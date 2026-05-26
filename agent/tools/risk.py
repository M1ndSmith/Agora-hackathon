"""
Portfolio risk management — position caps, theme exposure, drawdown pause.

Pure functions; advisory hedge and early-close recommendations.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from agent.tools.metrics import cumulative_pnl
from config import Settings, get_settings
from models import (
    EarlyCloseRecommendation,
    HedgeRecommendation,
    Pick,
    PortfolioRecommendation,
    RiskAssessment,
)


def theme_key(question: str, domain: str = "") -> str:
    """
    Lightweight correlation group key from domain + top keywords.
    """
    q = (question or "").lower()
    tokens = re.findall(r"[a-z0-9]{4,}", q)
    stop = {
        "will", "what", "when", "that", "this", "with", "from", "have",
        "been", "before", "after", "than", "into", "over", "under",
    }
    keywords = sorted({t for t in tokens if t not in stop})[:3]
    dom = (domain or "general").lower()
    if keywords:
        return f"{dom}:{','.join(keywords)}"
    return dom


def cap_position_size(
    size_usdc: float,
    bankroll: float,
    max_fraction: float,
) -> Tuple[float, bool]:
    cap = bankroll * max_fraction
    if size_usdc <= cap:
        return size_usdc, False
    return round(cap, 4), True


def _rescale_group(
    items: List[Dict[str, Any]],
    max_total: float,
    size_key: str = "size_usdc",
) -> List[Dict[str, Any]]:
    total = sum(i[size_key] for i in items)
    if total <= max_total or total <= 0:
        return items
    scale = max_total / total
    out = []
    for i in items:
        copy = dict(i)
        copy[size_key] = round(copy[size_key] * scale, 4)
        copy["capped"] = True
        out.append(copy)
    return out


def cap_total_exposure(
    recommendations: List[Dict[str, Any]],
    bankroll: float,
    max_fraction: float,
) -> List[Dict[str, Any]]:
    max_total = bankroll * max_fraction
    return _rescale_group(recommendations, max_total)


def cap_theme_exposure(
    recommendations: List[Dict[str, Any]],
    bankroll: float,
    max_fraction: float,
) -> List[Dict[str, Any]]:
    max_theme = bankroll * max_fraction
    by_theme: Dict[str, List[Dict[str, Any]]] = {}
    for r in recommendations:
        tk = r.get("theme_key", "general")
        by_theme.setdefault(tk, []).append(r)

    result: List[Dict[str, Any]] = []
    for group in by_theme.values():
        result.extend(_rescale_group(group, max_theme))
    return result


def drawdown_paused(
    history: List[dict],
    bankroll: float,
    threshold: float,
) -> bool:
    """
    Pause new sizing if cumulative realized P&L drawdown exceeds threshold
    (threshold is negative, e.g. -0.10 for -10% of bankroll).
    """
    if bankroll <= 0:
        return False
    series = cumulative_pnl(history)
    if not series:
        return False
    total_pnl = series[-1][1]
    return total_pnl <= (bankroll * threshold)


def recommend_hedge(
    pick: dict,
    current_market_prob: float,
    threshold: float,
) -> HedgeRecommendation:
    """
    Suggest hedge when market price moved against the agent's side.
    """
    mid = str(pick.get("market_id", ""))
    entry_prob = float(pick.get("market_prob") or 0.5)
    ai_prob = float(pick.get("ai_prob") or 0.5)
    bet_yes = ai_prob >= 0.5
    move = current_market_prob - entry_prob
    move_pp = move * 100

    if bet_yes and move <= -threshold:
        return HedgeRecommendation(
            market_id=mid,
            suggested=True,
            reason=f"YES pick: market fell {abs(move_pp):.1f}pp against position",
            hedge_side="BUY_NO",
            move_pp=round(move_pp, 2),
        )
    if not bet_yes and move >= threshold:
        return HedgeRecommendation(
            market_id=mid,
            suggested=True,
            reason=f"NO pick: market rose {abs(move_pp):.1f}pp against position",
            hedge_side="BUY_YES",
            move_pp=round(move_pp, 2),
        )
    return HedgeRecommendation(market_id=mid, suggested=False, move_pp=round(move_pp, 2))


def recommend_early_close(
    pick: dict,
    current_market_prob: float,
    profit_threshold: float,
    loss_threshold: float,
) -> EarlyCloseRecommendation:
    """
    Mark-to-market using price move in the agent's direction.

    YES pick profits when market_prob rises toward ai_prob; take profit when
    move is large enough. Cut loss when price moves against the position.
    """
    from agent.tools.ev import calculate_ev

    mid = str(pick.get("market_id", ""))
    ai_prob = float(pick.get("ai_prob") or 0.5)
    entry_prob = float(pick.get("market_prob") or 0.5)
    bet_yes = ai_prob >= entry_prob
    move = current_market_prob - entry_prob
    favorable = move if bet_yes else -move
    unrealized = calculate_ev(current_market_prob, ai_prob) - calculate_ev(entry_prob, ai_prob)

    if favorable >= profit_threshold:
        return EarlyCloseRecommendation(
            market_id=mid,
            action="take_profit",
            reason=f"Price moved {favorable:+.1%} in favor of position",
            unrealized_ev=round(unrealized, 4),
        )
    if favorable <= loss_threshold:
        return EarlyCloseRecommendation(
            market_id=mid,
            action="cut_loss",
            reason=f"Price moved {favorable:+.1%} against position",
            unrealized_ev=round(unrealized, 4),
        )
    return EarlyCloseRecommendation(
        market_id=mid,
        action="hold",
        reason="Within normal mark-to-market range",
        unrealized_ev=round(unrealized, 4),
    )


def assess_risk(
    picks: List[Pick],
    history: List[dict],
    bankroll: float,
    settings: Optional[Settings] = None,
) -> PortfolioRecommendation:
    """
    Apply per-market, total, and theme exposure caps to raw Kelly sizes.
    """
    settings = settings or get_settings()
    paused = drawdown_paused(
        history, bankroll, settings.drawdown_pause_threshold
    )

    recs: List[Dict[str, Any]] = []
    theme_groups: Dict[str, List[str]] = {}

    for pick in picks:
        tk = theme_key(pick.question, pick.domain)
        theme_groups.setdefault(tk, []).append(pick.market_id)
        raw = float(pick.kelly_fraction or 0.0)
        if paused:
            adj = 0.0
            warnings = ["drawdown_pause: new sizing halted"]
            capped = True
        else:
            adj, capped = cap_position_size(
                raw, bankroll, settings.max_position_fraction
            )
            warnings = []
            if capped:
                warnings.append("per_market_cap")

        recs.append({
            "market_id": pick.market_id,
            "theme_key": tk,
            "size_usdc": adj,
            "raw_size_usdc": raw,
            "warnings": warnings,
            "capped": capped,
        })

    if not paused and recs:
        recs = cap_theme_exposure(
            recs, bankroll, settings.max_theme_exposure_fraction
        )
        recs = cap_total_exposure(
            recs, bankroll, settings.max_total_exposure_fraction
        )

    assessments = [
        RiskAssessment(
            market_id=r["market_id"],
            theme_key=r["theme_key"],
            raw_size_usdc=r["raw_size_usdc"],
            adjusted_size_usdc=r["size_usdc"],
            warnings=r.get("warnings", []),
            capped=r.get("capped", False),
        )
        for r in recs
    ]

    return PortfolioRecommendation(
        bankroll=bankroll,
        total_exposure_usdc=round(sum(a.adjusted_size_usdc for a in assessments), 4),
        pick_count=len(picks),
        drawdown_paused=paused,
        theme_groups=theme_groups,
        assessments=assessments,
    )
