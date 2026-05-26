"""
Pick performance metrics — P&L, Brier score, calibration, hit rate.

Pure functions only (no I/O, no LLM). Used by CLI and Streamlit dashboards.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple


def _is_resolved(pick: dict) -> bool:
    return bool(pick.get("resolved")) and pick.get("outcome") in ("yes", "no")


def _outcome_binary(pick: dict) -> Optional[float]:
    if not _is_resolved(pick):
        return None
    return 1.0 if pick.get("outcome") == "yes" else 0.0


def _bet_yes(pick: dict) -> bool:
    """Agent side: YES if ai_prob >= 0.5 else NO."""
    return float(pick.get("ai_prob") or 0.5) >= 0.5


def _is_hit(pick: dict) -> bool:
    if not _is_resolved(pick):
        return False
    outcome = pick.get("outcome")
    ai_prob = float(pick.get("ai_prob") or 0.5)
    if ai_prob > 0.5:
        return outcome == "yes"
    if ai_prob < 0.5:
        return outcome == "no"
    return outcome == "yes"


def _parse_created_at(pick: dict) -> datetime:
    raw = pick.get("created_at")
    if isinstance(raw, datetime):
        return raw
    if not raw:
        return datetime.min
    try:
        return datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except ValueError:
        return datetime.min


def realized_pnl(pick: dict, bankroll: float = 100.0) -> Optional[float]:
    """
    Realized P&L for a resolved pick using stored kelly_fraction as stake (USDC).

    YES bet: win profit = stake * (1/market_prob - 1), lose = -stake
    NO bet: win profit = stake * (1/(1-market_prob) - 1), lose = -stake
    """
    if not _is_resolved(pick):
        return None

    stake = float(pick.get("kelly_fraction") or 0.0)
    if stake <= 0:
        return 0.0

    market_prob = float(pick.get("market_prob") or 0.5)
    market_prob = max(0.001, min(0.999, market_prob))
    outcome = pick.get("outcome")
    bet_yes = _bet_yes(pick)

    if bet_yes:
        if outcome == "yes":
            return stake * (1.0 / market_prob - 1.0)
        return -stake

    no_price = 1.0 - market_prob
    if outcome == "no":
        return stake * (1.0 / no_price - 1.0)
    return -stake


def brier_score(pick: dict) -> Optional[float]:
    """(ai_prob - outcome_binary)^2 for resolved picks."""
    ob = _outcome_binary(pick)
    if ob is None:
        return None
    ai_prob = float(pick.get("ai_prob") or 0.5)
    return (ai_prob - ob) ** 2


def cumulative_pnl(
    picks: List[dict],
    bankroll: float = 100.0,
) -> List[Tuple[datetime, float]]:
    """Chronological cumulative realized P&L."""
    resolved = [p for p in picks if _is_resolved(p)]
    resolved.sort(key=_parse_created_at)
    total = 0.0
    series: List[Tuple[datetime, float]] = []
    for p in resolved:
        pnl = realized_pnl(p, bankroll=bankroll)
        if pnl is None:
            continue
        total += pnl
        series.append((_parse_created_at(p), total))
    return series


def rolling_brier(picks: List[dict], window: int = 20) -> List[float]:
    """Rolling mean Brier over resolved picks (chronological)."""
    resolved = [p for p in picks if _is_resolved(p)]
    resolved.sort(key=_parse_created_at)
    scores: List[float] = []
    rolling: List[float] = []
    for p in resolved:
        b = brier_score(p)
        if b is None:
            continue
        scores.append(b)
        start = max(0, len(scores) - window)
        rolling.append(sum(scores[start:]) / len(scores[start:]))
    return rolling


def calibration_bins(picks: List[dict], n_bins: int = 10) -> List[dict]:
    """
    Bin ai_prob into n_bins. Each bin returns predicted mean, actual YES rate, count.
    """
    resolved = [p for p in picks if _is_resolved(p)]
    if not resolved or n_bins < 1:
        return []

    bins: List[List[dict]] = [[] for _ in range(n_bins)]
    for p in resolved:
        prob = float(p.get("ai_prob") or 0.0)
        prob = max(0.0, min(1.0, prob))
        idx = min(n_bins - 1, int(prob * n_bins))
        bins[idx].append(p)

    result: List[dict] = []
    for i, bucket in enumerate(bins):
        if not bucket:
            continue
        lower = i / n_bins
        upper = (i + 1) / n_bins
        prob_mean = sum(float(p.get("ai_prob") or 0) for p in bucket) / len(bucket)
        yes_count = sum(1 for p in bucket if p.get("outcome") == "yes")
        actual_yes_rate = yes_count / len(bucket)
        result.append(
            {
                "bin": i,
                "lower": lower,
                "upper": upper,
                "prob_mean": prob_mean,
                "actual_yes_rate": actual_yes_rate,
                "count": len(bucket),
            }
        )
    return result


def calibration_error(bins: List[dict]) -> float:
    """Expected calibration error (ECE): weighted mean |predicted - actual| per bin."""
    if not bins:
        return 0.0
    total = sum(b["count"] for b in bins)
    if total == 0:
        return 0.0
    err = 0.0
    for b in bins:
        err += b["count"] * abs(b["prob_mean"] - b["actual_yes_rate"])
    return err / total


def hit_rate_by_confidence(picks: List[dict]) -> Dict[str, Dict[str, Any]]:
    """Hit rate per confidence tier (resolved picks only)."""
    tiers = ("low", "medium", "high")
    out: Dict[str, Dict[str, Any]] = {
        t: {"hits": 0, "total": 0, "rate": None} for t in tiers
    }
    for p in picks:
        if not _is_resolved(p):
            continue
        conf = (p.get("confidence") or "low").lower()
        if conf not in out:
            conf = "low"
        out[conf]["total"] += 1
        if _is_hit(p):
            out[conf]["hits"] += 1
    for t in tiers:
        total = out[t]["total"]
        out[t]["rate"] = (out[t]["hits"] / total) if total else None
    return out


def total_stats(picks: List[dict], bankroll: float = 100.0) -> dict:
    """Aggregate credibility stats for dashboards and CLI."""
    resolved = [p for p in picks if _is_resolved(p)]
    hits = sum(1 for p in resolved if _is_hit(p))
    pnls = [realized_pnl(p, bankroll) for p in resolved]
    pnls = [x for x in pnls if x is not None]
    briers = [brier_score(p) for p in resolved]
    briers = [x for x in briers if x is not None]
    bins = calibration_bins(resolved)

    return {
        "total_picks": len(picks),
        "resolved_count": len(resolved),
        "unresolved_count": len(picks) - len(resolved),
        "hit_rate": (hits / len(resolved)) if resolved else None,
        "hits": hits,
        "total_pnl": sum(pnls) if pnls else 0.0,
        "mean_brier": (sum(briers) / len(briers)) if briers else None,
        "ece": calibration_error(bins),
        "resolved_pct": (len(resolved) / len(picks)) if picks else 0.0,
    }
