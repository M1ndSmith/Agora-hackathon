"""
Cross-market arbitrage scaffolding (Polymarket internal divergences).

No Kalshi/Manifold integration yet — token-overlap similarity only.
"""
from __future__ import annotations

import re
from typing import List, Optional

from config import get_settings
from models import ArbitrageSignal, PolymarketMarket


def normalize_question(question: str) -> str:
    q = (question or "").lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    return " ".join(q.split())


def _tokens(question: str) -> set:
    stop = {
        "will", "what", "when", "that", "this", "with", "from", "have",
        "been", "before", "after", "than", "into", "over", "under", "the",
    }
    return {t for t in normalize_question(question).split() if len(t) >= 3 and t not in stop}


def similarity_score(a: str, b: str) -> float:
    """Jaccard similarity on question tokens."""
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    inter = len(ta & tb)
    union = len(ta | tb)
    return round(inter / union, 4) if union else 0.0


def build_arbitrage_signal(
    market_a: PolymarketMarket,
    market_b: PolymarketMarket,
    sim: float,
) -> ArbitrageSignal:
    div = abs(market_a.market_prob - market_b.market_prob)
    return ArbitrageSignal(
        market_a_id=market_a.id,
        market_b_id=market_b.id,
        question_a=market_a.question[:120],
        question_b=market_b.question[:120],
        prob_a=market_a.market_prob,
        prob_b=market_b.market_prob,
        divergence=round(div, 4),
        similarity=sim,
        note="internal Polymarket divergence; fees not included",
    )


def find_internal_price_divergences(
    markets: List[PolymarketMarket],
    min_similarity: float = 0.35,
    min_divergence: Optional[float] = None,
) -> List[ArbitrageSignal]:
    """
    Pair markets with similar questions but different implied YES prices.
    O(n^2) over shortlist — fine for scanner batch sizes.
    """
    settings = get_settings()
    min_div = min_divergence if min_divergence is not None else settings.arbitrage_min_divergence
    signals: List[ArbitrageSignal] = []

    for i, ma in enumerate(markets):
        for mb in markets[i + 1 :]:
            sim = similarity_score(ma.question, mb.question)
            if sim < min_similarity:
                continue
            div = abs(ma.market_prob - mb.market_prob)
            if div < min_div:
                continue
            signals.append(build_arbitrage_signal(ma, mb, sim))

    signals.sort(key=lambda s: s.divergence, reverse=True)
    return signals
