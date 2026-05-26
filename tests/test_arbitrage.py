from datetime import datetime, timezone

from agent.tools.arbitrage import (
    find_internal_price_divergences,
    normalize_question,
    similarity_score,
)
from models import PolymarketMarket


def _m(q, p, mid="1"):
    return PolymarketMarket(
        id=mid,
        question=q,
        market_prob=p,
        volume=10000,
        end_date=datetime.now(timezone.utc),
        url="http://x",
    )


def test_normalize_question_strips_punct():
    assert normalize_question("Will Trump win?") == "will trump win"


def test_similarity_score_high_overlap():
    a = "Will Trump win the 2026 US election"
    b = "Will Trump win election in 2026"
    assert similarity_score(a, b) >= 0.35


def test_find_internal_price_divergences():
    markets = [
        _m("Will Trump win the 2026 US election", 0.45, "a"),
        _m("Will Trump win election in 2026", 0.60, "b"),
    ]
    signals = find_internal_price_divergences(markets, min_similarity=0.3, min_divergence=0.05)
    assert len(signals) >= 1
    assert signals[0].divergence >= 0.05
