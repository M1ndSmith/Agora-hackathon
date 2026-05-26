from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from models import OrderTicket, Pick, PolymarketMarket, ResearchEstimate


def test_polymarket_clamp_prob_low():
    m = PolymarketMarket(
        id="1",
        question="Q?",
        market_prob=0.0,
        volume=1000,
        end_date=datetime.now(timezone.utc),
        url="http://x",
    )
    assert m.market_prob == 0.001


def test_polymarket_clamp_prob_high():
    m = PolymarketMarket(
        id="1",
        question="Q?",
        market_prob=1.0,
        volume=1000,
        end_date=datetime.now(timezone.utc),
        url="http://x",
    )
    assert m.market_prob == 0.999


def test_pick_required_fields():
    with pytest.raises(ValidationError):
        Pick(
            market_id="1",
            question="Q",
            market_prob=0.5,
            ai_prob=0.6,
            ev=0.1,
            # missing kelly_fraction, confidence, reasoning_trace
        )


def test_pick_defaults():
    p = Pick(
        market_id="1",
        question="Q",
        market_prob=0.5,
        ai_prob=0.6,
        ev=0.1,
        kelly_fraction=1.0,
        confidence="low",
        reasoning_trace="test",
    )
    assert p.key_evidence == []
    assert p.created_at is not None
    assert p.created_at.tzinfo is not None


def test_research_estimate_ai_prob_range():
    with pytest.raises(ValidationError):
        ResearchEstimate(
            ai_prob=1.5,
            confidence="high",
            reasoning="x",
        )


def test_research_estimate_valid():
    e = ResearchEstimate(
        ai_prob=0.5,
        confidence="medium",
        reasoning="Because evidence.",
    )
    assert e.ai_prob == 0.5


def test_order_ticket_defaults_dry_run():
    t = OrderTicket(
        market_id="m1",
        side="BUY_YES",
        limit_price=0.55,
        size_usdc=10.0,
    )
    assert t.dry_run is True
    assert t.valid is True
