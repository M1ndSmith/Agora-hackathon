from agent.tools.orders import (
    build_order_ticket,
    limit_price_for_pick,
    side_for_pick,
    validate_order_ticket,
)
from models import MicrostructureSignal, Pick


def _pick():
    return Pick(
        market_id="m1",
        question="Will X?",
        market_prob=0.4,
        ai_prob=0.6,
        ev=0.5,
        kelly_fraction=5.0,
        confidence="medium",
        reasoning_trace="r",
    )


def test_side_for_pick_yes():
    assert side_for_pick(0.6, 0.4) == "BUY_YES"


def test_side_for_pick_no():
    assert side_for_pick(0.3, 0.5) == "BUY_NO"


def test_limit_price_uses_microstructure():
    micro = MicrostructureSignal(best_bid=0.55, best_ask=0.58, spread=0.03, depth_usd=1000)
    assert limit_price_for_pick("BUY_YES", 0.4, micro) == 0.58


def test_build_order_ticket_dry_run():
    ticket = build_order_ticket(_pick(), size_usdc=8.0, dry_run=True)
    assert ticket.dry_run is True
    assert ticket.side == "BUY_YES"
    assert ticket.size_usdc == 8.0


def test_validate_rejects_oversize_vs_depth():
    micro = MicrostructureSignal(best_bid=0.5, best_ask=0.52, spread=0.02, depth_usd=10.0)
    ticket = build_order_ticket(_pick(), size_usdc=100.0, microstructure=micro)
    valid, notes = validate_order_ticket(ticket, microstructure=micro)
    assert valid is False
    assert any("depth" in n for n in notes)
