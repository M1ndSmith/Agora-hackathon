import pytest

from agent.tools.risk import (
    cap_position_size,
    cap_total_exposure,
    drawdown_paused,
    recommend_early_close,
    recommend_hedge,
    theme_key,
)


def test_theme_key_includes_domain():
    tk = theme_key("Will Trump win the 2026 election?", "politics")
    assert tk.startswith("politics:")


def test_cap_position_size():
    adj, capped = cap_position_size(50.0, bankroll=100.0, max_fraction=0.10)
    assert adj == 10.0
    assert capped is True


def test_cap_total_exposure_rescales():
    recs = [
        {"theme_key": "a", "size_usdc": 20.0},
        {"theme_key": "b", "size_usdc": 20.0},
    ]
    out = cap_total_exposure(recs, bankroll=100.0, max_fraction=0.35)
    total = sum(r["size_usdc"] for r in out)
    assert total == pytest.approx(35.0, rel=0.01)


def test_drawdown_paused_when_pnl_bad():
    history = [
        {"resolved": 1, "outcome": "no", "ai_prob": 0.8, "market_prob": 0.5,
         "kelly_fraction": 10.0, "created_at": "2026-01-01T00:00:00+00:00"},
    ]
    assert drawdown_paused(history, bankroll=100.0, threshold=-0.05) is True


def test_recommend_hedge_yes_pick_price_fell():
    pick = {"market_id": "m1", "market_prob": 0.6, "ai_prob": 0.7}
    h = recommend_hedge(pick, current_market_prob=0.45, threshold=0.10)
    assert h.suggested is True
    assert h.hedge_side == "BUY_NO"


def test_recommend_early_close_take_profit():
    pick = {"market_id": "m1", "market_prob": 0.4, "ai_prob": 0.7}
    e = recommend_early_close(pick, current_market_prob=0.55, profit_threshold=0.10, loss_threshold=-0.15)
    assert e.action == "take_profit"
