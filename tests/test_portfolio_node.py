import asyncio
from unittest.mock import AsyncMock, patch

from agent.nodes.portfolio import portfolio_node
from models import Pick


def _pick():
    return Pick(
        market_id="m1",
        question="Will Trump win the 2026 election?",
        market_prob=0.45,
        ai_prob=0.60,
        ev=0.33,
        kelly_fraction=8.0,
        confidence="medium",
        reasoning_trace="test",
        domain="politics",
        signals={"clob_bid": 0.44, "clob_ask": 0.46, "clob_spread": 0.02, "clob_depth_usd": 500},
    )


def test_portfolio_node_attaches_execution_signals():
    state = {"picks": [_pick()], "wallet_balance": 100.0}

    with patch("agent.nodes.portfolio.get_market_history", new_callable=AsyncMock) as mock_hist:
        mock_hist.return_value = {"outcomePrices": '["0.50", "0.50"]'}
        with patch("db.store.init_db", new_callable=AsyncMock):
            with patch("db.store.get_pick_history", new_callable=AsyncMock, return_value=[]):
                out = asyncio.run(portfolio_node(state))

    assert len(out["picks"]) == 1
    sig = out["picks"][0].signals
    assert "order_ticket" in sig
    assert sig.get("dry_run") is True
    assert "portfolio_size_usdc" in sig
    assert out.get("portfolio") is not None
    assert out.get("risk_summary") is not None
