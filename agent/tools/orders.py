"""
Dry-run Polymarket CLOB order ticket builder.

Builds and validates order payloads without signing or submitting.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

from config import Settings, get_settings
from models import MicrostructureSignal, OrderTicket, Pick


def side_for_pick(ai_prob: float, market_prob: float) -> str:
    """Agent bets YES when ai_prob >= market_prob, else NO."""
    if ai_prob >= market_prob:
        return "BUY_YES"
    return "BUY_NO"


def limit_price_for_pick(
    side: str,
    market_prob: float,
    microstructure: Optional[MicrostructureSignal] = None,
) -> float:
    """
    Limit price for dry-run ticket: use best ask for BUY_YES, best bid for NO.
    Falls back to market_prob when book unavailable.
    """
    if microstructure is not None:
        if side == "BUY_YES":
            return round(microstructure.best_ask, 4)
        return round(1.0 - microstructure.best_bid, 4)
    if side == "BUY_YES":
        return round(market_prob, 4)
    return round(1.0 - market_prob, 4)


def build_order_ticket(
    pick: Pick,
    size_usdc: float,
    microstructure: Optional[MicrostructureSignal] = None,
    dry_run: bool = True,
) -> OrderTicket:
    """Build a dry-run CLOB order ticket from a sized pick."""
    side = side_for_pick(pick.ai_prob, pick.market_prob)
    price = limit_price_for_pick(side, pick.market_prob, microstructure)
    notes: List[str] = []

    ticket = OrderTicket(
        dry_run=dry_run,
        market_id=pick.market_id,
        side=side,
        limit_price=price,
        size_usdc=round(max(0.0, size_usdc), 4),
        reason="positive edge after portfolio risk caps",
        valid=True,
        validation_notes=notes,
    )
    validated, notes = validate_order_ticket(ticket, microstructure=microstructure)
    ticket.valid = validated
    ticket.validation_notes = notes
    return ticket


def validate_order_ticket(
    ticket: OrderTicket,
    settings: Optional[Settings] = None,
    microstructure: Optional[MicrostructureSignal] = None,
) -> tuple:
    """
    Validate dry-run ticket: price bounds, min size, liquidity.
    Returns (valid, notes).
    """
    settings = settings or get_settings()
    notes: List[str] = []
    valid = True

    if not (0.001 <= ticket.limit_price <= 0.999):
        valid = False
        notes.append("limit_price out of bounds")

    if ticket.size_usdc <= 0:
        valid = False
        notes.append("size_usdc must be positive")

    if microstructure is not None and ticket.size_usdc > 0:
        usable = microstructure.depth_usd * settings.slippage_depth_fraction
        if ticket.size_usdc > usable and usable > 0:
            notes.append(f"size exceeds {settings.slippage_depth_fraction:.0%} of book depth")
            valid = False

    if ticket.dry_run and settings.execution_dry_run:
        notes.append("dry_run: not submitted to CLOB")

    return valid, notes


def ticket_to_dict(ticket: OrderTicket) -> Dict[str, Any]:
    return ticket.model_dump()
