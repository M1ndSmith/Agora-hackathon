"""
Executor node — Stage 4 of the Agora LangGraph.

A plain async function (not a ReAct agent) that:
1. Uses wallet balance from portfolio node (or fetches if missing)
2. For each confirmed pick: writes to SQLite, fires Arc proof tx
3. Attaches arc_tx_hash + arc_explorer_url to each pick record

Does not submit CLOB orders — dry-run tickets are stored in execution_json.
"""
import json
import logging
from typing import Any, Dict, List

from db import store
from models import Pick
from onchain.wallet import get_balance, get_explorer_url, send_proof_tx

logger = logging.getLogger(__name__)


async def executor_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    LangGraph node: persists picks and fires Arc proof transactions.
    """
    picks: List[Pick] = state.get("picks", [])
    if not picks:
        return {**state, "wallet_balance": 0.0}

    await store.init_db()

    wallet_balance = float(state.get("wallet_balance") or 0.0)
    if wallet_balance <= 0:
        wallet_balance = await get_balance()
    state = {**state, "wallet_balance": wallet_balance}

    portfolio_json = json.dumps(state.get("portfolio") or {})
    completed_picks: List[Pick] = []

    for pick in picks:
        try:
            tx_hash = await send_proof_tx(amount_usdc=0.01)
            explorer_url = get_explorer_url(tx_hash)

            # Keep portfolio-adjusted kelly from portfolio node
            kelly = pick.kelly_fraction
            if not (pick.signals or {}).get("portfolio_size_usdc"):
                kelly = _recompute_kelly(pick, wallet_balance)

            pick = pick.model_copy(
                update={
                    "arc_tx_hash": tx_hash,
                    "arc_explorer_url": explorer_url,
                    "kelly_fraction": kelly,
                }
            )

            row_id = await store.save_pick(pick, portfolio_json=portfolio_json)
            logger.info(
                f"Pick saved: id={row_id} market={pick.market_id} "
                f"ev={pick.ev:+.2%} size=${kelly:.2f} tx={tx_hash[:12]}..."
            )
            completed_picks.append(pick)

        except Exception as e:
            logger.error(f"Executor error for pick {pick.market_id}: {e}")
            try:
                await store.save_pick(pick, portfolio_json=portfolio_json)
                completed_picks.append(pick)
            except Exception as save_err:
                logger.error(f"Save failed for {pick.market_id}: {save_err}")

    return {**state, "picks": completed_picks}


def _recompute_kelly(pick: Pick, wallet_balance: float) -> float:
    """Fallback side-aware Kelly if portfolio node did not size the pick."""
    from agent.tools.ev import kelly_fraction_two_sided
    return round(
        kelly_fraction_two_sided(pick.market_prob, pick.ai_prob, wallet_balance),
        4,
    )
