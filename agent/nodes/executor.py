"""
Executor node — Stage 3 of the Agora LangGraph.

A plain async function (not a ReAct agent) that:
1. Fetches the current Arc wallet balance (for Kelly sizing)
2. For each confirmed pick: writes to SQLite, fires Arc proof tx
3. Attaches arc_tx_hash + arc_explorer_url to each pick record
"""
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

    # Initialise DB
    await store.init_db()

    # Get current wallet balance for Kelly sizing
    wallet_balance = await get_balance()
    state = {**state, "wallet_balance": wallet_balance}

    completed_picks: List[Pick] = []

    for pick in picks:
        try:
            # Send Arc proof transaction
            tx_hash = await send_proof_tx(amount_usdc=0.01)
            explorer_url = get_explorer_url(tx_hash)

            # Attach onchain proof to pick
            pick = pick.model_copy(
                update={
                    "arc_tx_hash": tx_hash,
                    "arc_explorer_url": explorer_url,
                    # Recompute Kelly with actual wallet balance
                    "kelly_fraction": _recompute_kelly(pick, wallet_balance),
                }
            )

            # Persist to SQLite
            row_id = await store.save_pick(pick)
            logger.info(
                f"Pick saved: id={row_id} market={pick.market_id} "
                f"ev={pick.ev:+.2%} tx={tx_hash[:12]}..."
            )
            completed_picks.append(pick)

        except Exception as e:
            logger.error(f"Executor error for pick {pick.market_id}: {e}")
            # Still save pick even without tx hash
            try:
                await store.save_pick(pick)
                completed_picks.append(pick)
            except Exception as save_err:
                logger.error(f"Save failed for {pick.market_id}: {save_err}")

    return {**state, "picks": completed_picks}


def _recompute_kelly(pick: Pick, wallet_balance: float) -> float:
    """Recompute Kelly fraction with real wallet balance."""
    from agent.tools.ev import kelly_fraction
    return round(
        kelly_fraction(pick.market_prob, pick.ai_prob, wallet_balance),
        4,
    )
