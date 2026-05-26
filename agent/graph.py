"""
LangGraph orchestration — Agora's two-stage agent pipeline.

Graph:
  scanner_node
      │
      ├─ candidates found ──► researcher_node ──► portfolio_node ──► executor_node ──► END
      │
      └─ no candidates ──────────────────────────────────────────► END

Checkpointing: AsyncSqliteSaver (LangGraph 1.0.8, confirmed via Context7)
"""
import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import END, START, StateGraph

from agent.nodes.executor import executor_node
from agent.nodes.portfolio import portfolio_node
from agent.nodes.researcher import researcher_node
from agent.nodes.scanner import scanner_node
from config import get_settings

logger = logging.getLogger(__name__)

DB_PATH = "agora.db"

# Allow-list our Pydantic models so LangGraph can deserialize them from
# the SQLite checkpoint without emitting the "future-version" warning.
# NOTE: must be passed via the constructor — `with_msgpack_allowlist` is a
# no-op when the existing list is `True` (the default in 4.1.1).
_ALLOWED_TYPES = (
    ("models", "Pick"),
    ("models", "PolymarketMarket"),
    ("models", "ResearchEstimate"),
    ("models", "ScannerCandidate"),
    ("models", "ScannerCandidates"),
    ("models", "MicrostructureSignal"),
    ("models", "EnsembleEstimate"),
    ("models", "OrderTicket"),
    ("models", "PortfolioRecommendation"),
    ("models", "RiskAssessment"),
)
_AGORA_SERDE = JsonPlusSerializer(allowed_msgpack_modules=_ALLOWED_TYPES)


@asynccontextmanager
async def _checkpointer(db_path: str = DB_PATH):
    """Open an AsyncSqliteSaver with our msgpack allow-list applied."""
    async with aiosqlite.connect(db_path) as conn:
        yield AsyncSqliteSaver(conn, serde=_AGORA_SERDE)


def _route_after_scanner(state: Dict[str, Any]) -> str:
    """Conditional edge: go to researcher if candidates exist, else END."""
    candidates = state.get("candidates", [])
    if candidates:
        logger.info(f"Routing to researcher with {len(candidates)} candidates")
        return "researcher"
    logger.info("No candidates found — ending early")
    return END


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph (without checkpointer)."""
    graph = StateGraph(dict)

    graph.add_node("scanner", scanner_node)
    graph.add_node("researcher", researcher_node)
    graph.add_node("portfolio", portfolio_node)
    graph.add_node("executor", executor_node)

    graph.set_entry_point("scanner")

    graph.add_conditional_edges(
        "scanner",
        _route_after_scanner,
        {"researcher": "researcher", END: END},
    )
    graph.add_edge("researcher", "portfolio")
    graph.add_edge("portfolio", "executor")
    graph.add_edge("executor", END)

    return graph


async def run_agent(
    min_ev: Optional[float] = None,
    min_volume: Optional[float] = None,
    top_n: Optional[int] = None,
    thread_id: str = "agora-main",
) -> Dict[str, Any]:
    """
    Run the full Agora agent pipeline with SQLite checkpointing.

    Returns the final state dict with picks, candidates, wallet_balance.
    """
    settings = get_settings()

    initial_state: Dict[str, Any] = {
        "markets": [],
        "candidates": [],
        "picks": [],
        "scan_config": {
            "min_ev": min_ev or settings.min_ev_threshold,
            "min_volume": min_volume or settings.min_volume,
            "top_n": top_n or settings.top_n_picks,
        },
        "wallet_balance": 0.0,
        "portfolio": None,
        "risk_summary": None,
        "arbitrage_signals": [],
        "error": None,
    }

    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    async with _checkpointer() as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        final_state = await app.ainvoke(initial_state, config=config)

    return final_state


async def stream_agent(
    min_ev: Optional[float] = None,
    min_volume: Optional[float] = None,
    top_n: Optional[int] = None,
    thread_id: str = "agora-stream",
):
    """
    Async generator that yields state updates as the agent runs.
    Used by Streamlit for live progress display.
    """
    settings = get_settings()

    initial_state: Dict[str, Any] = {
        "markets": [],
        "candidates": [],
        "picks": [],
        "scan_config": {
            "min_ev": min_ev or settings.min_ev_threshold,
            "min_volume": min_volume or settings.min_volume,
            "top_n": top_n or settings.top_n_picks,
        },
        "wallet_balance": 0.0,
        "portfolio": None,
        "risk_summary": None,
        "arbitrage_signals": [],
        "error": None,
    }

    graph = build_graph()
    config = {"configurable": {"thread_id": thread_id}}

    async with _checkpointer() as checkpointer:
        app = graph.compile(checkpointer=checkpointer)
        async for event in app.astream(initial_state, config=config, stream_mode="updates"):
            yield event
