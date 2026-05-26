"""
Async SQLite persistence layer using aiosqlite.

Tables:
  markets — raw Polymarket market snapshots
  picks   — confirmed agent picks with reasoning + Arc proof + x402 receipt
"""
import json
from datetime import datetime, timezone
from typing import List, Optional

import aiosqlite

from models import Pick, PolymarketMarket

DB_PATH = "agora.db"


async def init_db(db_path: str = DB_PATH) -> None:
    """Create tables if they don't exist. Uses ALTER TABLE for additive migrations."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS markets (
                id          TEXT PRIMARY KEY,
                question    TEXT,
                market_prob REAL,
                volume      REAL,
                end_date    TEXT,
                fetched_at  TEXT
            )
        """)
        await db.execute("""
            CREATE TABLE IF NOT EXISTS picks (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id         TEXT,
                question          TEXT,
                market_prob       REAL,
                ai_prob           REAL,
                ev                REAL,
                kelly_fraction    REAL,
                confidence        TEXT,
                reasoning_trace   TEXT,
                key_evidence      TEXT,
                bull_case         TEXT,
                bear_case         TEXT,
                arc_tx_hash       TEXT,
                arc_explorer_url  TEXT,
                builder_url       TEXT,
                created_at        TEXT,
                resolved          INTEGER DEFAULT 0,
                outcome           TEXT,
                x402_receipt      TEXT
            )
        """)
        # Additive migrations for existing DBs that predate these columns
        for col, col_type in [
            ("key_evidence", "TEXT"),
            ("bull_case", "TEXT"),
            ("bear_case", "TEXT"),
            ("x402_receipt", "TEXT"),
            ("domain", "TEXT"),
            ("signals_json", "TEXT"),
            ("portfolio_json", "TEXT"),
            ("execution_json", "TEXT"),
        ]:
            try:
                await db.execute(f"ALTER TABLE picks ADD COLUMN {col} {col_type}")
            except Exception:
                pass  # column already exists
        await db.commit()


async def save_market(market: PolymarketMarket, db_path: str = DB_PATH) -> None:
    """Upsert a market snapshot."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """
            INSERT OR REPLACE INTO markets (id, question, market_prob, volume, end_date, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                market.id,
                market.question,
                market.market_prob,
                market.volume,
                market.end_date.isoformat(),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        await db.commit()


def _execution_payload(signals: dict) -> str:
    """Extract Tier 3 execution fields for dedicated DB column."""
    if not signals:
        return "{}"
    payload = {
        k: signals[k]
        for k in (
            "order_ticket",
            "portfolio_size_usdc",
            "hedge",
            "early_close",
            "risk_warnings",
            "dry_run",
        )
        if k in signals
    }
    return json.dumps(payload)


async def save_pick(
    pick: Pick,
    portfolio_json: Optional[str] = None,
    db_path: str = DB_PATH,
) -> int:
    """Insert a pick and return its row id."""
    signals = pick.signals or {}
    execution = _execution_payload(signals)
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute(
            """
            INSERT INTO picks
              (market_id, question, market_prob, ai_prob, ev, kelly_fraction,
               confidence, reasoning_trace, key_evidence, bull_case, bear_case,
               arc_tx_hash, arc_explorer_url, builder_url, created_at,
               resolved, outcome, x402_receipt, domain, signals_json,
               portfolio_json, execution_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pick.market_id,
                pick.question,
                pick.market_prob,
                pick.ai_prob,
                pick.ev,
                pick.kelly_fraction,
                pick.confidence,
                pick.reasoning_trace,
                json.dumps(pick.key_evidence) if pick.key_evidence else "[]",
                pick.bull_case or "",
                pick.bear_case or "",
                pick.arc_tx_hash,
                pick.arc_explorer_url,
                pick.builder_url,
                (pick.created_at or datetime.now(timezone.utc)).isoformat(),
                int(pick.resolved),
                pick.outcome,
                pick.x402_receipt,
                pick.domain or "",
                json.dumps(signals),
                portfolio_json or "{}",
                execution,
            ),
        )
        await db.commit()
        return cursor.lastrowid


async def get_picks(limit: int = 50, db_path: str = DB_PATH) -> List[dict]:
    """Return most recent picks as dicts, newest first."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM picks ORDER BY id DESC LIMIT ?", (limit,)
        )
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["key_evidence"] = _parse_json_list(d.get("key_evidence"))
            d["signals"] = _parse_json_dict(d.get("signals_json"))
            d["portfolio"] = _parse_json_dict(d.get("portfolio_json"))
            d["execution"] = _parse_json_dict(d.get("execution_json"))
            result.append(d)
        return result


async def get_latest_unresolved_pick_for_market(
    market_id: str,
    db_path: str = DB_PATH,
) -> Optional[dict]:
    """Most recent unresolved pick for a market (for Bayesian prior)."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            """
            SELECT * FROM picks
            WHERE market_id = ? AND resolved = 0
            ORDER BY id DESC LIMIT 1
            """,
            (market_id,),
        )
        row = await cursor.fetchone()
        if not row:
            return None
        d = dict(row)
        d["key_evidence"] = _parse_json_list(d.get("key_evidence"))
        d["signals"] = _parse_json_dict(d.get("signals_json"))
        d["portfolio"] = _parse_json_dict(d.get("portfolio_json"))
        d["execution"] = _parse_json_dict(d.get("execution_json"))
        return d


async def get_unresolved_picks(db_path: str = DB_PATH) -> List[dict]:
    """Return picks that are not yet marked resolved."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM picks WHERE resolved = 0 ORDER BY id DESC"
        )
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["key_evidence"] = _parse_json_list(d.get("key_evidence"))
            d["signals"] = _parse_json_dict(d.get("signals_json"))
            d["portfolio"] = _parse_json_dict(d.get("portfolio_json"))
            d["execution"] = _parse_json_dict(d.get("execution_json"))
            result.append(d)
        return result


async def get_pick_history(
    resolved_only: bool = False,
    db_path: str = DB_PATH,
) -> List[dict]:
    """Return all picks optionally filtered to resolved ones."""
    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        if resolved_only:
            cursor = await db.execute(
                "SELECT * FROM picks WHERE resolved = 1 ORDER BY id DESC"
            )
        else:
            cursor = await db.execute("SELECT * FROM picks ORDER BY id DESC")
        rows = await cursor.fetchall()
        result = []
        for row in rows:
            d = dict(row)
            d["key_evidence"] = _parse_json_list(d.get("key_evidence"))
            d["signals"] = _parse_json_dict(d.get("signals_json"))
            d["portfolio"] = _parse_json_dict(d.get("portfolio_json"))
            d["execution"] = _parse_json_dict(d.get("execution_json"))
            result.append(d)
        return result


async def update_pick_outcome(
    pick_id: int,
    outcome: str,
    db_path: str = DB_PATH,
) -> None:
    """Mark a pick as resolved with its outcome (yes/no)."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE picks SET resolved = 1, outcome = ? WHERE id = ?",
            (outcome, pick_id),
        )
        await db.commit()


async def update_pick_x402(
    pick_id: int,
    tx_hash: str,
    db_path: str = DB_PATH,
) -> None:
    """Store x402 payment receipt for a pick (marks reasoning trace as unlocked)."""
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE picks SET x402_receipt = ? WHERE id = ?",
            (tx_hash, pick_id),
        )
        await db.commit()


async def count_picks(db_path: str = DB_PATH) -> int:
    """Total confirmed picks in the DB."""
    async with aiosqlite.connect(db_path) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM picks")
        row = await cursor.fetchone()
        return row[0] if row else 0


def _parse_json_list(value) -> list:
    """Safely parse a JSON list from a SQLite TEXT column."""
    if isinstance(value, list):
        return value
    if not value:
        return []
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        return []


def _parse_json_dict(value) -> dict:
    """Safely parse a JSON dict from a SQLite TEXT column."""
    if isinstance(value, dict):
        return value
    if not value:
        return {}
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}
