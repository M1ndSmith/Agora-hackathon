"""
Agora — Typer CLI entrypoint (stretch goal).

Commands:
  agora init          Generate a wallet keypair
  agora scan          Run the full agent scan
  agora picks         Show recent picks from the database
  agora wallet        Show wallet balance
  agora circle-init   Set up Circle Programmable Wallets (optional)
  agora resolve       Poll Polymarket for resolved pick outcomes
  agora metrics       Show credibility stats (P&L, Brier, hit rate)
  agora portfolio     Portfolio sizing and risk summary
  agora orders        Dry-run CLOB order tickets
  agora arbitrage     Internal Polymarket divergence scan
"""
import asyncio
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint

app = typer.Typer(
    name="agora",
    help="Agora — Prediction Market Intelligence Agent CLI",
    add_completion=False,
)
console = Console()


@app.command()
def init():
    """Generate a new Arc testnet wallet keypair and save to .env."""
    from onchain.wallet import init_wallet

    rprint("[bold yellow]Generating new wallet keypair...[/bold yellow]")
    info = init_wallet()
    rprint(f"[green]✓[/green] Wallet created: [bold]{info['address']}[/bold]")
    rprint("\n[yellow]Next steps:[/yellow]")
    rprint("  1. Fund your wallet from https://faucet.circle.com/")
    rprint("  2. Set ARC_RPC in your .env file")
    rprint("  3. Run: [bold]agora scan[/bold]")


@app.command()
def scan(
    min_ev: float = typer.Option(None, "--min-ev", "-e", help="Minimum EV threshold (default from .env)"),
    min_volume: float = typer.Option(None, "--min-volume", "-v", help="Minimum market volume (default from .env)"),
    top_n: int = typer.Option(None, "--top", "-n", help="Max picks to confirm (default from .env)"),
):
    """Run the full Agora agent pipeline: scan → research → execute."""
    from agent.graph import run_agent

    rprint("[bold blue]Agora Scanner starting...[/bold blue]")

    async def _run():
        return await run_agent(min_ev=min_ev, min_volume=min_volume, top_n=top_n)

    with console.status("[bold green]Running agent pipeline...[/bold green]"):
        final_state = asyncio.run(_run())

    candidates = final_state.get("candidates", [])
    picks = final_state.get("picks", [])
    balance = final_state.get("wallet_balance", 0.0)

    rprint(f"\n[green]✓[/green] Scan complete:")
    rprint(f"  • Candidates found: [bold]{len(candidates)}[/bold]")
    rprint(f"  • Picks confirmed: [bold]{len(picks)}[/bold]")
    rprint(f"  • Wallet balance: [bold]${balance:.4f} USDC[/bold]")

    if picks:
        rprint("\n[bold]Confirmed Picks:[/bold]")
        _print_picks_table(picks)


@app.command()
def picks(
    top: int = typer.Option(10, "--top", "-n", help="Number of picks to show"),
    resolved: bool = typer.Option(False, "--resolved", help="Show only resolved picks"),
):
    """Show recent picks from the database."""
    from db.store import get_pick_history, init_db

    async def _get():
        await init_db()
        return await get_pick_history(resolved_only=resolved)

    rows = asyncio.run(_get())

    if not rows:
        rprint("[yellow]No picks found. Run [bold]agora scan[/bold] first.[/yellow]")
        return

    rows = rows[:top]
    rprint(f"\n[bold]Recent Picks (showing {len(rows)}):[/bold]")
    _print_picks_table(rows)


@app.command()
def wallet():
    """Show Arc testnet wallet balance."""
    from config import get_settings
    from onchain.wallet import get_balance

    settings = get_settings()
    if not settings.agent_address:
        rprint("[red]No wallet configured. Run [bold]agora init[/bold] first.[/red]")
        raise typer.Exit(1)

    rprint(f"[bold]Address:[/bold] {settings.agent_address}")

    async def _bal():
        return await get_balance()

    with console.status("Fetching balance..."):
        balance = asyncio.run(_bal())

    rprint(f"[bold]Arc USDC Balance:[/bold] ${balance:.4f}")


def _print_picks_table(picks):
    """Render picks as a rich table."""
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Question", max_width=50, overflow="ellipsis")
    table.add_column("Market %", justify="right")
    table.add_column("AI %", justify="right")
    table.add_column("EV", justify="right")
    table.add_column("Conf", justify="center")
    table.add_column("Outcome", justify="center")
    table.add_column("Arc TX", max_width=16)

    for p in picks:
        if hasattr(p, "model_dump"):
            row = p.model_dump()
        else:
            row = dict(p)

        ev_pct = (row.get("ev") or 0) * 100
        ev_str = f"[green]+{ev_pct:.1f}%[/green]" if ev_pct >= 0 else f"[red]{ev_pct:.1f}%[/red]"
        conf = row.get("confidence", "low")
        conf_icon = {"high": "[H]", "medium": "[M]", "low": "[L]"}.get(conf, "[-]")
        tx = (row.get("arc_tx_hash") or "—")
        tx_short = tx[:12] + "..." if len(tx) > 12 else tx

        table.add_row(
            row.get("question", "")[:50],
            f"{(row.get('market_prob') or 0):.1%}",
            f"{(row.get('ai_prob') or 0):.1%}",
            ev_str,
            f"{conf_icon} {conf}",
            row.get("outcome") or "Pending",
            tx_short,
        )

    console.print(table)


@app.command()
def resolve():
    """Poll Polymarket and update resolved outcomes for open picks."""
    from agent.tools.outcomes import resolve_open_picks

    with console.status("[bold green]Checking Polymarket resolutions...[/bold green]"):
        summary = asyncio.run(resolve_open_picks())

    rprint(f"\n[green]Done[/green]")
    rprint(f"  Checked:         [bold]{summary['checked']}[/bold]")
    rprint(f"  Newly resolved:  [bold]{summary['newly_resolved']}[/bold]")
    rprint(f"  Still open:      [bold]{summary['still_open']}[/bold]")


@app.command()
def metrics():
    """Print aggregate credibility stats from the picks database."""
    from agent.tools.metrics import total_stats
    from db.store import get_pick_history, init_db

    async def _load():
        await init_db()
        return await get_pick_history()

    picks = asyncio.run(_load())
    if not picks:
        rprint("[yellow]No picks in database. Run [bold]agora scan[/bold] first.[/yellow]")
        raise typer.Exit(0)

    stats = total_stats(picks)
    table = Table(title="Agora Credibility Metrics", show_header=True)
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Total picks", str(stats["total_picks"]))
    table.add_row("Resolved", str(stats["resolved_count"]))
    table.add_row("Unresolved", str(stats["unresolved_count"]))
    hr = stats["hit_rate"]
    table.add_row("Hit rate", f"{hr:.1%}" if hr is not None else "—")
    table.add_row("Total P&L (USDC)", f"${stats['total_pnl']:.4f}")
    mb = stats["mean_brier"]
    table.add_row("Mean Brier", f"{mb:.4f}" if mb is not None else "—")
    table.add_row("Calibration error (ECE)", f"{stats['ece']:.4f}")

    console.print(table)


@app.command()
def portfolio(
    top: int = typer.Option(20, "--top", "-n", help="Max picks to include"),
):
    """Summarize portfolio sizing and risk caps for recent picks."""
    from db.store import get_pick_history, init_db

    async def _load():
        await init_db()
        return await get_pick_history()

    rows = asyncio.run(_load())[:top]
    if not rows:
        rprint("[yellow]No picks found. Run [bold]agora scan[/bold] first.[/yellow]")
        raise typer.Exit(0)

    from config import get_settings
    from agent.tools.risk import assess_risk
    from models import Pick

    settings = get_settings()
    bankroll = settings.fallback_bankroll
    pick_objs = []
    for row in rows:
        try:
            pick_objs.append(Pick(
                market_id=str(row["market_id"]),
                question=row.get("question", ""),
                market_prob=float(row.get("market_prob", 0.5)),
                ai_prob=float(row.get("ai_prob", 0.5)),
                ev=float(row.get("ev", 0)),
                kelly_fraction=float(row.get("kelly_fraction", 0)),
                confidence=row.get("confidence", "low"),
                reasoning_trace=row.get("reasoning_trace", ""),
                domain=row.get("domain", ""),
                signals=row.get("signals") or {},
            ))
        except Exception:
            continue

    pf = assess_risk(pick_objs, rows, bankroll, settings)
    rprint(f"\n[bold]Portfolio[/bold] (bankroll=${pf.bankroll:.2f})")
    rprint(f"  Total exposure: [bold]${pf.total_exposure_usdc:.2f}[/bold]")
    rprint(f"  Drawdown paused: [bold]{pf.drawdown_paused}[/bold]")
    rprint(f"  Theme groups: {len(pf.theme_groups)}")

    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Market", max_width=40)
    table.add_column("Size $", justify="right")
    table.add_column("Theme")
    table.add_column("Warnings")
    for a in pf.assessments[:top]:
        table.add_row(
            a.market_id[:12] + "...",
            f"{a.adjusted_size_usdc:.2f}",
            (a.theme_key or "")[:20],
            ", ".join(a.warnings) or "—",
        )
    console.print(table)


@app.command()
def orders(
    dry_run: bool = typer.Option(True, "--dry-run/--live", help="Show dry-run tickets only"),
    top: int = typer.Option(10, "--top", "-n"),
):
    """Print CLOB order tickets from recent picks (dry-run by default)."""
    from db.store import get_pick_history, init_db

    async def _load():
        await init_db()
        return await get_pick_history()

    rows = asyncio.run(_load())[:top]
    if not rows:
        rprint("[yellow]No picks found.[/yellow]")
        raise typer.Exit(0)

    table = Table(title="Dry-run order tickets", show_header=True)
    table.add_column("Market")
    table.add_column("Side")
    table.add_column("Price")
    table.add_column("Size $")
    table.add_column("Valid")

    for row in rows:
        ex = row.get("execution") or (row.get("signals") or {}).get("order_ticket") or {}
        if isinstance(ex, dict) and ex.get("order_ticket"):
            ex = ex["order_ticket"]
        if not ex:
            continue
        if dry_run and not ex.get("dry_run", True):
            continue
        table.add_row(
            str(row.get("market_id", ""))[:14],
            ex.get("side", "—"),
            f"{ex.get('limit_price', 0):.3f}",
            f"{ex.get('size_usdc', 0):.2f}",
            "yes" if ex.get("valid", True) else "no",
        )
    console.print(table)


@app.command()
def arbitrage(
    limit: int = typer.Option(50, "--limit", "-l", help="Markets to fetch"),
):
    """Scan Polymarket for internal price divergences (similar questions)."""
    from agent.tools.arbitrage import find_internal_price_divergences
    from agent.tools.polymarket import fetch_markets

    async def _run():
        return await fetch_markets(limit=limit)

    with console.status("Fetching markets..."):
        markets = asyncio.run(_run())

    signals = find_internal_price_divergences(markets)
    if not signals:
        rprint("[yellow]No arbitrage signals above threshold.[/yellow]")
        raise typer.Exit(0)

    table = Table(title="Arbitrage signals (internal)", show_header=True)
    table.add_column("Div %", justify="right")
    table.add_column("Sim", justify="right")
    table.add_column("Market A")
    table.add_column("Market B")
    for s in signals[:15]:
        table.add_row(
            f"{s.divergence:.1%}",
            f"{s.similarity:.2f}",
            s.question_a[:35] + "...",
            s.question_b[:35] + "...",
        )
    console.print(table)


@app.command("circle-init")
def circle_init():
    """Set up Circle Programmable Wallets: register secret then create wallet."""
    rprint("[bold yellow]Circle Programmable Wallets Setup[/bold yellow]")
    rprint("Requires CIRCLE_API_KEY and CIRCLE_ENTITY_SECRET in .env\n")

    try:
        from onchain.circle_setup.register_secret import run as register
        from onchain.circle_setup.create_wallet import run as create_wallet
    except ImportError:
        rprint(
            "[red]ERROR: circle SDK not installed.[/red]\n"
            "Run: [bold]pip install circle-sdk[/bold]"
        )
        raise typer.Exit(1)

    rprint("[bold]Step 1 — Register entity secret[/bold]")
    ok = register()
    if not ok:
        rprint("[red]Step 1 failed. Check your .env and try again.[/red]")
        raise typer.Exit(1)
    rprint("[green]✓[/green] Entity secret registered.\n")

    rprint("[bold]Step 2 — Create wallet set + wallet[/bold]")
    try:
        result = create_wallet()
    except Exception as e:
        rprint(f"[red]Step 2 failed: {e}[/red]")
        raise typer.Exit(1)

    rprint("[green]✓[/green] Wallet created.\n")
    rprint("[bold]Add these to your .env:[/bold]")
    rprint(f"  CIRCLE_WALLET_SET_ID={result['wallet_set_id']}")
    rprint(f"  CIRCLE_WALLET_ID={result['wallet_id']}")
    rprint(f"  AGENT_ADDRESS={result['address']}")
    rprint(
        "\n[yellow]Then fund your wallet from the Canteen Discord faucet.[/yellow]"
    )


if __name__ == "__main__":
    app()
