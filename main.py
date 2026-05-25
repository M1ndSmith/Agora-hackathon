"""
Agora — Typer CLI entrypoint (stretch goal).

Commands:
  agora init          Generate a wallet keypair
  agora scan          Run the full agent scan
  agora picks         Show recent picks from the database
  agora wallet        Show wallet balance
  agora circle-init   Set up Circle Programmable Wallets (optional)
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
