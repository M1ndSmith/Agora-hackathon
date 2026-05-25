"""
Agora — Prediction Market Intelligence Agent
Streamlit UI entrypoint
"""
import asyncio
import time
from typing import Any, Dict, List, Optional

import streamlit as st

# ─── Page config (must be first Streamlit call) ──────────────────────────────
st.set_page_config(
    page_title="Agora — Market Intelligence",
    page_icon="A",
    layout="wide",
    initial_sidebar_state="expanded",
)

from agent.graph import run_agent
from config import get_settings, resolve_provider
from db import store
from onchain.wallet import get_balance, init_wallet
from onchain.x402 import get_payment_details, verify_payment


def _run_async(coro):
    """Helper to run async coroutines from sync Streamlit context."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


@st.cache_data(ttl=30, show_spinner=False)
def _cached_balance(address: str) -> float:
    """
    Cache the Arc wallet balance for 30 seconds.

    Streamlit reruns the entire script on every interaction, so without
    this cache we'd hit the Arc RPC on every slider move / form change.
    Keyed by address so multiple wallets cache independently.
    """
    if not address:
        return 0.0
    return _run_async(get_balance(address))


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://raw.githubusercontent.com/twitter/twemoji/master/assets/svg/2696.svg", width=48)
    st.title("Agora")
    st.caption("Prediction Market Intelligence")
    st.divider()

    settings = get_settings()

    st.subheader("Scanner Config")
    min_ev = st.slider(
        "Minimum EV Threshold",
        min_value=0.01,
        max_value=0.50,
        value=float(settings.min_ev_threshold),
        step=0.01,
        help="Edge below this is ignored. EV = (AI prob − market prob) / market prob",
    )
    min_volume = st.number_input(
        "Min Market Volume (USD)",
        min_value=1_000.0,
        max_value=1_000_000.0,
        value=float(settings.min_volume),
        step=1_000.0,
        format="%.0f",
    )
    top_n = st.slider(
        "Max Picks to Confirm",
        min_value=1,
        max_value=20,
        value=int(settings.top_n_picks),
        step=1,
    )

    st.divider()
    st.subheader("Wallet")

    if not settings.agent_address:
        if st.button("Generate Wallet", type="secondary"):
            with st.spinner("Generating keypair..."):
                info = init_wallet()
            st.success(f"Wallet created: `{info['address']}`")
            st.info("Fund it from the Canteen Discord faucet, then restart the app.")
    else:
        st.code(f"{settings.agent_address[:10]}...{settings.agent_address[-6:]}")
        wallet_usdc = _cached_balance(settings.agent_address)
        st.metric("Arc USDC Balance", f"${wallet_usdc:.4f}")

        # Wallet source badge
        if settings.agent_private_key:
            st.info("Signing: Local key (.env)")

    st.divider()

    # Active LLM provider badge
    _provider = resolve_provider(settings)
    _provider_info = {
        "groq":   ("Groq", "green"),
        "nvidia": ("NVIDIA NIM", "violet"),
        "openai": ("OpenAI", "blue"),
    }.get(_provider, ("Unknown", "gray"))
    st.caption(f"LLM: :{_provider_info[1]}[{_provider_info[0]}]")
    _model_label = settings.llm_model or "—"
    st.caption(f"`{_model_label}`")

    st.divider()
    auto_refresh = st.checkbox("Auto-refresh (5 min)", value=False)
    st.caption("Share picks in [Canteen Discord](https://discord.gg/canteen)")

# ─── Main area ───────────────────────────────────────────────────────────────
st.markdown("# Agora — Prediction Market Intelligence")
st.markdown(
    "A two-stage AI agent that scans [Polymarket](https://polymarket.com), "
    "estimates true probabilities with LLM reasoning, and logs high-EV picks "
    "with onchain proof on the Arc testnet. "
    "Full reasoning traces are unlocked via **x402 micro-payments** (0.01 USDC)."
)

tab_scan, tab_history = st.tabs(["Scan & Picks", "History"])


# ─── Helper: render a single pick card ───────────────────────────────────────
def _render_pick_card(
    pdict: dict,
    idx: int,
    pick_db_id: Optional[int] = None,
    key_prefix: str = "pick",
):
    """
    Render a rich pick expander card with:
    - Metrics row (always visible)
    - Key evidence bullets (always visible)
    - Bull / Bear case columns (always visible)
    - x402 gated reasoning trace (blur → pay → reveal)

    `key_prefix` differentiates the same pick rendered in multiple tabs
    (e.g. "scan" vs "history") so Streamlit form keys stay unique.
    """
    ev_pct = (pdict.get("ev") or 0) * 100
    ev_label = f"+{ev_pct:.1f}%" if ev_pct >= 0 else f"{ev_pct:.1f}%"
    conf = pdict.get("confidence") or "low"
    conf_label = {"high": "[H]", "medium": "[M]", "low": "[L]"}.get(conf, "[-]")

    # Unique-per-card key — combines tab, market id, and index
    card_uid = f"{key_prefix}_{pdict.get('market_id', 'unknown')}_{idx}"

    with st.expander(
        f"{conf_label} {pdict.get('question', 'Unknown')} — EV {ev_label}",
        expanded=(idx == 0),
    ):
        # ── Metrics row ──
        col_a, col_b, col_c, col_d, col_e = st.columns(5)
        market_prob = pdict.get("market_prob") or 0
        ai_prob = pdict.get("ai_prob") or 0
        abs_edge_pp = (ai_prob - market_prob) * 100
        col_a.metric("Market Prob", f"{market_prob:.1%}")
        col_b.metric("AI Prob", f"{ai_prob:.1%}")
        col_c.metric("Edge", f"{abs_edge_pp:+.1f}pp")
        col_d.metric("EV", ev_label)
        col_e.metric("Confidence", conf.title())

        # ── Key Evidence ──
        key_evidence = pdict.get("key_evidence") or []
        if isinstance(key_evidence, str):
            import json as _json
            try:
                key_evidence = _json.loads(key_evidence)
            except Exception:
                key_evidence = []

        if key_evidence:
            st.markdown("**Key Evidence**")
            for ev_item in key_evidence:
                st.markdown(f"- {ev_item}")

        # ── Bull / Bear case columns ──
        bull = pdict.get("bull_case") or ""
        bear = pdict.get("bear_case") or ""
        if bull or bear:
            col_bull, col_bear = st.columns(2)
            with col_bull:
                st.markdown("**Bull Case (YES)**")
                st.info(bull or "—")
            with col_bear:
                st.markdown("**Bear Case (NO)**")
                st.warning(bear or "—")

        st.divider()

        # ── x402 Gated Reasoning Trace ──
        x402_receipt = pdict.get("x402_receipt")
        session_key = f"x402_unlocked_{pdict.get('market_id', idx)}"

        # Check session unlock state
        is_unlocked = bool(x402_receipt) or st.session_state.get(session_key, False)

        if is_unlocked:
            st.markdown("**Full Reasoning Trace** (unlocked)")
            if x402_receipt:
                st.caption(f"Unlocked via x402 · receipt: `{x402_receipt[:20]}...`")
            reasoning = pdict.get("reasoning_trace") or "No trace available"
            st.markdown(
                f"<div style='background:#1a2235;padding:1rem;border-radius:8px;"
                f"font-size:0.85rem;max-height:350px;overflow-y:auto;border:1px solid #334;"
                f"color:#e0e6f0'>"
                f"{reasoning.replace(chr(10), '<br>')}"
                f"</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown("**Reasoning Trace** — locked")
            payment_info = get_payment_details()

            # Blurred placeholder
            st.markdown(
                "<div style='background:#1a2235;padding:1rem;border-radius:8px;"
                "font-size:0.85rem;height:120px;filter:blur(4px);overflow:hidden;"
                "border:1px solid #334;color:#e0e6f0;user-select:none'>"
                "The agent has analyzed multiple data sources and identified a significant "
                "probability edge. Market sentiment diverges from fundamental signals. "
                "Key catalysts identified. Full reasoning chain available after payment."
                "</div>",
                unsafe_allow_html=True,
            )

            st.markdown(
                f"**Unlock full trace for `{payment_info['amount_usdc']} USDC`** "
                f"on Arc testnet — send to:"
            )
            st.code(payment_info["to"], language=None)

            with st.form(key=f"x402_form_{card_uid}"):
                payer_address = st.text_input(
                    "Your Arc wallet address (that sent the payment)",
                    placeholder="0x...",
                    key=f"payer_{card_uid}",
                )
                tx_hash_manual = st.text_input(
                    "Or paste your tx hash directly",
                    placeholder="0x...",
                    key=f"txhash_{card_uid}",
                )
                verify_btn = st.form_submit_button("Verify Payment & Unlock")

            if verify_btn:
                if tx_hash_manual and len(tx_hash_manual) == 66:
                    # Direct tx hash path
                    from onchain.x402 import verify_payment_by_hash
                    with st.spinner("Verifying tx on Arc testnet..."):
                        valid = _run_async(verify_payment_by_hash(tx_hash_manual))
                    if valid:
                        st.session_state[session_key] = True
                        if pick_db_id:
                            _run_async(store.update_pick_x402(pick_db_id, tx_hash_manual))
                        st.success("Payment verified! Reasoning trace unlocked.")
                        st.rerun()
                    else:
                        st.error("Payment not found or insufficient. Check your tx hash.")
                elif payer_address:
                    with st.spinner("Scanning Arc testnet for your payment..."):
                        receipt = _run_async(verify_payment(payer_address))
                    if receipt:
                        st.session_state[session_key] = True
                        if pick_db_id:
                            _run_async(store.update_pick_x402(pick_db_id, receipt))
                        st.success(f"Payment verified! Receipt: `{receipt[:20]}...`")
                        st.rerun()
                    else:
                        st.error(
                            "No qualifying payment found in recent blocks. "
                            "Please send the payment and try again."
                        )
                else:
                    st.warning("Enter your wallet address or tx hash to verify.")

        # ── Onchain proof ──
        tx = pdict.get("arc_tx_hash")
        explorer = pdict.get("arc_explorer_url")
        builder = pdict.get("builder_url") or pdict.get("builder_code_url")

        if tx and tx.startswith("circle:"):
            circle_id = tx[len("circle:"):]
            st.markdown(
                f"**Proof (Circle HSM):** `{circle_id[:16]}...` · "
                f"[View on Circle Dashboard]({explorer})"
            )
        elif tx and tx != "0x" + "0" * 64:
            st.markdown(
                f"**Arc Proof:** [`{tx[:16]}...`]({explorer}) · "
                f"[View on Explorer]({explorer})"
            )
        elif tx:
            st.caption(f"Arc proof tx (testnet mock): `{tx[:16]}...`")

        if builder:
            st.markdown(f"[View on Polymarket ↗]({builder})")


# ─── Tab 1: Live Scan ─────────────────────────────────────────────────────────
with tab_scan:
    col_btn, col_status = st.columns([1, 4])
    with col_btn:
        scan_btn = st.button("Scan Markets", type="primary", width="stretch")

    if "last_state" not in st.session_state:
        st.session_state.last_state = {}
    if "scan_running" not in st.session_state:
        st.session_state.scan_running = False

    last = st.session_state.last_state

    # ── Metrics row ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Markets Scanned", last.get("markets_scanned", "—"))
    m2.metric("Candidates", len(last.get("candidates", [])) if last else "—")
    m3.metric("Confirmed Picks", len(last.get("picks", [])) if last else "—")
    # Cached balance — avoids hammering Arc RPC on every Streamlit rerun
    _live_balance = _cached_balance(settings.agent_address or "")
    m4.metric("Wallet USDC", f"${_live_balance:.4f}")

    st.divider()

    if scan_btn:
        st.session_state.scan_running = True
        progress_area = st.empty()

        with progress_area.container():
            st.info("Scanner running — fetching Polymarket markets...")

        with st.spinner("Running Agora agent pipeline..."):
            try:
                final_state = _run_async(
                    run_agent(min_ev=min_ev, min_volume=min_volume, top_n=top_n)
                )
                candidates = final_state.get("candidates", [])
                picks = final_state.get("picks", [])

                st.session_state.last_state = {
                    "markets_scanned": "~50",
                    "candidates": candidates,
                    "picks": picks,
                    "wallet_balance": final_state.get("wallet_balance", 0.0),
                }

                progress_area.success(
                    f"Scan complete — {len(candidates)} candidates, {len(picks)} confirmed picks"
                )
            except Exception as e:
                progress_area.error(f"Agent error: {e}")
                st.exception(e)

        st.session_state.scan_running = False
        st.rerun()

    # ── Picks list ──
    picks = last.get("picks", [])

    if picks:
        st.subheader(f"Confirmed Picks ({len(picks)})")
        for i, pick in enumerate(picks):
            pdict = pick.model_dump() if hasattr(pick, "model_dump") else dict(pick)
            _render_pick_card(pdict, i, key_prefix="scan")
    elif not scan_btn:
        st.info("Hit **Scan Markets** to start the Agora agent pipeline.")


# ─── Tab 2: History ───────────────────────────────────────────────────────────
with tab_history:
    st.subheader("Pick History")

    try:
        history = _run_async(store.get_pick_history())
    except Exception:
        history = []

    if not history:
        st.info("No picks yet. Run a scan to populate history.")
    else:
        st.caption(f"{len(history)} total picks logged")

        # Summary table
        table_data = []
        for row in history:
            ev_pct = (row.get("ev") or 0) * 100
            table_data.append(
                {
                    "Question": (row.get("question") or "")[:65] + "...",
                    "Market %": f"{(row.get('market_prob') or 0):.1%}",
                    "AI %": f"{(row.get('ai_prob') or 0):.1%}",
                    "EV": f"{ev_pct:+.1f}%",
                    "Confidence": (row.get("confidence") or "").title(),
                    "Outcome": row.get("outcome") or "Pending",
                    "Unlocked": "Yes" if row.get("x402_receipt") else "No",
                    "Arc TX": (row.get("arc_tx_hash") or "—")[:12] + "..."
                    if row.get("arc_tx_hash") else "—",
                }
            )

        st.dataframe(table_data, width="stretch")

        st.subheader("Detail View")
        for i, row in enumerate(history[:10]):
            _render_pick_card(row, i, pick_db_id=row.get("id"), key_prefix="history")


# ─── Auto-refresh ─────────────────────────────────────────────────────────────
if auto_refresh and not st.session_state.get("scan_running"):
    time.sleep(300)
    st.rerun()
