# Agora — Prediction Market Intelligence Agent

> **Hackathon:** Agora Agents Hackathon (Canteen x Circle) · RFB-02 Prediction Market Trader Intelligence

A three-stage AI agent that scans Polymarket prediction markets, estimates true event probabilities using LLM reasoning over live news and market history, applies statistical sanity gates to filter false signals, and logs confirmed picks with onchain proof via the Arc testnet. Reasoning traces are gated behind x402 micro-payments.

**The agent's reasoning trace is the product** — not just a number, but a legible, auditable chain of thought for every pick, with cryptographic proof of when it was made.

---

## Demo

A full walkthrough of the app is included in the repo: [`demo.mp4`](demo.mp4).

It shows a live scan, the confirmed picks with bull/bear cases, the Arc onchain proof transaction, and the x402 micro-payment flow that unlocks a reasoning trace.

---

## Architecture

```
Streamlit UI / Typer CLI
        |
        v
LangGraph Orchestrator
        |
        |--[scanner_node]------------- Polymarket Gamma API
        |    Deterministic pipeline     2 parallel Tavily searches
        |    Gate 1: drop extreme       One structured LLM call
        |    prices (< 3% or > 97%)     EV filter
        |         | candidates found
        |--[researcher_node]----------- Market history fetch
        |    Concurrent per candidate   2 parallel Tavily searches
        |    Gate 2: min abs edge       One structured LLM call
        |    Gate 3: logit distance     ResearchEstimate output
        |         |
        |--[portfolio_node]------------ Risk caps + dry-run CLOB tickets
        |         |
        `--[executor_node]------------- SQLite (agora.db)
             Plain async function       Arc testnet proof tx
                                        0.01 USDC self-transfer
```

### Pipeline Stages


| Stage      | Type                               | Job                                    | Output                    |
| ---------- | ---------------------------------- | -------------------------------------- | ------------------------- |
| Scanner    | Deterministic pipeline             | Broad market scan, Gate 1, EV filter   | `candidates[]`            |
| Researcher | Deterministic pipeline, concurrent | Deep research per candidate, Gates 2+3 | `picks[]` with full trace |
| Portfolio  | Deterministic risk + sizing        | Risk caps, slippage cap, dry-run tickets | `portfolio[]`, `risk_summary` |
| Executor   | Plain async function               | Persist + Arc proof tx                 | SQLite row + tx hash      |


### Sanity Gates


| Gate | Stage      | Rule                                     | Purpose                                           |
| ---- | ---------- | ---------------------------------------- | ------------------------------------------------- |
| 1    | Scanner    | Drop if market_prob < 3% or > 97%        | Crowd-resolved markets — AI edge is noise         |
| 2    | Researcher | Drop if abs(ai_prob - market_prob) < 5pp | Prevents tiny denominators inflating EV           |
| 3    | Researcher | Drop if logit_distance > 2.5             | Rejects implausible AI estimates (12x odds ratio) |


### Tier 2 intelligence (shipped)

| Feature | What it does | Default |
| ------- | ------------ | ------- |
| Source credibility | Tavily snippets ranked by outlet tier + recency | On |
| CLOB microstructure | Bid/ask spread and depth from Polymarket order book | On (`CLOB_ENABLED=true`) |
| Domain prompts | Politics / sports / crypto / science estimation templates | On |
| Bayesian prior | Re-scan injects last unresolved `ai_prob` for same market | On |
| LLM ensemble | Median across groq + nvidia + openai when all keys set | Off (`ENSEMBLE_ENABLED=false`) |
| Scanner structured output | `ScannerCandidates` Pydantic schema (regex fallback) | On |

### Tier 3 execution (dry-run, shipped)

| Feature | What it does | Default |
| ------- | ------------ | ------- |
| Portfolio node | Risk caps + theme exposure + slippage-aware sizing | On |
| Dry-run CLOB tickets | `BUY_YES` / `BUY_NO` limit orders (not submitted) | On |
| Hedge / early-close | Advisory when price moves for/against position | On |
| Arbitrage scan | Internal Polymarket question similarity + price divergence | On |

Pipeline: `scanner -> researcher -> portfolio -> executor`. CLI: `python main.py portfolio`, `orders --dry-run`, `arbitrage`.

Each pick card shows a **Signals** line: domain, top source score, CLOB spread/depth, ensemble spread (if enabled), prior update (on re-scan).

The Scan tab also shows an **Execution Summary** (bankroll, suggested exposure, drawdown pause) and per-pick **Execution** lines (size, dry-run ticket, risk cap reason). Empty hedge/early-close advisories are hidden so only actionable lines appear.

Optional env flags:

```bash
ENSEMBLE_ENABLED=true
ENSEMBLE_DISAGREEMENT_THRESHOLD=0.15
DOMAIN_ROUTING_ENABLED=true
CLOB_ENABLED=true
SOCIAL_ENABLED=false
TWITTER_BEARER_TOKEN=
```

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/M1ndSmith/Agora-hackathon
cd Agora-hackathon
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env — pick one LLM provider block
```

Required keys:


| Key              | Where to get it                                                   |
| ---------------- | ----------------------------------------------------------------- |
| `GROQ_API_KEY`   | [console.groq.com](https://console.groq.com) — free               |
| `TAVILY_API_KEY` | [app.tavily.com](https://app.tavily.com) — 1k free searches/month |
| `ARC_RPC`        | Canteen Discord                                                   |


LLM options (set one block in `.env`):

```bash
# Option A — Groq (recommended)
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_...
LLM_MODEL=llama-3.3-70b-versatile

# Option B — NVIDIA NIM
# LLM_PROVIDER=nvidia
# LLM_API_KEY=nvapi-...
# LLM_MODEL=meta/llama-3.1-70b-instruct

# Option C — OpenAI
# LLM_PROVIDER=openai
# LLM_API_KEY=sk-...
# LLM_MODEL=gpt-4o-mini
```

### 3. Generate an Arc wallet

```bash
python main.py init
# Prints your wallet address — fund it from the Canteen Discord faucet
```

### 4. Run the app

```bash
# Streamlit UI
streamlit run app.py

# Or CLI
python main.py scan
python main.py picks --top 10
python main.py wallet
python main.py resolve
python main.py metrics
python main.py portfolio
python main.py orders --dry-run
python main.py arbitrage
```

### Tests

```bash
pip install -r requirements.txt
# If pytest fails due to ROS plugins on your machine:
./scripts/run_tests.sh
# Or:
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/
```

Unit tests cover EV math (including two-sided Kelly), Tier 1 metrics, Tier 2 modules (credibility, CLOB, domain, ensemble, prior, scanner), Tier 3 modules (risk, orders, arbitrage, portfolio node), Polymarket parsers, and onchain helpers (no live LLM or HTTP).

With coverage:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q tests/ --cov=agent --cov=onchain --cov-report=term-missing
```

### 5. Circle setup (optional)

```bash
# One command sets up Circle Programmable Wallets
python main.py circle-init
```

---

## EV Math

```
EV = (ai_prob - market_prob) / market_prob
```

Position sizing uses a conservative quarter-Kelly criterion on both sides (`kelly_fraction_two_sided` in `agent/tools/ev.py`):

```
# YES-side edge (bet YES)
kelly = (b * p - q) / b   where b = (1 / market_prob) - 1, p = ai_prob, q = 1 - p

# NO-side edge (bet NO) — mirrors probabilities before the same formula
kelly_no = kelly with p' = 1 - ai_prob, market' = 1 - market_prob

raw_position = kelly * bankroll * 0.25
```

The **portfolio node** applies risk caps first (per-market, total exposure, theme), then caps size by CLOB depth:

```
position = min(risk_capped_size, depth_at_price * SLIPPAGE_DEPTH_FRACTION)
```

No real orders are submitted (`EXECUTION_DRY_RUN=true`); validated tickets are advisory only.

---

## Onchain Proof

Each confirmed pick triggers a minimal USDC self-transfer on Arc testnet. The block timestamp is immutable — anyone can verify when this pick was logged, making it tamper-evident and impossible to backdate.

The `arc_tx_hash` and Arc explorer URL are stored with every pick in SQLite and displayed in the UI.

---

## x402 Micro-Payments

Reasoning traces are blurred by default. To unlock:

1. Send 0.01 USDC to the agent's Arc testnet address
2. Paste your sending wallet address (or tx hash) into the unlock form
3. The app scans Arc testnet for the matching ERC-20 Transfer event
4. If found, the trace is revealed and the receipt is stored permanently

No smart contract required — verification reads standard ERC-20 logs directly.

---

## File Structure

```
agora-hackathon/
|-- app.py                  Streamlit UI (Scan, History, Leaderboard)
|-- main.py                 Typer CLI
|-- config.py               Pydantic settings + LLM provider selection
|-- models.py               Pydantic schemas (Pick, ResearchEstimate, Tier 3 models)
|-- requirements.txt
|-- .env.example
|-- demo.mp4                Full walkthrough video
|-- CODEBASE_GUIDE.md       Developer map of the repo
|-- PREREQUISITES.md        Setup checklist
|-- FUTURE_IMPROVEMENTS.md  Roadmap beyond hackathon
|
|-- agent/
|   |-- graph.py            LangGraph: scanner -> researcher -> portfolio -> executor
|   |-- nodes/
|   |   |-- scanner.py      Stage 1: Gate 1, structured shortlist, arbitrage hints
|   |   |-- researcher.py   Stage 2: evidence + estimate + Gates 2+3
|   |   |-- portfolio.py    Stage 3: risk caps, slippage sizing, dry-run tickets
|   |   `-- executor.py     Stage 4: SQLite + Arc proof tx
|   `-- tools/
|       |-- polymarket.py   Gamma API client
|       |-- ev.py           EV, Kelly (two-sided), slippage-aware sizing
|       |-- search.py       Tavily + search_weighted() credibility ranking
|       |-- credibility.py  Outlet tier + recency scoring
|       |-- clob.py         Polymarket CLOB microstructure
|       |-- domain.py       Domain classification + prompts
|       |-- prior.py        Bayesian prior from last unresolved pick
|       |-- ensemble.py     Multi-LLM median (opt-in)
|       |-- social.py       Social sentiment stub
|       |-- outcomes.py     Resolved market polling
|       |-- metrics.py      P&L, Brier, calibration
|       |-- risk.py         Exposure caps, hedge/early-close advisories
|       |-- orders.py       Dry-run CLOB ticket build + validate
|       `-- arbitrage.py    Internal price-divergence scanner
|
|-- onchain/
|   |-- wallet.py           Arc USDC balance + web3 proof transactions
|   |-- x402.py             Payment verification via eth_getLogs
|   |-- circle_wallet.py    Circle REST client (optional)
|   `-- circle_setup/       One-time Circle setup scripts
|
|-- db/
|   `-- store.py            aiosqlite CRUD (picks, signals, portfolio, execution JSON)
|
`-- tests/                  Unit tests (no live LLM or HTTP)
```

---

---

## License

MIT
