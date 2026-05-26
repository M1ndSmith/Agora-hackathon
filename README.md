# Agora — Prediction Market Intelligence Agent

> **Hackathon:** Agora Agents Hackathon (Canteen x Circle) · RFB-02 Prediction Market Trader Intelligence

A three-stage AI agent that scans Polymarket prediction markets, estimates true event probabilities using LLM reasoning over live news and market history, applies statistical sanity gates to filter false signals, and logs confirmed picks with onchain proof via the Arc testnet. Reasoning traces are gated behind x402 micro-payments.


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
        `--[executor_node]------------- SQLite (agora.db)
             Plain async function       Arc testnet proof tx
                                        0.01 USDC self-transfer
```

### Pipeline Stages


| Stage      | Type                               | Job                                    | Output                    |
| ---------- | ---------------------------------- | -------------------------------------- | ------------------------- |
| Scanner    | Deterministic pipeline             | Broad market scan, Gate 1, EV filter   | `candidates[]`            |
| Researcher | Deterministic pipeline, concurrent | Deep research per candidate, Gates 2+3 | `picks[]` with full trace |
| Executor   | Plain async function               | Persist + Arc proof tx                 | SQLite row + tx hash      |


### Sanity Gates


| Gate | Stage      | Rule                                     | Purpose                                           |
| ---- | ---------- | ---------------------------------------- | ------------------------------------------------- |
| 1    | Scanner    | Drop if market_prob < 3% or > 97%        | Crowd-resolved markets — AI edge is noise         |
| 2    | Researcher | Drop if abs(ai_prob - market_prob) < 5pp | Prevents tiny denominators inflating EV           |
| 3    | Researcher | Drop if logit_distance > 2.5             | Rejects implausible AI estimates (12x odds ratio) |


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

Position sizing uses a conservative quarter-Kelly criterion:

```
kelly = (b * p - q) / b   where b = (1 / market_prob) - 1
position = kelly x bankroll x 0.25
```

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
|-- app.py                  Streamlit UI
|-- main.py                 Typer CLI
|-- config.py               Pydantic settings + LLM provider selection
|-- models.py               Pydantic schemas (Market, Pick, ResearchEstimate)
|-- requirements.txt
|-- .env.example
|-- demo.mp4                Full walkthrough video
|
|-- agent/
|   |-- graph.py            LangGraph StateGraph + custom msgpack serializer
|   |-- nodes/
|   |   |-- scanner.py      Stage 1: deterministic pipeline, Gate 1
|   |   |-- researcher.py   Stage 2: concurrent research, Gates 2+3
|   |   `-- executor.py     Stage 3: persist + Arc proof tx
|   `-- tools/
|       |-- polymarket.py   Gamma API client, shared httpx client
|       |-- ev.py           EV, Kelly, logit_distance math
|       `-- search.py       Tavily wrapper + search_compact()
|
|-- onchain/
|   |-- wallet.py           Arc USDC balance + web3 proof transactions
|   |-- x402.py             Payment verification via eth_getLogs
|   |-- circle_wallet.py    Circle REST client (wallet creation + balance)
|   `-- circle_setup/       One-time Circle setup scripts
|       |-- register_secret.py
|       |-- create_wallet.py
|       `-- README.md
|
`-- db/
    `-- store.py            aiosqlite CRUD for picks + markets
```

---

---

## License

MIT
