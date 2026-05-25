# Circle Setup

Two-step setup for Circle Programmable Wallets. Run once, in order.

**Easier:** just run `python main.py circle-init` — it does both steps automatically.

---

## Manual steps

### Step 1 — Register your entity secret

```bash
python -m onchain.circle_setup.register_secret
```

Requires `CIRCLE_API_KEY` and `CIRCLE_ENTITY_SECRET` in `.env`.
Saves a `circle_recovery_*.dat` file in the project root — back this up.

### Step 2 — Create wallet set + wallet

```bash
python -m onchain.circle_setup.create_wallet
```

Prints three values to add to `.env`:
- `CIRCLE_WALLET_SET_ID`
- `CIRCLE_WALLET_ID`
- `AGENT_ADDRESS`

---

## Getting your credentials

| Credential | Where |
|---|---|
| `CIRCLE_API_KEY` | [console.circle.com](https://console.circle.com) → API Keys → Create (use Sandbox) |
| `CIRCLE_ENTITY_SECRET` | Generate locally: `python -c "import secrets; print(secrets.token_hex(32))"` |
| `CIRCLE_WALLET_SET_ID` | Output of Step 2 |
| `CIRCLE_WALLET_ID` | Output of Step 2 |
