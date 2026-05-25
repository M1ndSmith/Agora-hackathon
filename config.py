from functools import lru_cache
from typing import Optional

from langchain_core.language_models import BaseChatModel
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # LLM provider selection
    # Values: "groq" | "nvidia" | "openai"
    # If not set, auto-detected from which API key is present.
    llm_provider: Optional[str] = None

    # Groq (fastest free tier, strong tool-calling)
    groq_api_key: Optional[str] = None
    # Default Groq model — llama-3.3-70b-versatile supports tools + streaming
    llm_model: str = "llama-3.3-70b-versatile"

    # NVIDIA NIM / generic OpenAI-compatible endpoint
    llm_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_api_key: str = "changeme"

    # Search
    tavily_api_key: str = "changeme"

    # Arc testnet
    arc_rpc: str = "https://rpc.testnet.arc-node.thecanteenapp.com/v1/changeme"
    agent_address: Optional[str] = None
    agent_private_key: Optional[str] = None

    # Arc USDC contract on testnet (0x3600...0000 is the canonical Arc testnet USDC)
    arc_usdc_contract: str = "0x3600000000000000000000000000000000000000"

    # Agent tuning
    min_ev_threshold: float = 0.05
    min_volume: float = 10_000.0
    top_n_picks: int = 10

    # ── False-signal sanity gates ───────────────────────────────────────────
    # Skip markets the crowd has already effectively resolved.
    extreme_low: float = 0.03
    extreme_high: float = 0.97
    # Minimum absolute probability gap (in [0, 1] space) to accept a pick.
    # Prevents tiny denominators from inflating EV ratios.
    min_abs_edge: float = 0.05
    # Maximum acceptable log-odds disagreement between AI and market.
    # ~2.5 ≈ 12x odds ratio. Larger = LLM is almost certainly hallucinating.
    max_logit_distance: float = 2.5

    # x402 micro-payment price for unlocking a reasoning trace
    x402_price_usdc: float = 0.01
    x402_enabled: bool = True
    # How far back (in blocks) to scan when looking for an x402 payment.
    # Arc testnet is ~0.5s/block, so 50,000 ≈ 7 hours of history.
    x402_scan_blocks: int = 50_000
    # Chunk size for paginated eth_getLogs calls (most RPCs cap ~10k blocks).
    x402_scan_chunk: int = 5_000

    # Circle Programmable Wallets (optional)
    circle_api_key: Optional[str] = None
    circle_entity_secret: Optional[str] = None
    circle_wallet_set_id: Optional[str] = None
    circle_wallet_id: Optional[str] = None

    # Feature flags
    arc_enabled: bool = True


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def resolve_provider(s: Settings) -> str:
    """
    Determine the active LLM provider.

    Priority:
    1. Explicit LLM_PROVIDER env var
    2. GROQ_API_KEY present → groq
    3. LLM_API_KEY starts with 'nvapi-' → nvidia
    4. Default → openai (user sets their own LLM_API_KEY + LLM_BASE_URL)
    """
    if s.llm_provider:
        return s.llm_provider.lower().strip()
    if s.groq_api_key and s.groq_api_key != "changeme":
        return "groq"
    if s.llm_api_key and s.llm_api_key.startswith("nvapi-"):
        return "nvidia"
    return "openai"


def get_llm(streaming: bool = False) -> BaseChatModel:
    """
    Return the configured LangChain chat model.

    Supports three providers selectable via LLM_PROVIDER in .env:
      groq   — ChatGroq  (free, fast, strong tool calling)
      nvidia — ChatOpenAI pointed at NVIDIA NIM endpoint
      openai — ChatOpenAI with user-configured base URL / key
    """
    s = get_settings()
    provider = resolve_provider(s)

    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model=s.llm_model,
            api_key=s.groq_api_key,
            temperature=0.3,
            max_tokens=4096,
            streaming=streaming,
        )

    if provider == "nvidia":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=s.llm_api_key,
            model=s.llm_model,
            temperature=0.3,
            max_tokens=4096,
            streaming=streaming,
        )

    # openai (or any OpenAI-compatible endpoint via LLM_BASE_URL)
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(
        base_url=s.llm_base_url,
        api_key=s.llm_api_key,
        model=s.llm_model,
        temperature=0.3,
        max_tokens=4096,
        streaming=streaming,
    )
