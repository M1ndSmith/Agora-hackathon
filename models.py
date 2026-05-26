from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class PolymarketMarket(BaseModel):
    id: str
    question: str
    market_prob: float
    volume: float
    end_date: datetime
    url: str
    builder_code_url: str = ""
    clob_token_id: Optional[str] = None

    @field_validator("market_prob", mode="before")
    @classmethod
    def clamp_prob(cls, v: float) -> float:
        return max(0.001, min(0.999, float(v)))


class Pick(BaseModel):
    market_id: str
    question: str
    market_prob: float
    ai_prob: float
    ev: float
    kelly_fraction: float
    confidence: str  # low / medium / high
    reasoning_trace: str
    # Structured researcher fields (Improvement 3)
    key_evidence: list = Field(default_factory=list)
    bull_case: str = ""
    bear_case: str = ""
    arc_tx_hash: Optional[str] = None
    arc_explorer_url: Optional[str] = None
    builder_url: str = ""
    created_at: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    outcome: Optional[str] = None  # yes / no / null
    # x402 payment receipt — set when user unlocks the reasoning trace
    x402_receipt: Optional[str] = None
    # Tier 2 intelligence signals (domain, CLOB, ensemble, prior, sources)
    domain: str = ""
    signals: Dict[str, Any] = Field(default_factory=dict)


class ScannerCandidate(BaseModel):
    """Single scanner shortlist entry."""
    market_id: str
    question: str
    market_prob: float = Field(ge=0.0, le=1.0)
    ai_prob: float = Field(ge=0.0, le=1.0)


class ScannerCandidates(BaseModel):
    """Structured scanner output."""
    candidates: List[ScannerCandidate] = Field(default_factory=list)


class MicrostructureSignal(BaseModel):
    """Polymarket CLOB order book summary."""
    best_bid: float = Field(ge=0.0, le=1.0)
    best_ask: float = Field(ge=0.0, le=1.0)
    spread: float = Field(ge=0.0)
    depth_usd: float = Field(ge=0.0)


class EnsembleEstimate(BaseModel):
    """Aggregated multi-provider probability estimate."""
    ai_prob: float = Field(ge=0.0, le=1.0)
    spread: float = Field(ge=0.0)
    providers: List[str] = Field(default_factory=list)
    downgrade_confidence: bool = False
    reasoning: str = ""
    key_evidence: List[str] = Field(default_factory=list)
    bull_case: str = ""
    bear_case: str = ""
    confidence: str = "medium"


class OrderTicket(BaseModel):
    """Dry-run Polymarket CLOB order ticket (not submitted)."""
    dry_run: bool = True
    market_id: str
    side: str  # BUY_YES | BUY_NO
    limit_price: float = Field(ge=0.0, le=1.0)
    size_usdc: float = Field(ge=0.0)
    reason: str = ""
    valid: bool = True
    validation_notes: List[str] = Field(default_factory=list)


class RiskAssessment(BaseModel):
    """Per-pick risk output from portfolio sizing."""
    market_id: str
    theme_key: str = ""
    raw_size_usdc: float = 0.0
    adjusted_size_usdc: float = 0.0
    warnings: List[str] = Field(default_factory=list)
    capped: bool = False


class PortfolioRecommendation(BaseModel):
    """Portfolio-level sizing summary."""
    bankroll: float
    total_exposure_usdc: float
    pick_count: int
    drawdown_paused: bool = False
    theme_groups: Dict[str, List[str]] = Field(default_factory=dict)
    assessments: List[RiskAssessment] = Field(default_factory=list)


class HedgeRecommendation(BaseModel):
    """Advisory hedge when price moves against the pick."""
    market_id: str
    suggested: bool = False
    reason: str = ""
    hedge_side: str = ""
    move_pp: float = 0.0


class EarlyCloseRecommendation(BaseModel):
    """Advisory take-profit or cut-loss before resolution."""
    market_id: str
    action: str = ""  # take_profit | cut_loss | hold
    reason: str = ""
    unrealized_ev: float = 0.0


class ArbitrageSignal(BaseModel):
    """Internal cross-market price divergence signal."""
    market_a_id: str
    market_b_id: str
    question_a: str
    question_b: str
    prob_a: float
    prob_b: float
    divergence: float
    similarity: float
    note: str = ""


class ResearchEstimate(BaseModel):
    """Structured output schema for the researcher's final probability estimate."""
    ai_prob: float = Field(ge=0.0, le=1.0, description="Calibrated YES probability")
    confidence: str = Field(description="low, medium, or high")
    reasoning: str = Field(description="Full evidence-based reasoning chain")
    key_evidence: list = Field(default_factory=list, description="Bullet points of key evidence found")
    bull_case: str = Field(default="", description="Why YES could win")
    bear_case: str = Field(default="", description="Why NO could win")
