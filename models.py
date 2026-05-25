from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class PolymarketMarket(BaseModel):
    id: str
    question: str
    market_prob: float
    volume: float
    end_date: datetime
    url: str
    builder_code_url: str = ""

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


class ResearchEstimate(BaseModel):
    """Structured output schema for the researcher's final probability estimate."""
    ai_prob: float = Field(ge=0.0, le=1.0, description="Calibrated YES probability")
    confidence: str = Field(description="low, medium, or high")
    reasoning: str = Field(description="Full evidence-based reasoning chain")
    key_evidence: list = Field(default_factory=list, description="Bullet points of key evidence found")
    bull_case: str = Field(default="", description="Why YES could win")
    bear_case: str = Field(default="", description="Why NO could win")
