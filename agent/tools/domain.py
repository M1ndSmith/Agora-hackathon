"""
Lightweight domain classifier for market questions.

Routes researcher prompts to domain-specific estimation templates.
"""
from typing import Dict

KEYWORDS: Dict[str, list] = {
    "politics": [
        "election", "president", "vote", "senate", "congress", "governor",
        "democrat", "republican", "trump", "biden", "parliament", "referendum",
        "primary", "ballot", "impeach", "cabinet", "minister",
    ],
    "sports": [
        "win", "match", "game", "vs", "championship", "league", "nfl", "nba",
        "mlb", "soccer", "football", "tennis", "golf", "olympics", "super bowl",
        "playoff", "team", "score", "mvp",
    ],
    "crypto": [
        "btc", "eth", "bitcoin", "ethereum", "crypto", "token", "defi",
        "solana", "blockchain", "altcoin", "sec", "etf", "stablecoin",
        "binance", "coinbase",
    ],
    "science": [
        "vaccine", "trial", "discovery", "study", "fda", "clinical", "drug",
        "disease", "virus", "climate", "nasa", "research", "approval",
    ],
}

_DOMAIN_ORDER = ("politics", "sports", "crypto", "science")


def classify_domain(question: str) -> str:
    """
    Keyword scan on the market question.

    Returns first matching domain or 'general'.
    """
    q = (question or "").lower()
    if not q:
        return "general"

    scores = {d: 0 for d in _DOMAIN_ORDER}
    for domain, words in KEYWORDS.items():
        for w in words:
            if w in q:
                scores[domain] += 1

    best = max(scores.items(), key=lambda x: x[1])
    if best[1] > 0:
        return best[0]
    return "general"


_BASE_ESTIMATION = """You are a calibrated probability forecaster analyzing a Polymarket prediction market.

You will receive:
- The market question
- Current market probability (what bettors collectively think)
- Compact market metadata (volume, liquidity, spread, competitiveness)
- Optional CLOB microstructure (bid/ask spread, depth)
- Optional prior estimate from a previous scan (treat as Bayesian prior if present)
- 2 short credibility-weighted web research snippets

Produce a calibrated YES probability estimate. Be honest — not overconfident.

Consider:
- Base rates for similar events
- Recent developments visible in the research
- Market efficiency (the market price already embeds most public information)
- Quality and recency of the evidence
- Order book signals: wide spread may mean less efficient pricing

Output the structured ResearchEstimate. Reasoning should be 3-5 sentences citing
specific evidence. Key evidence: 3-5 bullets. Bull/bear: strongest single argument each.
"""

_DOMAIN_HINTS = {
    "politics": "Domain: politics. Consider polling error bands, incumbency, turnout models, and late-breaking news.",
    "sports": "Domain: sports. Consider injury reports, home advantage, recent form, and line movement.",
    "crypto": "Domain: crypto. Consider regulatory news, on-chain metrics, and correlation with BTC/ETH.",
    "science": "Domain: science. Consider trial phases, FDA timelines, base rates for drug approvals.",
    "general": "Domain: general. Use conservative base rates when evidence is thin.",
}

ESTIMATION_PROMPTS: Dict[str, str] = {
    domain: f"{_BASE_ESTIMATION}\n\n{_DOMAIN_HINTS[domain]}"
    for domain in ("politics", "sports", "crypto", "science", "general")
}


def get_estimation_prompt(domain: str) -> str:
    return ESTIMATION_PROMPTS.get(domain, ESTIMATION_PROMPTS["general"])
