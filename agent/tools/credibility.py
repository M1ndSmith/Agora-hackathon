"""
Source credibility scoring for web evidence.

Tier-1 outlets score highest; recency decay reduces weight of stale articles.
"""
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

from config import get_settings

TIER1_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "wsj.com",
    "nytimes.com",
    "ft.com",
    "bloomberg.com",
    "economist.com",
}

TIER2_DOMAINS = {
    "cnn.com",
    "theguardian.com",
    "axios.com",
    "politico.com",
    "nbcnews.com",
    "cbsnews.com",
    "washingtonpost.com",
    "forbes.com",
    "cnbc.com",
}


def _extract_domain(url: str) -> str:
    if not url:
        return ""
    try:
        host = urlparse(url if "://" in url else f"https://{url}").netloc.lower()
        if host.startswith("www."):
            host = host[4:]
        return host
    except Exception:
        return ""


def _base_tier_score(domain: str) -> float:
    if not domain:
        return 0.4
    if domain in TIER1_DOMAINS or any(domain.endswith(f".{d}") for d in TIER1_DOMAINS):
        return 1.0
    if domain in TIER2_DOMAINS or any(domain.endswith(f".{d}") for d in TIER2_DOMAINS):
        return 0.8
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.6
    return 0.4


def _recency_factor(
    published_at: Optional[datetime],
    now: Optional[datetime] = None,
) -> float:
    if published_at is None:
        return 0.85
    settings = get_settings()
    decay_days = max(1, settings.recency_decay_days)
    now = now or datetime.now(timezone.utc)
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    age_days = max(0.0, (now - published_at).total_seconds() / 86400.0)
    return max(0.2, 1.0 - age_days / decay_days)


def score_source(
    url: str,
    published_at: Optional[datetime] = None,
    now: Optional[datetime] = None,
) -> float:
    """
    Combined credibility score in [0.2, 1.0].

    tier1=1.0, tier2=0.8, gov/edu=0.6, unknown=0.4, multiplied by recency.
    """
    domain = _extract_domain(url)
    base = _base_tier_score(domain)
    return round(min(1.0, base * _recency_factor(published_at, now)), 3)


def rank_snippets(snippets: list[dict]) -> list[dict]:
    """Sort snippets by credibility score descending."""
    for s in snippets:
        if "score" not in s:
            s["score"] = score_source(
                s.get("url", ""),
                s.get("published_at"),
            )
    return sorted(snippets, key=lambda x: x.get("score", 0), reverse=True)
