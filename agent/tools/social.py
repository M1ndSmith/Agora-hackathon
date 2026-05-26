"""
Social sentiment signal fetcher (opt-in stub).

Returns None unless TWITTER_BEARER_TOKEN is set and social_enabled=true.
Real Twitter/Reddit integration can be added here without changing the pipeline.
"""
import logging
from typing import Optional

from config import get_settings

logger = logging.getLogger(__name__)


async def fetch_social_signal(query: str) -> Optional[dict]:
    """
    Fetch social sentiment summary for a market question.

    Returns dict with keys: mention_volume, sentiment, source — or None.
    """
    settings = get_settings()
    if not settings.social_enabled:
        return None
    if not settings.twitter_bearer_token:
        return None

    # Scaffold: API surface ready; implementation deferred
    logger.info(
        "Social signal requested but fetch not implemented yet "
        f"(query={query[:50]}...)"
    )
    return None
