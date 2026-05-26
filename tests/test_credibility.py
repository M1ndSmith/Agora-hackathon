from datetime import datetime, timedelta, timezone

from agent.tools.credibility import rank_snippets, score_source


def test_tier1_domain_scores_high():
    s = score_source("https://www.reuters.com/article/foo")
    # base 1.0 * recency ~0.85 when no published_at
    assert s >= 0.85


def test_tier2_domain_scores_medium():
    s = score_source("https://www.cnn.com/story")
    assert 0.65 <= s <= 0.85


def test_gov_domain_tier3():
    s = score_source("https://www.cdc.gov/report")
    assert 0.5 <= s <= 0.75


def test_unknown_domain_lower():
    s = score_source("https://random-blog.example.net/post")
    assert s <= 0.5


def test_recency_decay_old_article():
    old = datetime.now(timezone.utc) - timedelta(days=60)
    s = score_source("https://www.reuters.com/x", published_at=old)
    assert s >= 0.2


def test_rank_snippets_orders_by_score():
    snippets = [
        {"url": "https://blog.example.com", "content": "a"},
        {"url": "https://www.reuters.com", "content": "b"},
    ]
    ranked = rank_snippets(snippets)
    assert ranked[0]["url"].startswith("https://www.reuters")
