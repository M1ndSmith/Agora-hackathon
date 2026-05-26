"""Shared fixtures for Agora unit tests."""
from datetime import datetime, timezone

import pytest


@pytest.fixture
def bankroll():
    return 100.0


@pytest.fixture
def resolved_pick_yes_win(bankroll):
    return {
        "id": 1,
        "market_id": "m1",
        "question": "Will X happen?",
        "market_prob": 0.4,
        "ai_prob": 0.7,
        "ev": 0.75,
        "kelly_fraction": 5.0,
        "confidence": "high",
        "resolved": 1,
        "outcome": "yes",
        "created_at": "2026-05-01T12:00:00+00:00",
    }


@pytest.fixture
def resolved_pick_yes_loss(bankroll):
    return {
        "id": 2,
        "market_id": "m2",
        "question": "Will Y happen?",
        "market_prob": 0.5,
        "ai_prob": 0.65,
        "ev": 0.3,
        "kelly_fraction": 3.0,
        "confidence": "medium",
        "resolved": 1,
        "outcome": "no",
        "created_at": "2026-05-02T12:00:00+00:00",
    }


@pytest.fixture
def resolved_pick_no_win(bankroll):
    return {
        "id": 3,
        "market_id": "m3",
        "question": "Will Z happen?",
        "market_prob": 0.7,
        "ai_prob": 0.3,
        "ev": -0.57,
        "kelly_fraction": 4.0,
        "confidence": "high",
        "resolved": 1,
        "outcome": "no",
        "created_at": "2026-05-03T12:00:00+00:00",
    }


@pytest.fixture
def resolved_pick_no_loss(bankroll):
    return {
        "id": 4,
        "market_id": "m4",
        "question": "Will W happen?",
        "market_prob": 0.6,
        "ai_prob": 0.35,
        "kelly_fraction": 2.0,
        "confidence": "low",
        "resolved": 1,
        "outcome": "yes",
        "created_at": "2026-05-04T12:00:00+00:00",
    }


@pytest.fixture
def unresolved_pick(bankroll):
    return {
        "id": 5,
        "market_id": "m5",
        "question": "Open market?",
        "market_prob": 0.5,
        "ai_prob": 0.55,
        "kelly_fraction": 1.0,
        "confidence": "low",
        "resolved": 0,
        "outcome": None,
        "created_at": "2026-05-05T12:00:00+00:00",
    }


@pytest.fixture
def well_calibrated_picks():
    """Synthetic picks: predicted prob bands align with outcome rates."""
    picks = []
    for i in range(100):
        bucket = i % 10
        ai_prob = (bucket + 0.5) / 10.0
        outcome = "yes" if (i % 10) < bucket + 1 else "no"
        picks.append(
            {
                "id": i,
                "market_prob": 0.5,
                "ai_prob": ai_prob,
                "kelly_fraction": 1.0,
                "confidence": "medium",
                "resolved": 1,
                "outcome": outcome,
                "created_at": datetime(2026, 5, 1, tzinfo=timezone.utc).isoformat(),
            }
        )
    return picks


@pytest.fixture
def overconfident_picks():
    """High ai_prob but mostly NO outcomes."""
    picks = []
    for i in range(20):
        picks.append(
            {
                "id": i,
                "market_prob": 0.5,
                "ai_prob": 0.9,
                "kelly_fraction": 2.0,
                "confidence": "high",
                "resolved": 1,
                "outcome": "no",
                "created_at": f"2026-05-{i+1:02d}T12:00:00+00:00",
            }
        )
    return picks
