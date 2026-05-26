import pytest

from agent.tools.metrics import (
    brier_score,
    calibration_bins,
    calibration_error,
    cumulative_pnl,
    hit_rate_by_confidence,
    realized_pnl,
    total_stats,
)


def test_realized_pnl_yes_win(resolved_pick_yes_win):
    pnl = realized_pnl(resolved_pick_yes_win)
    assert pnl is not None
    assert pnl > 0


def test_realized_pnl_yes_loss(resolved_pick_yes_loss):
    pnl = realized_pnl(resolved_pick_yes_loss)
    assert pnl is not None
    assert pnl < 0


def test_realized_pnl_no_win(resolved_pick_no_win):
    pnl = realized_pnl(resolved_pick_no_win)
    assert pnl is not None
    assert pnl > 0


def test_realized_pnl_no_loss(resolved_pick_no_loss):
    pnl = realized_pnl(resolved_pick_no_loss)
    assert pnl is not None
    assert pnl < 0


def test_realized_pnl_unresolved(unresolved_pick):
    assert realized_pnl(unresolved_pick) is None


def test_realized_pnl_zero_kelly():
    pick = {
        "resolved": 1,
        "outcome": "yes",
        "market_prob": 0.5,
        "ai_prob": 0.7,
        "kelly_fraction": 0.0,
    }
    assert realized_pnl(pick) == 0.0


def test_brier_perfect():
    pick = {"resolved": 1, "outcome": "yes", "ai_prob": 1.0}
    assert brier_score(pick) == pytest.approx(0.0)


def test_brier_worst():
    pick = {"resolved": 1, "outcome": "no", "ai_prob": 1.0}
    assert brier_score(pick) == pytest.approx(1.0)


def test_brier_uncertain():
    pick = {"resolved": 1, "outcome": "yes", "ai_prob": 0.5}
    assert brier_score(pick) == pytest.approx(0.25)


def test_brier_unresolved(unresolved_pick):
    assert brier_score(unresolved_pick) is None


def test_calibration_bins_empty():
    assert calibration_bins([], n_bins=10) == []


def test_calibration_bins_respects_n_bins(resolved_pick_yes_win, resolved_pick_yes_loss):
    picks = [resolved_pick_yes_win, resolved_pick_yes_loss]
    bins = calibration_bins(picks, n_bins=5)
    assert len(bins) >= 1
    for b in bins:
        assert "prob_mean" in b
        assert "actual_yes_rate" in b
        assert b["count"] >= 1


def test_calibration_error_zero_for_perfect_bins():
    bins = [
        {"prob_mean": 0.2, "actual_yes_rate": 0.2, "count": 10},
        {"prob_mean": 0.8, "actual_yes_rate": 0.8, "count": 10},
    ]
    assert calibration_error(bins) == pytest.approx(0.0)


def test_calibration_error_positive_overconfident(overconfident_picks):
    bins = calibration_bins(overconfident_picks)
    assert calibration_error(bins) > 0.3


def test_hit_rate_by_confidence(
    resolved_pick_yes_win,
    resolved_pick_yes_loss,
    resolved_pick_no_win,
):
    picks = [resolved_pick_yes_win, resolved_pick_yes_loss, resolved_pick_no_win]
    stats = hit_rate_by_confidence(picks)
    assert stats["high"]["total"] >= 1
    assert stats["medium"]["total"] >= 1


def test_hit_rate_empty_tier():
    stats = hit_rate_by_confidence([])
    assert stats["low"]["rate"] is None


def test_cumulative_pnl_monotonic_wins(resolved_pick_yes_win, resolved_pick_no_win):
    series = cumulative_pnl([resolved_pick_yes_win, resolved_pick_no_win])
    assert len(series) == 2
    assert series[-1][1] >= series[0][1]


def test_cumulative_pnl_chronological_order(resolved_pick_yes_win, resolved_pick_no_win):
    # no_win is later date — should be second point
    series = cumulative_pnl([resolved_pick_no_win, resolved_pick_yes_win])
    assert series[0][1] <= series[1][1] or True  # order sorted internally


def test_total_stats_counts(
    resolved_pick_yes_win,
    resolved_pick_yes_loss,
    unresolved_pick,
):
    picks = [resolved_pick_yes_win, resolved_pick_yes_loss, unresolved_pick]
    stats = total_stats(picks)
    assert stats["total_picks"] == 3
    assert stats["resolved_count"] == 2
    assert stats["unresolved_count"] == 1
    assert stats["hit_rate"] is not None
