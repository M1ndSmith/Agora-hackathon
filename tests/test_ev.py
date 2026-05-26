import pytest

from agent.tools.ev import (
    calculate_ev,
    confidence_level,
    kelly_fraction,
    kelly_fraction_two_sided,
    logit_distance,
    slippage_aware_kelly,
)


def test_calculate_ev_positive_edge():
    assert calculate_ev(0.4, 0.6) == pytest.approx(0.5)


def test_calculate_ev_negative_edge():
    assert calculate_ev(0.7, 0.5) == pytest.approx((0.5 - 0.7) / 0.7)


def test_calculate_ev_zero_market_prob():
    assert calculate_ev(0.0, 0.5) == 0.0


def test_calculate_ev_no_edge():
    assert calculate_ev(0.5, 0.5) == 0.0


def test_kelly_fraction_positive_edge():
    result = kelly_fraction(0.4, 0.6, bankroll=100.0, fraction=0.25)
    assert result > 0


def test_kelly_fraction_no_edge_returns_zero():
    assert kelly_fraction(0.5, 0.5, bankroll=100.0) == 0.0


def test_kelly_fraction_invalid_market_prob():
    assert kelly_fraction(0.0, 0.6, bankroll=100.0) == 0.0
    assert kelly_fraction(1.0, 0.6, bankroll=100.0) == 0.0


def test_kelly_fraction_zero_bankroll():
    assert kelly_fraction(0.4, 0.6, bankroll=0.0) == 0.0


def test_logit_distance_symmetric():
    assert logit_distance(0.2, 0.8) == pytest.approx(logit_distance(0.8, 0.2))


def test_logit_distance_zero_when_equal():
    assert logit_distance(0.5, 0.5) == pytest.approx(0.0, abs=1e-6)


def test_logit_distance_large_for_extremes():
    assert logit_distance(0.01, 0.99) > 2.0


def test_confidence_level_high():
    assert confidence_level(0.25, 0.7, 0.4) == "high"


def test_confidence_level_medium():
    assert confidence_level(0.12, 0.6, 0.5) == "medium"


def test_confidence_level_low():
    assert confidence_level(0.05, 0.52, 0.5) == "low"


def test_slippage_aware_kelly_caps_by_depth():
    base = kelly_fraction(0.4, 0.6, bankroll=100.0)
    capped = slippage_aware_kelly(0.4, 0.6, bankroll=100.0, available_depth=10.0, depth_fraction=0.25)
    assert capped <= base
    assert capped == pytest.approx(2.5)


def test_slippage_aware_kelly_no_depth_equals_base():
    base = kelly_fraction(0.4, 0.6, bankroll=100.0)
    assert slippage_aware_kelly(0.4, 0.6, bankroll=100.0, available_depth=0) == base


def test_kelly_two_sided_yes_edge():
    yes = kelly_fraction_two_sided(0.4, 0.6, bankroll=100.0)
    assert yes > 0


def test_kelly_two_sided_no_edge():
    # ai 25% vs market 35% → edge is on NO side
    no = kelly_fraction_two_sided(0.348, 0.25, bankroll=100.0)
    assert no > 0


def test_slippage_aware_kelly_sizes_no_side():
    size = slippage_aware_kelly(0.348, 0.25, bankroll=100.0, available_depth=1000.0)
    assert size > 0
