import pytest

from agent.tools.polymarket import _is_binary, _outcome_from_prices, _parse_binary_prob


def test_parse_binary_prob_valid():
    assert _parse_binary_prob('["0.55", "0.45"]') == pytest.approx(0.55)


def test_parse_binary_prob_invalid_json():
    assert _parse_binary_prob("not json") is None


def test_parse_binary_prob_non_binary():
    assert _parse_binary_prob('["0.3", "0.3", "0.4"]') is None


def test_parse_binary_prob_non_numeric():
    assert _parse_binary_prob('["yes", "no"]') is None


def test_is_binary_two_outcomes():
    assert _is_binary('["Yes", "No"]') is True


def test_is_binary_three_outcomes():
    assert _is_binary('["A", "B", "C"]') is False


def test_is_binary_malformed():
    assert _is_binary("broken") is False


def test_outcome_from_prices_yes_wins():
    assert _outcome_from_prices(1.0, 0.0) == "yes"


def test_outcome_from_prices_no_wins():
    assert _outcome_from_prices(0.0, 1.0) == "no"
