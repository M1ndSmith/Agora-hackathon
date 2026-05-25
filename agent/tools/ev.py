"""
Expected Value and Kelly Criterion calculations.
Pure math — no LLM or IO calls.
"""
import math


def logit_distance(p1: float, p2: float) -> float:
    """
    Symmetric log-odds distance between two probabilities.

    Used as a sanity check: tiny absolute differences in the [0, 1] space
    can mean enormous differences in odds (0.001 vs 0.05 is 50x odds).
    Logit space makes this meaningful and symmetric around 0.5.

    Returns a non-negative float. A distance of 2.5 corresponds to roughly
    a 12x odds ratio.
    """
    def _logit(p: float) -> float:
        clamped = max(0.001, min(0.999, float(p)))
        return math.log(clamped / (1.0 - clamped))

    return abs(_logit(p1) - _logit(p2))


def calculate_ev(market_prob: float, ai_prob: float) -> float:
    """
    Calculate the edge (expected value) of a bet.

    EV = (ai_prob - market_prob) / market_prob

    Positive EV means the agent thinks the event is more likely
    than the market implies — potential value on the YES side.
    Negative EV means value on the NO side.
    """
    if market_prob <= 0:
        return 0.0
    return (ai_prob - market_prob) / market_prob


def kelly_fraction(
    market_prob: float,
    ai_prob: float,
    bankroll: float,
    fraction: float = 0.25,
) -> float:
    """
    Quarter-Kelly criterion position sizing.

    b = decimal odds - 1 = (1/market_prob) - 1
    p = ai estimated probability of YES
    q = 1 - p
    kelly = (b*p - q) / b

    Returns fraction of bankroll to bet (capped at 0, conservative quarter-Kelly).
    """
    if market_prob <= 0 or market_prob >= 1:
        return 0.0
    if bankroll <= 0:
        return 0.0

    decimal_odds = 1.0 / market_prob
    b = decimal_odds - 1.0
    p = ai_prob
    q = 1.0 - p

    if b <= 0:
        return 0.0

    kelly = (b * p - q) / b
    return max(0.0, kelly * bankroll * fraction)


def confidence_level(ev: float, ai_prob: float, market_prob: float) -> str:
    """
    Classify confidence as low / medium / high based on edge size and probability distance.
    """
    abs_ev = abs(ev)
    prob_diff = abs(ai_prob - market_prob)

    if abs_ev >= 0.20 and prob_diff >= 0.15:
        return "high"
    elif abs_ev >= 0.10 and prob_diff >= 0.07:
        return "medium"
    else:
        return "low"
