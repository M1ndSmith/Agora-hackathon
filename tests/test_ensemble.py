from agent.tools.ensemble import aggregate_estimates
from models import ResearchEstimate


def _est(prob: float) -> ResearchEstimate:
    return ResearchEstimate(
        ai_prob=prob,
        confidence="medium",
        reasoning="test",
        key_evidence=["a"],
        bull_case="b",
        bear_case="c",
    )


def test_aggregate_median():
    estimates = [
        ("groq", _est(0.6)),
        ("nvidia", _est(0.7)),
        ("openai", _est(0.65)),
    ]
    agg = aggregate_estimates(estimates, disagreement_threshold=0.2)
    assert agg.ai_prob == 0.65
    assert len(agg.providers) == 3


def test_aggregate_downgrade_on_disagreement():
    estimates = [
        ("groq", _est(0.3)),
        ("openai", _est(0.8)),
    ]
    agg = aggregate_estimates(estimates, disagreement_threshold=0.15)
    assert agg.downgrade_confidence is True
    assert agg.confidence == "low"
    assert agg.spread == 0.5


def test_aggregate_single_provider():
    agg = aggregate_estimates([("groq", _est(0.55))], disagreement_threshold=0.15)
    assert agg.ai_prob == 0.55
    assert agg.spread == 0.0
    assert agg.downgrade_confidence is False
