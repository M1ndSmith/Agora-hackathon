from agent.tools.domain import classify_domain, get_estimation_prompt


def test_politics():
    assert classify_domain("Will Trump win the 2026 election?") == "politics"


def test_sports():
    assert classify_domain("Will the Lakers win the NBA championship?") == "sports"


def test_crypto():
    assert classify_domain("Will Bitcoin reach $100k by June?") == "crypto"


def test_science():
    assert classify_domain("Will the FDA approve the new vaccine trial?") == "science"


def test_general_ambiguous():
    assert classify_domain("Will it rain tomorrow in Paris?") == "general"


def test_estimation_prompt_per_domain():
    p = get_estimation_prompt("politics")
    assert "politics" in p.lower()
