from agent.tools.clob import parse_orderbook


def test_parse_orderbook_happy():
    raw = {
        "bids": [{"price": "0.55", "size": "100"}, {"price": "0.54", "size": "50"}],
        "asks": [{"price": "0.57", "size": "80"}, {"price": "0.58", "size": "40"}],
    }
    sig = parse_orderbook(raw)
    assert sig is not None
    assert sig.best_bid == 0.55
    assert sig.best_ask == 0.57
    assert sig.spread == 0.02
    assert sig.depth_usd > 0


def test_parse_orderbook_empty():
    assert parse_orderbook({}) is None
    assert parse_orderbook({"bids": [], "asks": []}) is None


def test_parse_orderbook_malformed():
    assert parse_orderbook(None) is None
    assert parse_orderbook({"bids": "bad"}) is None
