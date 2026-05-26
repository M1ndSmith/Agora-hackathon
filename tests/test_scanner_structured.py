import pytest
from pydantic import ValidationError

from models import ScannerCandidate, ScannerCandidates


def test_scanner_candidate_valid():
    c = ScannerCandidate(
        market_id="1",
        question="Will X?",
        market_prob=0.4,
        ai_prob=0.55,
    )
    assert c.market_prob == 0.4


def test_scanner_candidates_list():
    sc = ScannerCandidates(
        candidates=[
            ScannerCandidate(
                market_id="1",
                question="Q",
                market_prob=0.5,
                ai_prob=0.6,
            )
        ]
    )
    assert len(sc.candidates) == 1


def test_scanner_candidate_rejects_invalid_prob():
    with pytest.raises(ValidationError):
        ScannerCandidate(
            market_id="1",
            question="Q",
            market_prob=1.5,
            ai_prob=0.5,
        )
