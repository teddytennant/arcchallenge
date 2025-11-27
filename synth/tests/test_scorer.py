import pytest
from synth.scorer import score_program


def test_score_program_returns_float():
    """Test that scorer returns a float."""
    score = score_program(None, None)

    assert isinstance(score, float)
    assert 0.0 <= score <= 1.0


def test_score_program_multiple_calls():
    """Test that scorer can be called multiple times."""
    scores = [score_program(None, None) for _ in range(10)]

    assert len(scores) == 10
    assert all(isinstance(s, float) for s in scores)
