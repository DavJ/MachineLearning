"""
Tests for probability scaling correctness.

Each number's probability must represent P(number appears in a draw).
The sum over all 49 numbers must therefore be approximately 7
(since exactly 7 numbers are drawn each time).
"""

import numpy as np
import pytest


def test_probability_sum_uniform():
    """Uniform probability 7/49 per number sums to 7."""
    p = np.ones(49) * (7 / 49)
    assert 6.5 < p.sum() < 7.5


def test_assert_probability_scale_passes():
    """assert_probability_scale should not raise for a valid vector."""
    from sportka.evaluation import assert_probability_scale

    p = np.ones(49) * (7 / 49)
    assert_probability_scale(p)  # must not raise


def test_assert_probability_scale_fails_on_softmax():
    """assert_probability_scale should raise for a softmax-normalised vector (sum=1)."""
    from sportka.evaluation import assert_probability_scale

    p = np.ones(49) / 49  # sum == 1
    with pytest.raises(AssertionError):
        assert_probability_scale(p)


def test_global_freq_predictor_sum():
    """GlobalFreqPredictor predictions must sum to ~7 over the 49 numbers."""
    from sportka.models import GlobalFreqPredictor

    rng = np.random.default_rng(0)
    # Simulate 200 draws: binary matrix (n, 49), exactly 7 ones per row
    Y = np.zeros((200, 49), dtype=np.float32)
    for i in range(200):
        chosen = rng.choice(49, size=7, replace=False)
        Y[i, chosen] = 1.0
    X_dummy = np.zeros((200, 1), dtype=np.float32)

    model = GlobalFreqPredictor()
    model.fit(X_dummy, Y)
    pred = model.predict_proba(np.zeros((10, 1), dtype=np.float32))

    row_sums = pred.sum(axis=1)
    assert np.all(row_sums > 6.5), f"Row sums too low: {row_sums}"
    assert np.all(row_sums < 7.5), f"Row sums too high: {row_sums}"
