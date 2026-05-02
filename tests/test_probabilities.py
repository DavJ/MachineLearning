"""
Tests that probability vectors produced by models and feature functions
correctly represent P(number appears in a draw) and therefore sum to ≈ 7.
"""

import numpy as np
import pytest


def _make_random_draws(n: int = 200, seed: int = 0) -> np.ndarray:
    """Return (n, 49) binary matrix with exactly 7 ones per row."""
    rng = np.random.default_rng(seed)
    Y = np.zeros((n, 49), dtype=np.float32)
    for i in range(n):
        chosen = rng.choice(49, size=7, replace=False)
        Y[i, chosen] = 1.0
    return Y


# ---------------------------------------------------------------------------
# Probability sum ≈ 7
# ---------------------------------------------------------------------------

def test_random_predictor_sum():
    """RandomPredictor outputs 7/49 per number → sum = 7."""
    from sportka.models import RandomPredictor

    model = RandomPredictor()
    model.fit(np.zeros((1, 1)), np.zeros((1, 49)))
    pred = model.predict_proba(np.zeros((5, 1)))

    row_sums = pred.sum(axis=1)
    assert np.allclose(row_sums, 7.0), f"Expected 7.0, got {row_sums}"


def test_global_freq_predictor_sum():
    """GlobalFreqPredictor uses appearance_count/n_draws per number; sum over 49 numbers ≈ 7."""
    from sportka.models import GlobalFreqPredictor

    Y = _make_random_draws(200)
    model = GlobalFreqPredictor()
    model.fit(np.zeros((200, 1)), Y)
    pred = model.predict_proba(np.zeros((10, 1)))

    row_sums = pred.sum(axis=1)
    assert np.all(row_sums > 6.5), f"Row sums too low: {row_sums}"
    assert np.all(row_sums < 7.5), f"Row sums too high: {row_sums}"


def test_assert_probability_scale_valid():
    """assert_probability_scale accepts a valid ~7-sum vector."""
    from sportka.evaluation import assert_probability_scale

    p = np.ones(49) * (7.0 / 49.0)
    assert_probability_scale(p)  # must not raise


def test_assert_probability_scale_rejects_softmax():
    """assert_probability_scale rejects a softmax-normalised vector (sum=1)."""
    from sportka.evaluation import assert_probability_scale

    p = np.ones(49) / 49.0  # sum == 1
    with pytest.raises(AssertionError):
        assert_probability_scale(p)


def test_rolling_freq_sum():
    """_rolling_freq returns per-draw probabilities that sum to ≈ 7."""
    from sportka.features import _rolling_freq

    rng = np.random.default_rng(1)
    numbers_list = [
        list(map(int, rng.choice(49, size=7, replace=False) + 1))
        for _ in range(100)
    ]
    freq = _rolling_freq(numbers_list, window=10)
    # Skip first rows where window is smaller; check last 80 rows
    row_sums = freq[20:].sum(axis=1)
    assert np.all(row_sums > 6.0), f"_rolling_freq sum too low: {row_sums.min()}"
    assert np.all(row_sums < 8.0), f"_rolling_freq sum too high: {row_sums.max()}"


def test_exp_decay_freq_sum():
    """_exp_decay_freq returns per-draw probabilities that sum to ≈ 7."""
    from sportka.features import _exp_decay_freq

    rng = np.random.default_rng(2)
    numbers_list = [
        list(map(int, rng.choice(49, size=7, replace=False) + 1))
        for _ in range(50)
    ]
    freq = _exp_decay_freq(numbers_list)
    row_sums = freq.sum(axis=1)
    assert np.all(row_sums > 6.0), f"_exp_decay_freq sum too low: {row_sums.min()}"
    assert np.all(row_sums < 8.0), f"_exp_decay_freq sum too high: {row_sums.max()}"
