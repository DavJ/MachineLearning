"""Independent enumeration and integration checks for complete-set scores."""

from itertools import combinations
from math import comb, log

import numpy as np
import pytest

from sportka.fixed_cardinality import FixedCardinality, conditional_set_metrics


def outcomes(pool, count):
    sets = list(combinations(range(pool), count))
    y = np.zeros((len(sets), pool))
    for row, selected in zip(y, sets):
        row[list(selected)] = 1
    return y


@pytest.mark.parametrize("count", [0, 1, 3, 7, 8])
@pytest.mark.parametrize("kind", ["uneven", "boundary", "equal"])
def test_complete_law_and_marginals_match_enumerated_bernoulli_products(count, kind):
    y = outcomes(8, count)
    p = {"uneven": np.array([.01, .07, .12, .21, .4, .6, .9, .99]),
         "boundary": np.array([0., 1., 0., 1., .001, .1, .8, .999]),
         "equal": np.full(8, .8)}[kind]
    # Direct independent-Bernoulli product enumeration, without odds or DP.
    clipped = np.clip(p, 1e-9, 1 - 1e-9)
    raw_mass = np.prod(np.where(y, clipped, 1 - clipped), axis=1)
    expected = raw_mass / raw_mass.sum()
    law = FixedCardinality(np.tile(p, (len(y), 1)), count)
    actual = np.exp(law.log_prob(y))
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=2e-14)
    assert actual.sum() == pytest.approx(1., abs=1e-12)
    expected_marginals = expected @ y
    np.testing.assert_allclose(law.marginals(), np.tile(expected_marginals, (len(y), 1)), atol=2e-13)
    np.testing.assert_allclose(law.marginals().sum(axis=1), count, atol=1e-12)
    selected = np.zeros(8)
    if count:
        selected[np.argsort(p)[-count:]] = 1
    assert float(law.log_prob(np.tile(selected, (len(y), 1)))[0]) == pytest.approx(np.log(expected.max()), abs=2e-12)


@pytest.mark.parametrize("score", [0., .01, 6 / 49, .9, 1.])
def test_any_equal_scores_give_exact_uniform_six_set_law(score):
    y = np.zeros((2, 49))
    y[0, :6] = 1
    y[1, -6:] = 1
    law = FixedCardinality(np.full_like(y, score))
    np.testing.assert_allclose(law.log_prob(y), -log(comb(49, 6)), atol=2e-13)
    np.testing.assert_allclose(law.marginals(), 6 / 49, atol=1e-14)
    metrics = conditional_set_metrics(y, np.full_like(y, score))
    assert metrics["conditional_set_gain_vs_uniform"] == pytest.approx(0., abs=2e-13)
    assert metrics["conditional_brier"] == pytest.approx((6 / 49) * (43 / 49))


def test_common_odds_scale_cancels_but_multiplying_probabilities_does_not():
    p = np.array([[.02, .08, .13, .19, .37, .55, .61, .85]])
    odds = p / (1 - p)
    shifted = 1000 * odds / (1 + 1000 * odds)
    y = outcomes(8, 3)
    a = FixedCardinality(np.repeat(p, len(y), axis=0), 3)
    b = FixedCardinality(np.repeat(shifted, len(y), axis=0), 3)
    np.testing.assert_allclose(a.log_prob(y), b.log_prob(y), atol=1e-11)
    np.testing.assert_allclose(a.marginals(), b.marginals(), atol=1e-12)
    c = FixedCardinality(p / 2, 3)
    assert not np.allclose(a.marginals()[0], c.marginals()[0])


def test_batch_row_and_ball_permutations_and_input_mutation():
    rng = np.random.default_rng(8)
    p = rng.uniform(.01, .99, size=(5, 49))
    y = np.zeros_like(p)
    for row in y:
        row[rng.choice(49, 6, replace=False)] = 1
    original = p.copy()
    law = FixedCardinality(p)
    marginals, log_probs = law.marginals(), law.log_prob(y)
    np.testing.assert_array_equal(p, original)
    p[:] = .9
    np.testing.assert_array_equal(law.marginals(), marginals)
    np.testing.assert_array_equal(law.log_prob(y), log_probs)
    rows, balls = rng.permutation(5), rng.permutation(49)
    other = FixedCardinality(original[rows][:, balls])
    np.testing.assert_allclose(other.marginals(), marginals[rows][:, balls], atol=2e-13)
    np.testing.assert_allclose(other.log_prob(y[rows][:, balls]), log_probs[rows], atol=2e-13)
    for i in range(len(y)):
        single = FixedCardinality(original[i:i + 1])
        np.testing.assert_array_equal(single.log_prob(y[i:i + 1]), log_probs[i:i + 1])


def test_float32_endpoints_have_positive_support_for_impossible_raw_targets():
    p = np.zeros((2, 49), dtype=np.float32)
    p[:, :6] = 1
    y = p.copy()
    y[1] = np.roll(y[1], 10)
    law = FixedCardinality(p)
    scores = law.log_prob(y)
    assert np.isfinite(scores).all()
    assert scores[0] == pytest.approx(0., abs=1e-10)
    assert scores[1] < -200
    assert np.isfinite(law.marginals()).all()
    assert all(np.isfinite(v) for v in conditional_set_metrics(y, p).values())


def test_single_favored_ball_matches_closed_form_49_ball_marginal():
    # All other weights are one; only the membership of this ball matters.
    p = np.full((2, 49), .5)
    p[:, 17] = .8
    y = np.zeros_like(p)
    y[:, :6] = 1
    y[1, 0], y[1, 17] = 0, 1
    z = comb(48, 6) + 4 * comb(48, 5)
    law = FixedCardinality(p)
    np.testing.assert_allclose(law.log_prob(y), [-log(z), log(4) - log(z)], atol=3e-13)
    np.testing.assert_allclose(law.marginals()[:, 17], 4 * comb(48, 5) / z, atol=1e-14)


@pytest.mark.parametrize("scores", [[], np.ones(49), np.zeros((0, 49)), np.zeros((2, 0)),
                                   [[np.nan]], [[np.inf]], [[-.1]], [[1.1]]])
def test_invalid_scores_are_rejected(scores):
    with pytest.raises(ValueError):
        FixedCardinality(scores)


@pytest.mark.parametrize("count", [True, 2.5, -1, 9])
def test_invalid_counts_are_rejected(count):
    with pytest.raises(ValueError, match="count"):
        FixedCardinality(np.full((1, 8), .2), count)


@pytest.mark.parametrize("target", [np.ones((1, 7)), np.ones((1, 8)),
                                    [[1, 1, .5, .5, 0, 0, 0, 0]],
                                    [[1, 1, np.nan, 0, 0, 0, 0, 0]]])
def test_invalid_targets_are_rejected(target):
    with pytest.raises(ValueError):
        FixedCardinality(np.full((1, 8), .2), 3).log_prob(target)


def test_common_evaluator_preserves_raw_scores_and_adds_conditioned_scores():
    from sportka.evaluation import compute_all_metrics

    y = np.zeros((1, 49))
    y[:, :6] = 1
    p = np.full_like(y, .9)
    metrics = compute_all_metrics(y, p)
    expected_raw_bce = -(6 * log(.9) + 43 * log(.1)) / 49
    assert metrics["bce"] == pytest.approx(expected_raw_bce)
    assert metrics["conditional_set_nll"] == pytest.approx(log(comb(49, 6)))
    assert metrics["conditional_bce"] < metrics["bce"]
    assert metrics["conditional_brier"] < metrics["brier"]
