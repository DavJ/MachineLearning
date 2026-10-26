"""Causal selection, coherent mixtures, independent null arithmetic and caches."""
from itertools import combinations
from math import comb, exp, fsum, log

import numpy as np
import pytest

from sportka.experiments import neural_recovery as m
from sportka.experiments.signal_audit import sample_uniform
from sportka.fixed_cardinality import FixedCardinality


@pytest.mark.parametrize("weight", [0., .01, .3, 1.])
def test_mixture_likelihood_has_unit_uniform_expectation_even_after_past_selection(weight):
    y = np.zeros((comb(8, 3), 8))
    for row, subset in zip(y, combinations(range(8), 3)):
        row[list(subset)] = 1
    scores = np.tile([.02, .05, .1, .2, .3, .6, .8, .9], (len(y), 1))
    ratio = FixedCardinality(scores, 3).log_prob(y) + log(comb(8, 3))
    # Every possible first outcome may choose a different second-step weight.
    first = np.exp(m.blend_gain(ratio, weight))
    second = [np.exp(m.blend_gain(ratio, .3 if row[0] else .01)).mean() for row in y]
    assert first.mean() == pytest.approx(1., abs=1e-13)
    assert np.mean(first * second) == pytest.approx(1., abs=1e-13)


@pytest.mark.parametrize("bad", [-1, 1.1, np.nan, np.inf])
def test_invalid_mixture_weight(bad):
    with pytest.raises(ValueError):
        m.blend_gain(np.zeros(3), bad)


@pytest.mark.parametrize("bad", [[], [np.nan], [[1, 2]]])
def test_invalid_gains(bad):
    with pytest.raises(ValueError):
        m.evidence(bad)
    with pytest.raises(ValueError):
        m.blend_gain(bad, .3)


def test_evidence_threshold_direction_and_finite_log_extremes():
    assert not m.evidence([log(39)])['terminal_crossing']
    assert m.evidence([log(40)])['terminal_crossing']
    assert not m.evidence([-1000])['terminal_crossing']
    assert m.evidence([1000])['terminal_e'] is None
    assert m.evidence([-1000])['terminal_e'] is None
    assert m.evidence([0])['terminal_e'] == 1


class FixedScores:
    n_iter_ = 1

    def predict_proba(self, x):
        return np.full((len(x), 49), 6 / 49)


def test_fit_and_validation_choice_cannot_see_changed_test_targets(tmp_path, monkeypatch):
    rows = sample_uniform(300, np.random.default_rng(5))
    observed = []
    def trainer(x, y, **kwargs):
        observed.append((x.copy(), y.copy()))
        return FixedScores(), False
    monkeypatch.setattr(m, 'train_or_load', trainer)
    first = m.pipeline(rows, tmp_path)
    changed = rows.copy()
    start = first['test_source_start']
    changed[start:] = np.roll(changed[start:], 1, axis=1)
    second = m.pipeline(changed, tmp_path)
    assert first['selected'] == second['selected'] == {'kind': 'uniform', 'weight': 0., 'validation_mean_joint_gain': 0.}
    assert sum(first['split_samples'].values()) == len(rows) - 52
    for i in range(4):
        np.testing.assert_array_equal(observed[i][0], observed[i + 4][0])
        np.testing.assert_array_equal(observed[i][1], observed[i + 4][1])
        assert len(observed[i][1]) == first['split_samples']['training']
    assert first['test']['terminal_e'] == 1


def test_real_checkpoint_roundtrip_reuses_all_four_without_refitting(tmp_path, monkeypatch):
    rows = sample_uniform(200, np.random.default_rng(7))
    first = m.pipeline(rows, tmp_path)
    from sklearn.neural_network import MLPClassifier
    def forbidden(*args, **kwargs):
        raise AssertionError('Identical data must load saved weights')
    monkeypatch.setattr(MLPClassifier, 'fit', forbidden)
    second = m.pipeline(rows, tmp_path)
    assert all(x['cache_reused'] for x in second['fit_diagnostics'].values())
    assert not any(x['cache_reused'] for x in first['fit_diagnostics'].values())
    assert first['selected'] == second['selected']
    assert first['test'] == second['test']


@pytest.mark.parametrize("kind", ['number', 'pair', 'temporal'])
def test_oracle_has_unit_null_expectation_by_independent_category_arithmetic(kind):
    rows = np.zeros((4, 49), dtype=np.uint8)
    rows[0, [0, 1, 2, 3, 4, 5]] = 1
    rows[1] = rows[0]                         # contains both / a complete copy
    rows[2, [0, 2, 3, 4, 5, 6]] = 1         # contains ball 1 only
    rows[3, [2, 3, 4, 5, 6, 7]] = 1         # neither; noncopy
    ratios = np.exp(m.oracle_gain(rows, kind, .25, 1))
    if kind == 'number':
        probs = [6 / 49, 0, 43 / 49]
    elif kind == 'pair':
        both = comb(47, 4) / comb(49, 6)
        neither = comb(47, 6) / comb(49, 6)
        probs = [both, 1 - both - neither, neither]
    else:
        probs = [1 / comb(49, 6), 1 - 1 / comb(49, 6), 0]
    assert fsum(p * r for p, r in zip(probs, ratios)) == pytest.approx(1., abs=1e-14)


def test_sensitivity_records_every_independent_trial_and_wilson_bounds(tmp_path, monkeypatch):
    tasks = []
    def control(task):
        tasks.append(task)
        i, j, *_ = task
        return i, j, {'selected': {'kind': 'uniform'}, 'test': m.evidence([0]), 'oracle': m.evidence([0])}
    monkeypatch.setattr(m, '_control', control)
    result = m.sensitivity(200, tmp_path, trials=2, workers=1)
    assert len(tasks) == len(m.LEVELS) * 2
    assert len({(t[0], t[1]) for t in tasks}) == len(tasks)
    assert all(x['crossings'] == 0 and len(x['trials_detail']) == 2 for x in result['levels'])
    assert all(0 <= x['wilson95'][0] <= x['wilson95'][1] <= 1 for x in result['levels'])


@pytest.mark.parametrize("options", [{'trials': 0}, {'workers': 5}, {'seed': -1}])
def test_invalid_control_configuration_fails_before_fitting(tmp_path, options):
    with pytest.raises(ValueError):
        m.sensitivity(200, tmp_path, **options)


def test_missing_source_never_becomes_a_synthetic_result(tmp_path):
    with pytest.raises(FileNotFoundError):
        m.run(tmp_path / 'missing.csv', tmp_path, power=False)
    with pytest.raises(ValueError, match='200'):
        m.pipeline(sample_uniform(199, np.random.default_rng(1)), tmp_path)
