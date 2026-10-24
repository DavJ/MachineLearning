"""Coherent set laws, training/selection separation and planted-signal recovery."""

from collections import Counter
from datetime import date, timedelta
import hashlib
from itertools import combinations
import json
import math
import subprocess
import sys

import numpy as np
import pytest

from sportka.experiments import joint_signal_recovery as j
from sportka.experiments.signal_audit import sample_planted, sample_uniform
from tests.test_sportka_data import HEADER, ROW


def tiny_lottery():
    rows = np.zeros((math.comb(8, 3), 8), dtype=np.uint8)
    for row, chosen in zip(rows, combinations(range(8), 3)):
        row[list(chosen)] = 1
    previous = np.zeros_like(rows)
    previous[:, [0, 2, 5]] = 1
    return rows, previous


@pytest.mark.parametrize("model", [j.Model("uniform"), j.Model("number", (0,), 0.),
    j.Model("number", (0,), .4), j.Model("pair", (0, 1), 0.),
    j.Model("pair", (0, 1), .4), j.Model("copy"), j.Model("copy", strength=.4)])
def test_all_joint_laws_normalize_and_imply_reported_marginals_and_map(model):
    rows, previous = tiny_lottery()
    ratios = np.exp(j.log_ratios(rows, previous, model))
    probabilities = ratios / len(rows)
    assert math.fsum(probabilities) == pytest.approx(1., abs=2e-15)
    # Independently enumerate every possible combination, including dependence.
    expected_marginals = np.array([math.fsum(probabilities * rows[:, n]) for n in range(8)])
    np.testing.assert_allclose(j.marginals(previous[:1], model)[0], expected_marginals, atol=1e-15)
    assert expected_marginals.sum() == pytest.approx(3)
    maximum = j.map_sets(previous[:1], model)
    assert maximum.sum() == 3
    assert np.exp(j.log_ratios(maximum, previous[:1], model))[0] == pytest.approx(ratios.max())


def test_joint_pair_signal_is_visible_despite_identical_marginal_probability_scores():
    rows = sample_planted(20000, "pair", .5, np.random.default_rng(1))
    true = j.metrics(rows[1:], rows[:-1], j.Model("pair", (0, 1), .5))
    uniform = j.metrics(rows[1:], rows[:-1], j.Model("uniform"))
    assert true["joint_gain_nats_per_draw"] > .07
    assert true["joint_nll"] < uniform["joint_nll"]
    assert true["marginal_bce"] == uniform["marginal_bce"]
    assert true["marginal_brier"] == uniform["marginal_brier"]


@pytest.mark.parametrize("kind,strength,expected", [("number", .25, "number"),
                                                  ("pair", .4, "pair"),
                                                  ("temporal", .04, "copy")])
def test_fitted_pipeline_recovers_strong_known_structure_on_unseen_targets(kind, strength, expected):
    rows = sample_planted(3000, kind, strength, np.random.default_rng(12))
    result = j.pipeline(rows)
    assert result["selected_model"] == expected
    assert result["selected_test_joint_gain"] > .03
    fitted = result["fitted_models"][expected]
    assert fitted["balls"] == ({"number": [1], "pair": [1, 2], "copy": []}[expected])
    assert fitted["strength"] == pytest.approx(strength, abs=.04)


def test_pair_mle_agrees_with_independent_scalar_grid_likelihood():
    rows = sample_planted(4000, "pair", .4, np.random.default_rng(51))
    target, past = rows[1:], rows[:-1]
    fitted = next(model for model in j.fit_models(target, past) if model.kind == "pair")
    assert fitted.balls == (0, 1)
    categories = Counter(int(row[0]) + int(row[1]) for row in target)
    grid = np.linspace(0, .95, 4001)
    # Direct category likelihood q, not the vectorized derivative implementation.
    fair = np.array([math.comb(47, 6), 2 * math.comb(47, 5), math.comb(47, 4)]) / math.comb(49, 6)
    alternative = np.array([43 / 49, 0, 6 / 49])
    likelihoods = [sum(categories[k] * math.log((1-s)*fair[k] + s*alternative[k]) for k in range(3))
                   for s in grid]
    assert abs(fitted.strength - grid[int(np.argmax(likelihoods))]) <= grid[1] - grid[0]


def test_test_targets_do_not_change_any_fitted_parameter_or_validation_selection(monkeypatch):
    rows = sample_planted(1200, "number", .3, np.random.default_rng(11))
    calls = []
    original = j.fit_models
    def capture(target, past):
        calls.append((target.copy(), past.copy()))
        return original(target, past)
    monkeypatch.setattr(j, "fit_models", capture)
    before = j.pipeline(rows)
    start = int(.85 * (len(rows) - 1))
    changed = rows.copy()
    changed[start + 1:] = 0
    changed[start + 1:, 43:49] = 1
    after = j.pipeline(changed)
    train_count = int(.7 * (len(rows) - 1))
    for target, past in calls:
        np.testing.assert_array_equal(target, rows[1:train_count + 1])
        np.testing.assert_array_equal(past, rows[:train_count])
    assert before["fitted_models"] == after["fitted_models"]
    assert before["validation_joint_gains"] == after["validation_joint_gains"]
    assert before["selected_model"] == after["selected_model"] == "number"
    assert before["selected_test_joint_gain"] != after["selected_test_joint_gain"]


def test_validation_targets_cannot_change_training_parameters():
    rows = sample_uniform(500, np.random.default_rng(5))
    before = j.pipeline(rows)
    train_count = before["splits"]["train"]
    changed = rows.copy()
    changed[train_count + 1:] = 0
    changed[train_count + 1:, :6] = 1
    after = j.pipeline(changed)
    assert before["fitted_models"] == after["fitted_models"]


def test_strength_cap_keeps_every_set_possible_even_for_degenerate_history():
    rows = np.zeros((300, 49), dtype=np.uint8)
    rows[:, :6] = 1
    result = j.pipeline(rows)
    assert result["selected_model"] == "copy"
    assert result["fitted_models"]["copy"]["strength"] == .95
    other = np.roll(rows, 10, axis=1)
    for model in j.fit_models(rows[1:], rows[:-1]):
        assert np.isfinite(j.log_ratios(other, rows, model)).all()
    assert 0 < result["test_scores"]["copy"]["joint_nll"] < .06
    json.dumps(result, allow_nan=False)


def test_uniform_wins_validation_ties_even_when_training_fits_differ(monkeypatch):
    rows = sample_uniform(200, np.random.default_rng(3))
    monkeypatch.setattr(j, "log_ratios", lambda target, previous, model: np.zeros(len(target)))
    assert j.pipeline(rows, details=False)["selected_model"] == "uniform"


def test_null_calibration_reexecutes_the_entire_pipeline_with_independent_histories(monkeypatch):
    calls = []
    def fake_pipeline(rows, *, details):
        assert not details and np.all(rows.sum(axis=1) == 6)
        calls.append(rows.copy())
        return {"selected_model": "pair", "selected_test_joint_gain": float(rows[-1, 0])}
    monkeypatch.setattr(j, "pipeline", fake_pipeline)
    scores, chosen = j.calibrate_null(200, resamples=5, seed=91)
    assert len(calls) == 5
    assert chosen == {"uniform": 0, "number": 0, "pair": 5, "copy": 0}
    assert not np.array_equal(calls[0], calls[1])
    second, _ = j.calibrate_null(200, resamples=5, seed=91)
    np.testing.assert_array_equal(scores, second)


def test_sensitivity_labels_oracles_and_reports_zero_marginal_gain_for_pair_controls():
    reference, _ = j.calibrate_null(200, resamples=39, seed=3)
    result = j.sensitivity(200, reference, trials=2, seed=7)
    assert len(result["levels"]) == 10
    for level in result["levels"]:
        assert sum(level["selected_model_counts"].values()) == 2
        lo, hi = level["conditional_wilson_95_interval"]
        assert 0 <= lo <= level["detection_rate"] <= hi <= 1
        if level["alternative"] == "pair":
            assert level["oracle_mean_marginal_bce_gain"] == 0


def test_source_labelled_run_calibrates_search_and_never_claims_prediction_advantage(tmp_path):
    path = tmp_path / "draws.csv"
    text = HEADER
    for i in range(180):
        day = date(2025, 1, 1) + timedelta(days=i)
        text += ROW.replace("1. 1. 2025", day.strftime("%d.%m.%Y"))
    path.write_text(text)
    result = j.run(path, resamples=39, power=False)
    assert result["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["prediction_edge"] == "NOT_ESTABLISHED"
    assert len(result["null_calibrations"]["180"]["selected_test_gains"]) == 39
    for game in result["games"].values():
        assert game["selected_model"] == "copy"
        assert game["historical_rejection"]
        assert game["across_games_adjusted_pvalue"] == .05
        assert sum(game["splits"].values()) == 179
    assert result["sensitivity"] == {}


def test_inadequate_monte_carlo_resolution_is_an_error():
    with pytest.raises(ValueError, match="More resamples"):
        j.run("/not/read.csv", resamples=19, power=False)


def test_invalid_previous_draw_cannot_define_a_copy_distribution():
    rows, previous = tiny_lottery()
    previous[:, 7] = 1  # Four balls instead of the three in the target geometry.
    with pytest.raises(ValueError, match="dimensions"):
        j.log_ratios(rows, previous, j.Model("copy", strength=.5))


@pytest.mark.parametrize("model", [("bad", (), .1), ("pair", (0, 0), .1),
                                    ("number", (49,), .1), ("copy", (), 1.),
                                    ("uniform", (), .1), ("copy", (), float("nan"))])
def test_invalid_joint_models_fail(model):
    with pytest.raises(ValueError):
        j.Model(*model)


def test_cli_missing_source_never_falls_back_to_synthetic_history():
    result = subprocess.run([sys.executable, "-m", "sportka.experiments.joint_signal_recovery",
                             "--csv", "/no/such/file.csv", "--no-power"],
                            text=True, capture_output=True, timeout=20)
    assert result.returncode != 0
    assert "error:" in result.stderr and not result.stdout
