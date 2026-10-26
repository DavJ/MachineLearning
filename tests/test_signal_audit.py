"""Audit calibration, planted alternatives and multiple-comparison contracts."""

import itertools
import json
import subprocess
import sys

import numpy as np
import pytest

from sportka.experiments.signal_audit import (
    LAGS, PAIR_I, PAIR_J, P_NUMBER, P_PAIR, audit_rows, calibrate_uniform,
    mc_pvalue, permutation_maxima, run, sample_planted, sample_uniform,
    sensitivity, summarize_family, temporal_components, temporal_context,
    uniform_components, validate_rows, wilson_interval,
)


def test_uniform_draws_have_six_distinct_main_numbers_and_correct_marginals():
    rows = sample_uniform(40000, np.random.default_rng(8))
    assert rows.shape == (40000, 49)
    assert np.all(rows.sum(axis=1) == 6)
    assert np.max(np.abs(rows.mean(axis=0) - 6 / 49)) < .007
    assert abs((rows[:, 0] * rows[:, 1]).mean() - 30 / (49 * 48)) < .002


@pytest.mark.parametrize("kind", ["number", "pair", "temporal"])
def test_planted_controls_remain_valid_lottery_draws(kind):
    rows = sample_planted(500, kind, .2, np.random.default_rng(17))
    validate_rows(rows)
    assert np.all(rows.sum(axis=1) == 6)


def test_pair_control_has_joint_signal_without_marginal_bias():
    rows = sample_planted(80000, "pair", 1.0, np.random.default_rng(13))
    # Under this control, balls 1/2 occur together or are both absent. Every
    # ball's analytical marginal remains 6/49, including the other 47 balls.
    np.testing.assert_array_equal(rows[:, 0], rows[:, 1])
    assert np.max(np.abs(rows.mean(axis=0) - P_NUMBER)) < .005
    assert (rows[:, 0] * rows[:, 1]).mean() > .11
    assert P_PAIR < .013


def test_pair_counts_match_independent_explicit_enumeration():
    rows = sample_uniform(30, np.random.default_rng(1))
    components = uniform_components(rows)
    brute = {pair: 0 for pair in itertools.combinations(range(49), 2)}
    for row in rows:
        for pair in itertools.combinations(np.flatnonzero(row), 2):
            brute[pair] += 1
    expected = [brute[(int(i), int(j))] for i, j in zip(PAIR_I, PAIR_J)]
    np.testing.assert_array_equal(components["pairs"][0], expected)
    assert components["pairs"][0].sum() == 30 * 15
    assert len(expected) == 1176
    np.testing.assert_allclose(components["pairs"][1], 30 * P_PAIR)


def test_temporal_counts_and_permutation_center_by_hand():
    rows = np.zeros((4, 49), dtype=np.uint8)
    rows[0, :6] = 1
    rows[1, :3] = rows[1, 6:9] = 1
    rows[2, 3:9] = 1
    rows[3, :6] = 1
    lags = (1, 2, 3)
    observed, expected, scores = temporal_components(rows, lags)
    assert list(observed[[49, 99, 149]]) == [9, 6, 6]
    # Conditional permutation expectation equals the average over every order.
    all_counts = [temporal_components(rows[list(order)], lags)[0]
                  for order in itertools.permutations(range(4))]
    np.testing.assert_allclose(np.mean(all_counts, axis=0), expected)
    assert len(scores) == 150


def test_constant_draws_have_no_detectable_ordering_information():
    rows = np.zeros((50, 49), dtype=np.uint8)
    rows[:, :6] = 1
    score = temporal_components(rows)[2].max()
    null = permutation_maxima(rows, 99, seed=2)
    assert score == 0
    assert np.all(null == 0)
    assert mc_pvalue(score, null) == 1


def test_monte_carlo_pvalues_include_ties_and_cannot_be_zero():
    assert mc_pvalue(10, np.arange(9)) == .1
    assert mc_pvalue(0, np.arange(9)) == 1
    assert mc_pvalue(1, np.array([1., 1., 1.])) == 1
    assert mc_pvalue(1 + np.finfo(float).eps, np.array([1., 1.])) == 1
    with pytest.raises(ValueError):
        mc_pvalue(1, [])


def test_maximum_statistic_and_across_family_correction_are_both_applied():
    components = (np.array([9, 10]), np.array([5., 5.]), np.array([4., 5.]))
    maxima = np.array([0., 1., 2., 3., 4., 5., 6., 7., 8.])
    report = summarize_family(components, maxima, [1, 2], family_count=3, alpha=.05)
    assert report["maxstat_pvalue"] == .5  # four null maxima >= 5, plus observed
    assert report["overall_adjusted_pvalue"] == 1
    assert not report["rejected_after_correction"]
    assert report["top_features"][0]["feature"] == 2


def test_cache_round_trip_uses_safe_arrays_and_complete_identity(tmp_path):
    first, meta = calibrate_uniform(60, 19, seed=1, cache_dir=tmp_path)
    second, cached = calibrate_uniform(60, 19, seed=1, cache_dir=tmp_path)
    assert not meta["cache_reused"]
    assert cached["cache_reused"]
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])
    _, different = calibrate_uniform(60, 19, seed=2, cache_dir=tmp_path)
    assert different["identity"] != meta["identity"]
    for file in tmp_path.glob("*.npz"):
        with np.load(file, allow_pickle=False) as saved:
            assert json.loads(saved["metadata"].item())["n"] == 60


@pytest.fixture(scope="module")
def calibration():
    return calibrate_uniform(800, resamples=199, seed=30)[0]


def test_detectors_find_strong_known_alternatives(calibration):
    for kind, family, strength in (("number", "numbers", .25), ("pair", "pairs", .5),
                                   ("temporal", "ordering", .25)):
        rows = sample_planted(800, kind, strength, np.random.default_rng(55))
        result = audit_rows(rows, calibration, resamples=199, seed=3, family_count=6)
        assert result[family]["rejected_after_correction"], (kind, result[family])


def test_sensitivity_checks_controls_and_reports_uncertainty():
    references, _ = calibrate_uniform(100, 199, seed=2)
    report = sensitivity(100, references, trials=2, permutations=199, seed=7)
    assert len(report["levels"]) == 12
    assert [row["mixture_strength"] for row in report["levels"]].count(0) == 3
    for row in report["levels"]:
        lo, hi = row["conditional_wilson_95_interval"]
        assert 0 <= lo <= row["detection_rate"] <= hi <= 1
    assert wilson_interval(0, 20)[0] == 0
    assert wilson_interval(20, 20)[1] == 1


def test_real_csv_run_is_source_labelled_and_never_claims_prediction_edge(tmp_path):
    from tests.test_sportka_data import HEADER, ROW
    from datetime import date, timedelta
    path = tmp_path / "draws.csv"
    contents = HEADER
    for i in range(25):
        day = date(2025, 1, 1) + timedelta(days=i)
        contents += ROW.replace("1. 1. 2025", day.strftime("%d.%m.%Y"))
    path.write_text(contents)
    report = run(path, resamples=199, power=False, cache_dir=tmp_path / "calibration")
    assert report["families_in_correction"] == 6
    assert report["games"]["1"]["draw_count"] == 25
    assert report["games"]["1"]["families"]["numbers"]["rejected_after_correction"]
    assert not report["games"]["1"]["families"]["ordering"]["rejected_after_correction"]
    assert report["prediction_edge"] == "NOT_ESTABLISHED"
    assert len(report["source_sha256"]) == 64
    assert report["sensitivity"] == {}


@pytest.mark.parametrize("lags", [(), (0,), (-1,), (1, 1), (50,)])
def test_invalid_lags_rejected(lags):
    with pytest.raises(ValueError):
        temporal_context(sample_uniform(50, np.random.default_rng(2)), lags)


def test_cli_errors_do_not_fall_back_to_synthetic_data():
    result = subprocess.run([sys.executable, "-m", "sportka.experiments.signal_audit",
                             "--csv", "/no/such/file.csv", "--no-power"],
                            text=True, capture_output=True, timeout=20)
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert not result.stdout


@pytest.mark.parametrize("options", [dict(resamples=19, power=False),
                                    dict(resamples=199, power_permutations=19)])
def test_insufficient_corrected_monte_carlo_resolution_is_an_error(options):
    # Fail before even trying the source: an impossible rejection threshold
    # must not be reported as a negative statistical finding.
    with pytest.raises(ValueError, match="More .* needed"):
        run("/not/read/because/resolution/is/invalid.csv", **options)
