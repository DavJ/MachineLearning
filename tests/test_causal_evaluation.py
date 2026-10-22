"""Exercise actual production feature/evaluation paths, including split edges."""

import numpy as np
import pytest

from sportka.evaluation import _align_xy, assert_probability_scale, compute_all_metrics, temporal_feature_split
from sportka.features import build_features, generate_synthetic_data, winding_history_features
from sportka.models import LogisticRegressionModel, MLPModel, RollingFreqPredictor


def test_all_features_invariant_to_appending_future_draws():
    df = generate_synthetic_data(25, seed=9)
    groups = ["base", "time", "winding", "torus", "theta"]
    short = build_features(df.iloc[:10], groups, max_draw_index=7)
    full = build_features(df, groups, max_draw_index=7)
    np.testing.assert_allclose(short, full[:10], atol=1e-6)
    short_x, _ = _align_xy(df.iloc[:10], groups)
    full_x, _ = _align_xy(df, groups)
    np.testing.assert_allclose(short_x, full_x[:9], atol=1e-6)


def test_train_features_cannot_see_future_targets():
    a = generate_synthetic_data(40, seed=1)
    b = a.copy(deep=True)
    for i in range(28, len(b)):
        b.at[i, "numbers"] = [1, 2, 3, 4, 5, 6]
    groups = ["base", "time", "winding", "torus", "theta"]
    left, right = temporal_feature_split(a, groups), temporal_feature_split(b, groups)
    np.testing.assert_array_equal(left[0], right[0])
    np.testing.assert_array_equal(left[1], right[1])
    np.testing.assert_array_equal(left[2][0], right[2][0])


def test_split_preserves_history_and_every_heldout_target():
    df = generate_synthetic_data(20)
    xtr, ytr, xv, yv, xt, yt = temporal_feature_split(df, ["base", "winding"])
    assert (len(ytr), len(yv), len(yt)) == (13, 3, 3)
    binary = build_features(df, ["base"])
    np.testing.assert_array_equal(xv[0, :49], binary[13])
    np.testing.assert_array_equal(yv[0], binary[14])
    np.testing.assert_array_equal(xt[0, :49], binary[16])
    np.testing.assert_array_equal(yt[0], binary[17])
    # The first held-out rolling window includes draws from the training set.
    np.testing.assert_allclose(xv[0, 49 + 49:49 + 98], binary[10:14].mean(axis=0))


def test_winding_difference_has_consistent_inclusion_probability_units():
    df = generate_synthetic_data(20)
    winding = winding_history_features(df)
    np.testing.assert_allclose(winding[:, -49:].sum(axis=1), 0, atol=1e-6)


def test_rolling_baseline_reads_block_at_exact_boundary():
    x = np.zeros((1, 49), dtype=np.float32)
    x[0, :6] = 1
    model = RollingFreqPredictor(rolling_freq_feature_index=0)
    model.fit(x, x)
    different = np.roll(x, 10, axis=1)
    np.testing.assert_array_equal(model.predict_proba(different), different)


def test_probability_check_validates_each_row_and_bounds():
    valid = np.full((2, 49), 6 / 49)
    assert_probability_scale(valid)
    for invalid in [valid * [[0.5], [1.5]], np.full((2, 49), np.nan),
                    np.array([[7, -1] + [0] * 47]), np.full((2, 48), 6 / 48)]:
        with pytest.raises(AssertionError):
            assert_probability_scale(invalid)


@pytest.mark.parametrize("model_cls", [LogisticRegressionModel, MLPModel])
def test_models_preserve_constant_zero_and_one_labels(model_cls):
    pytest.importorskip("sklearn")
    x = np.zeros((3, 2), dtype=np.float32)
    y = np.zeros((3, 49), dtype=np.float32)
    y[:, :6] = 1
    model = model_cls().fit(x, y)
    np.testing.assert_array_equal(model.predict_proba(x), y)


def test_ticket_hit_metrics_use_six_main_targets_only():
    y = np.zeros((1, 49), dtype=np.float32)
    y[0, :6] = 1
    metrics = compute_all_metrics(y, y)
    assert metrics["avg_hits_6"] == 6
    assert metrics["ticket_6_rate"] == 1
    assert metrics["brier"] == 0
    assert np.isfinite(metrics["bce"])
    assert metrics["bce"] < 1e-7
    with pytest.raises(ValueError, match="six main"):
        compute_all_metrics(np.ones((1, 49)), np.full((1, 49), 6 / 49))
