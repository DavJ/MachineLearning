"""
Tests that verify walk-forward evaluation contains no future leakage.
"""

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# build_walk_forward_features: strict past-only access
# ---------------------------------------------------------------------------

def test_walk_forward_builder_no_leakage():
    """
    Each call to feature_builder must only receive draws strictly before
    the current draw.  History length at step i must equal i+1.
    """
    from sportka.walk_forward import build_walk_forward_features

    data = list(range(15))
    seen = []

    def capturing(history):
        seen.append(len(history))
        return sum(history)

    features = build_walk_forward_features(data, capturing)

    # 14 predictions (skipping first row which cannot be predicted)
    assert len(features) == 14

    # History at step i must be i+1 (grows by 1 each step)
    for i, h in enumerate(seen):
        assert h == i + 1, f"Step {i}: expected history {i+1}, got {h}"


def test_walk_forward_builder_returns_correct_values():
    """Feature values must be computed from past rows only."""
    from sportka.walk_forward import build_walk_forward_features

    data = [1, 2, 3, 4, 5]

    def sum_builder(history):
        return sum(history)

    features = build_walk_forward_features(data, sum_builder)

    # Step 0: history=[1]       → feature=1
    # Step 1: history=[1,2]     → feature=3
    # Step 2: history=[1,2,3]   → feature=6
    # Step 3: history=[1,2,3,4] → feature=10
    expected = [1, 3, 6, 10]
    assert features == expected, f"Expected {expected}, got {features}"


# ---------------------------------------------------------------------------
# _align_xy: targets shifted by one step (no leakage from Y into X)
# ---------------------------------------------------------------------------

def test_align_xy_temporal_shift():
    """
    _align_xy must return X[i] = features of draw i and Y[i] = draw i+1,
    so the model only sees past data when predicting the next draw.
    """
    from sportka.evaluation import _align_xy

    import pandas as pd
    import datetime

    # Build a tiny deterministic DataFrame with 5 draws
    rows = []
    for i in range(5):
        date = datetime.date(2020, 1, 1) + datetime.timedelta(weeks=i)
        nums = [(i * 7 + j) % 49 + 1 for j in range(7)]
        rows.append({"date": pd.Timestamp(date), "numbers": nums})

    df = pd.DataFrame(rows)
    df["draw_index"] = np.arange(len(df))

    X, Y = _align_xy(df, ["base"])

    # With 5 draws there should be 4 (X, Y) pairs
    assert len(X) == 4
    assert len(Y) == 4

    # X[0] must match features of draw 0, Y[0] must match draw 1
    # base_features: 49-dim binary indicator
    from sportka.evaluation import _binary_matrix
    binary = _binary_matrix(df)

    np.testing.assert_array_equal(X[0], binary[0])
    np.testing.assert_array_equal(Y[0], binary[1])


# ---------------------------------------------------------------------------
# feature_layout: no hardcoded magic offsets
# ---------------------------------------------------------------------------

def test_feature_layout_consistency():
    """compute_feature_layout offsets must match FEATURE_DIMS values."""
    from sportka.feature_layout import compute_feature_layout, FEATURE_DIMS

    groups = ["base", "time", "winding", "torus", "theta"]
    layout = compute_feature_layout(groups)

    offset = 0
    for g in groups:
        start, end = layout[g]
        assert start == offset, f"{g}: expected start={offset}, got {start}"
        assert end == offset + FEATURE_DIMS[g], (
            f"{g}: expected end={offset + FEATURE_DIMS[g]}, got {end}"
        )
        offset = end


# ---------------------------------------------------------------------------
# Time normalisation: global range, not per-split
# ---------------------------------------------------------------------------

def test_time_features_global_normalisation():
    """
    When max_draw_index is provided, complex_time_features must use it for
    normalisation so the val/test splits are not re-centred to [0, 1].
    """
    from sportka.features import complex_time_features
    import pandas as pd
    import datetime

    rows = []
    for i in range(10):
        date = datetime.date(2020, 1, 1) + datetime.timedelta(weeks=i)
        rows.append({"date": pd.Timestamp(date), "draw_index": i})

    df_all = pd.DataFrame(rows)
    df_val = df_all.iloc[7:].copy().reset_index(drop=True)

    # Per-split normalisation (default) makes val start at 0
    feat_per_split = complex_time_features(df_val, max_draw_index=None)
    assert feat_per_split[0, 0] == pytest.approx(0.0, abs=1e-5), (
        "Per-split: first val row should normalise to 0"
    )

    # Global normalisation keeps val indices above 0
    global_max = float(df_all["draw_index"].max())
    feat_global = complex_time_features(df_val, max_draw_index=global_max)
    assert feat_global[0, 0] > 0.5, (
        f"Global: first val row index ({feat_global[0, 0]:.3f}) should be > 0.5"
    )
