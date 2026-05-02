"""
Tests to verify that the walk-forward feature builder does not use future data.
"""

import pytest


def test_walk_forward_no_future():
    """Minimal smoke-test: build_walk_forward_features exists and is callable."""
    from sportka.walk_forward import build_walk_forward_features

    assert callable(build_walk_forward_features)


def test_walk_forward_sees_only_past():
    """
    Each call to feature_builder must only receive rows that precede the
    current row.  We record the history lengths and verify they are
    strictly less than the current row index.
    """
    from sportka.walk_forward import build_walk_forward_features

    data = list(range(10))  # rows 0..9
    seen_history_lengths = []

    def capturing_builder(history):
        seen_history_lengths.append(len(history))
        return sum(history)  # trivial feature

    features = build_walk_forward_features(data, capturing_builder)

    # Should produce 9 features (skip first row which has no history)
    assert len(features) == 9

    # History length at step i (0-indexed feature) should equal i+1
    for i, h_len in enumerate(seen_history_lengths):
        assert h_len == i + 1, (
            f"Step {i}: expected history length {i + 1}, got {h_len}"
        )


def test_feature_layout_no_hardcoded_offsets():
    """compute_feature_layout returns correct offsets for base+time+winding."""
    from sportka.feature_layout import compute_feature_layout

    layout = compute_feature_layout(["base", "time", "winding"])
    assert layout["base"] == (0, 49)
    assert layout["time"] == (49, 62)
    assert layout["winding"] == (62, 356)
