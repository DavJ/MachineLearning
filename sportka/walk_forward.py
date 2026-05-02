"""
Walk-forward feature builder for the Sportka experiment.

Ensures that when building features for validation or test rows, each row
only has access to past draws — no future leakage.
"""

from __future__ import annotations

from typing import Callable, List, Any


def build_walk_forward_features(
    data: List[Any],
    feature_builder: Callable[[List[Any]], Any],
) -> List[Any]:
    """
    Build features sequentially so each row only sees past draws.

    For each row at position i, the feature builder is called with
    ``history[:i]`` (all rows strictly before i), so no future data
    is visible.

    Args:
        data:            List of raw rows (e.g. dicts or lists) in
                         chronological order.
        feature_builder: Callable that takes a list of past rows and
                         returns a feature vector for the *next* draw.

    Returns:
        List of feature vectors; length = len(data) - 1 (the first row
        has no history so it is skipped as a prediction target).
    """
    X: List[Any] = []
    history: List[Any] = []

    for row in data:
        if len(history) == 0:
            history.append(row)
            continue

        x = feature_builder(history)
        X.append(x)
        history.append(row)

    return X
