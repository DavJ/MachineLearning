"""
Feature layout utilities for the Sportka experiment.

Provides a single source of truth for the start/end offsets of each
feature group within the combined feature matrix, removing hardcoded
magic numbers such as ``49 + 13``.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

# Canonical dimensions for each feature group (must match features.py)
FEATURE_DIMS: Dict[str, int] = {
    "base": 49,
    "time": 13,
    "winding": 294,
    "torus": 112,
    "theta": 98,
}


def compute_feature_layout(
    groups: List[str],
) -> Dict[str, Tuple[int, int]]:
    """
    Compute (start, end) offsets for each group in a combined feature vector.

    Args:
        groups: Ordered list of feature group names (e.g. ``["base", "time",
                "winding"]``).  Must be keys of ``FEATURE_DIMS``.

    Returns:
        Dict mapping group name → (start_offset, end_offset) where
        ``combined_vector[start:end]`` selects that group's features.

    Example::

        layout = compute_feature_layout(["base", "time", "winding"])
        # layout == {"base": (0, 49), "time": (49, 62), "winding": (62, 356)}
    """
    layout: Dict[str, Tuple[int, int]] = {}
    offset = 0
    for name in groups:
        dim = FEATURE_DIMS[name]
        layout[name] = (offset, offset + dim)
        offset += dim
    return layout
