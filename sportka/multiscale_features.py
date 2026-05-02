"""
Multi-scale Temporal Features — v2
====================================
Builds the (C, 7, 7) theta-transformed multi-scale tensor for each draw.

Pipeline for draw i:

  Step 1 — Torus embedding
    Map the 49-dim binary vector → 7×7 grid (see torus_embedding.py).

  Step 2 — Rolling average (per window w)
    Compute the mean binary grid over draws [i-w+1, i].
    For w=1 this is the current draw itself.

  Step 3 — Theta transform
    Apply the element-wise theta transform to the averaged 7×7 grid
    (see ubt_theta_transform.py).

  Step 4 — Stack channels
    Concatenate one feature map per window, optionally with the raw
    binary grid as an extra channel.

    Channels (default windows=[1, 16, 52], include_raw=True):
        ch 0: theta( avg_grid(w=1)  )   — short history
        ch 1: theta( avg_grid(w=16) )   — medium history
        ch 2: theta( avg_grid(w=52) )   — long history
        ch 3: raw binary grid            — current draw indicator

    Output shape: (T, C, 7, 7)   where C = len(windows) + include_raw

  Flattening to (T, C*49) is available via build_flat_features() for use
  with sklearn-based models.
"""

from __future__ import annotations

import numpy as np
from typing import List

from sportka.torus_embedding import binary_to_grid
from sportka.ubt_theta_transform import theta_transform_grid, DEFAULT_ALPHA, DEFAULT_N

# Window sizes: short=1, medium=16, long=52 (from spec)
DEFAULT_WINDOWS: List[int] = [1, 16, 52]


def build_multiscale_tensors(
    binary_history: np.ndarray,
    windows: List[int] | None = None,
    alpha: float = DEFAULT_ALPHA,
    N: int = DEFAULT_N,
    include_raw: bool = True,
) -> np.ndarray:
    """
    Build multi-scale theta-transformed feature tensors for all draws.

    Args:
        binary_history: (T, 49) binary float matrix, chronological order.
        windows:        Window sizes for rolling averages.
                        Default: [1, 16, 52] (short, medium, long).
        alpha:          Theta transform decay parameter.
        N:              Theta transform truncation order.
        include_raw:    If True, append the raw 7×7 binary grid as an
                        extra channel (no theta transform applied).

    Returns:
        (T, C, 7, 7) float32 array.
        C = len(windows) + (1 if include_raw else 0)
    """
    if windows is None:
        windows = DEFAULT_WINDOWS

    binary_history = np.asarray(binary_history, dtype=np.float32)
    T = len(binary_history)
    n_channels = len(windows) + (1 if include_raw else 0)
    out = np.zeros((T, n_channels, 7, 7), dtype=np.float32)

    for i in range(T):
        ch = 0
        for w in windows:
            start = max(0, i - w + 1)
            block = binary_history[start : i + 1]   # (<= w, 49)
            avg_binary = block.mean(axis=0)          # (49,)
            avg_grid = binary_to_grid(avg_binary)    # (7, 7)
            out[i, ch] = theta_transform_grid(avg_grid, alpha=alpha, N=N)
            ch += 1
        if include_raw:
            out[i, ch] = binary_history[i].reshape(7, 7)

    return out


def flatten_multiscale_tensors(tensors: np.ndarray) -> np.ndarray:
    """
    Flatten (T, C, 7, 7) tensors to (T, C*49) for flat models.

    Args:
        tensors: (T, C, 7, 7) float32 array.

    Returns:
        (T, C*49) float32 array.
    """
    T = tensors.shape[0]
    return tensors.reshape(T, -1).astype(np.float32)


def build_flat_features(
    binary_history: np.ndarray,
    windows: List[int] | None = None,
    alpha: float = DEFAULT_ALPHA,
    N: int = DEFAULT_N,
    include_raw: bool = True,
) -> np.ndarray:
    """
    Convenience wrapper: build multi-scale tensors and flatten to (T, C*49).

    Args:
        binary_history: (T, 49) binary float matrix, chronological order.
        windows:        Window sizes for rolling averages (default: [1, 16, 52]).
        alpha:          Theta transform decay parameter.
        N:              Theta transform truncation order.
        include_raw:    If True, append raw grid channel.

    Returns:
        (T, C*49) float32 feature matrix.
    """
    tensors = build_multiscale_tensors(
        binary_history, windows=windows, alpha=alpha, N=N, include_raw=include_raw
    )
    return flatten_multiscale_tensors(tensors)
