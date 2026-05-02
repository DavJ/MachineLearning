"""
UBT Theta Transform — v2
========================
Implements the truncated Jacobi-theta-like spectral transform used by the
UBT v2 pipeline.

For a scalar value v the transform is:

    θ(v; α, N) = Σ_{n=0}^{N}  exp(-α · n²) · cos(2π · n · v)

This is the real part of the standard Jacobi theta function θ₃ with
nome q = exp(-α), evaluated at the point 2πv.  By truncating at N harmonics
we control smoothness: large α damps high harmonics quickly; small N limits
computational cost.

Applied element-wise over the 7×7 toroidal grid produced by torus_embedding.

Hyperparameter guidance (from spec):
    N     ~ 7         (number of harmonics)
    alpha ~ 0.5       (spectral decay; tunable)
"""

from __future__ import annotations

import numpy as np

# Default hyperparameters (from problem spec)
DEFAULT_N: int = 7
DEFAULT_ALPHA: float = 0.5


def theta_transform(
    v: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    N: int = DEFAULT_N,
) -> np.ndarray:
    """
    Apply the truncated theta-like transform element-wise.

        θ(v) = Σ_{n=0}^{N} exp(-α · n²) · cos(2π · n · v)

    Args:
        v:     Input array of any shape; values are typically in [0, 1].
        alpha: Spectral decay rate (> 0).  Larger → faster decay of harmonics.
        N:     Truncation order (number of harmonics, inclusive).

    Returns:
        Transformed float32 array of the same shape as v.
    """
    v_f = np.asarray(v, dtype=np.float64)
    out = np.zeros_like(v_f)
    two_pi_v = 2.0 * np.pi * v_f
    for n in range(N + 1):
        out += np.exp(-alpha * n * n) * np.cos(n * two_pi_v)
    return out.astype(np.float32)


def theta_transform_grid(
    grid: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    N: int = DEFAULT_N,
) -> np.ndarray:
    """
    Apply the theta transform to a single 7×7 grid.

    Args:
        grid:  (7, 7) float array; values typically in [0, 1].
        alpha: Spectral decay rate.
        N:     Truncation order.

    Returns:
        (7, 7) float32 transformed grid.
    """
    grid = np.asarray(grid, dtype=np.float32)
    if grid.shape != (7, 7):
        raise ValueError(f"Expected (7, 7) grid, got {grid.shape}")
    return theta_transform(grid, alpha=alpha, N=N)


def theta_transform_batch(
    grids: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
    N: int = DEFAULT_N,
) -> np.ndarray:
    """
    Batch-apply the theta transform to multiple 7×7 grids.

    Args:
        grids: (T, 7, 7) float array.
        alpha: Spectral decay rate.
        N:     Truncation order.

    Returns:
        (T, 7, 7) float32 transformed grids.
    """
    grids = np.asarray(grids, dtype=np.float32)
    if grids.ndim != 3 or grids.shape[1:] != (7, 7):
        raise ValueError(f"Expected (T, 7, 7) grids, got {grids.shape}")
    return theta_transform(grids, alpha=alpha, N=N)
