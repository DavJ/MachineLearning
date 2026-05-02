"""
Torus Embedding — v2
====================
Utilities for mapping Sportka draw data onto a 7×7 toroidal grid.

The bijection number → grid cell is:

    row = (number - 1) // 7
    col = (number - 1) % 7

giving row bands:
    row 0: numbers  1– 7
    row 1: numbers  8–14
    …
    row 6: numbers 43–49

Toroidal (periodic) boundary conditions are enforced via np.roll, so
convolution-like operations wrap around both edges seamlessly.
"""

from __future__ import annotations

import numpy as np
from typing import List


def numbers_to_grid(numbers: List[int]) -> np.ndarray:
    """
    Map a list of drawn numbers (1–49) to a 7×7 binary grid.

    row = (number - 1) // 7,  col = (number - 1) % 7

    Args:
        numbers: List of integers in [1, 49].

    Returns:
        (7, 7) binary float32 array.
    """
    grid = np.zeros((7, 7), dtype=np.float32)
    for num in numbers:
        r, c = (num - 1) // 7, (num - 1) % 7
        grid[r, c] = 1.0
    return grid


def binary_to_grid(binary: np.ndarray) -> np.ndarray:
    """
    Reshape a 49-dimensional binary (or continuous) vector to a 7×7 grid.

    Args:
        binary: (49,) float array.

    Returns:
        (7, 7) float32 array.
    """
    binary = np.asarray(binary, dtype=np.float32)
    if binary.shape != (49,):
        raise ValueError(f"Expected (49,) vector, got {binary.shape}")
    return binary.reshape(7, 7)


def grid_to_binary(grid: np.ndarray) -> np.ndarray:
    """
    Flatten a 7×7 grid back to a 49-dimensional vector.

    Args:
        grid: (7, 7) array.

    Returns:
        (49,) float32 array.
    """
    grid = np.asarray(grid, dtype=np.float32)
    if grid.shape != (7, 7):
        raise ValueError(f"Expected (7, 7) grid, got {grid.shape}")
    return grid.ravel()


def toroidal_roll(grid: np.ndarray, dr: int, dc: int) -> np.ndarray:
    """
    Shift a 7×7 grid by (dr, dc) with toroidal (periodic) wrapping.

    Args:
        grid: (7, 7) float array.
        dr:   Row shift (positive → down).
        dc:   Column shift (positive → right).

    Returns:
        Shifted (7, 7) float32 array.
    """
    return np.roll(np.roll(grid, dr, axis=0), dc, axis=1)


def toroidal_conv2d(
    grid: np.ndarray,
    kernel: np.ndarray,
) -> np.ndarray:
    """
    2D convolution on a 7×7 grid with toroidal (circular) boundary conditions.

    Implemented via np.roll: for each kernel offset (dr, dc) the grid is
    shifted and weighted by the corresponding kernel element.  No external
    dependencies required; works well for small kernels (e.g. 3×3 on 7×7).

    Args:
        grid:   (7, 7) float array.
        kernel: (kH, kW) float array; kH and kW should be odd.

    Returns:
        (7, 7) float32 convolved array.
    """
    grid = np.asarray(grid, dtype=np.float32)
    kernel = np.asarray(kernel, dtype=np.float32)
    kH, kW = kernel.shape
    pH, pW = kH // 2, kW // 2
    out = np.zeros((7, 7), dtype=np.float32)
    for dr in range(-pH, pH + 1):
        for dc in range(-pW, pW + 1):
            w = kernel[dr + pH, dc + pW]
            if w != 0.0:
                out += w * toroidal_roll(grid, -dr, -dc)
    return out


def toroidal_avg_pool(grid: np.ndarray, radius: int = 1) -> np.ndarray:
    """
    Toroidal average pooling over a (2*radius+1)^2 neighbourhood.

    Args:
        grid:   (7, 7) float array.
        radius: Neighbourhood radius (default 1 → 3×3 window, 9 cells).

    Returns:
        (7, 7) float32 smoothed array.
    """
    size = 2 * radius + 1
    kernel = np.ones((size, size), dtype=np.float32) / (size * size)
    return toroidal_conv2d(grid, kernel)
