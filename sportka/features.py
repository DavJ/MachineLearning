"""
Feature engineering for the Sportka UBT/theta experiment.

Features are grouped into five families:
  1. base_features        – 49-dim binary vector per draw
  2. complex_time_features – normalised time index + cyclical sin/cos encodings
  3. winding_history       – rolling / exponential-decay frequency vectors
  4. torus_embedding       – 7×7 grid projections and toroidal convolutions
  5. theta_features        – truncated Jacobi-theta-like transforms (experimental)

All feature functions accept a pandas DataFrame with columns:
    draw_index  int      monotone draw counter (chronological)
    date        datetime
    numbers     list[int]  7 numbers drawn (first game)

and return a 2-D numpy array  (n_draws, n_features).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import List


# ---------------------------------------------------------------------------
# 1. Base features
# ---------------------------------------------------------------------------

def base_features(df: pd.DataFrame) -> np.ndarray:
    """49-dimensional binary indicator vector for each draw."""
    n = len(df)
    out = np.zeros((n, 49), dtype=np.float32)
    for i, nums in enumerate(df["numbers"]):
        for num in nums:
            out[i, num - 1] = 1.0
    return out


# ---------------------------------------------------------------------------
# 2. Complex time features
# ---------------------------------------------------------------------------

def complex_time_features(df: pd.DataFrame, max_draw_index: float | None = None) -> np.ndarray:
    """
    Cyclical and normalised temporal features.

    Returns an array of shape (n, 13):
        [0]  normalised draw index  (0 → 1)
        [1]  sin(weekday)  [2] cos(weekday)
        [3]  sin(month)    [4] cos(month)
        [5..12] sin/cos for draw-index periods [4, 16, 52, 104]

    Args:
        max_draw_index: If provided, normalise the draw index by this value
            (avoids per-split min/max normalisation).  If None, falls back
            to per-split range normalisation.
    """
    n = len(df)
    idx = df["draw_index"].values.astype(np.float32)
    if max_draw_index is not None:
        norm_idx = idx / max(float(max_draw_index), 1.0)
    else:
        norm_idx = (idx - idx.min()) / max(idx.max() - idx.min(), 1.0)

    dates = pd.to_datetime(df["date"])
    weekday = dates.dt.dayofweek.values.astype(np.float32)  # 0–6
    month = dates.dt.month.values.astype(np.float32)        # 1–12

    sin_wd = np.sin(2 * np.pi * weekday / 7)
    cos_wd = np.cos(2 * np.pi * weekday / 7)
    sin_mo = np.sin(2 * np.pi * (month - 1) / 12)
    cos_mo = np.cos(2 * np.pi * (month - 1) / 12)

    cols = [norm_idx, sin_wd, cos_wd, sin_mo, cos_mo]
    for period in [4, 16, 52, 104]:
        cols.append(np.sin(2 * np.pi * idx / period).astype(np.float32))
        cols.append(np.cos(2 * np.pi * idx / period).astype(np.float32))

    return np.stack(cols, axis=1)  # (n, 13)


# ---------------------------------------------------------------------------
# 3. Winding / history features
# ---------------------------------------------------------------------------

def _rolling_freq(numbers_list: List[List[int]], window: int) -> np.ndarray:
    """
    For each draw i, compute frequency of each number in the preceding
    `window` draws (draws [i-window, i]).  Returns (n, 49).

    Note: draw i is *included* because these features are used to predict
    draw i+1 (the temporal target is shifted by one step outside this function).
    """
    n = len(numbers_list)
    out = np.zeros((n, 49), dtype=np.float32)
    for i in range(n):
        start = max(0, i - window + 1)  # include draw i itself
        count = i - start + 1
        freq = np.zeros(49, dtype=np.float32)
        for nums in numbers_list[start: i + 1]:
            for num in nums:
                freq[num - 1] += 1.0
        out[i] = freq / count  # P(number in draw); sum ≈ 7
    return out


def _exp_decay_freq(numbers_list: List[List[int]], decay: float = 0.05) -> np.ndarray:
    """
    Exponentially decayed historical frequency.  Weight of draw k draws
    back is exp(-decay * k).  Draw i itself is weight 1 (k=0).  Returns (n, 49).
    """
    n = len(numbers_list)
    out = np.zeros((n, 49), dtype=np.float32)
    for i in range(n):
        freq = np.zeros(49, dtype=np.float32)
        total_weight = 0.0
        for k, j in enumerate(range(i, -1, -1)):  # include draw i (k=0)
            w = np.exp(-decay * k)
            for num in numbers_list[j]:
                freq[num - 1] += w
            total_weight += w
        if total_weight > 0:
            out[i] = freq / total_weight
        else:
            out[i] = 1.0 / 49.0
    return out


def winding_history_features(df: pd.DataFrame) -> np.ndarray:
    """
    Returns (n, 49*6) array:
        rolling windows [1, 4, 16, 52] × 49  +  exp-decay × 49  +  diff-from-global × 49
    """
    numbers_list: List[List[int]] = df["numbers"].tolist()
    n = len(numbers_list)

    # Global frequency (computed on the entire training set for reference;
    # during walk-forward evaluation this is only the training portion)
    global_freq = np.zeros(49, dtype=np.float32)
    for nums in numbers_list:
        for num in nums:
            global_freq[num - 1] += 1.0
    total = global_freq.sum()
    global_freq = global_freq / total if total > 0 else np.full(49, 1.0 / 49.0, dtype=np.float32)

    parts = []
    for w in [1, 4, 16, 52]:
        parts.append(_rolling_freq(numbers_list, w))

    exp_freq = _exp_decay_freq(numbers_list)
    parts.append(exp_freq)
    parts.append(exp_freq - global_freq[np.newaxis, :])  # diff from global

    return np.concatenate(parts, axis=1)  # (n, 49*6 = 294)


# ---------------------------------------------------------------------------
# 4. Torus embedding
# ---------------------------------------------------------------------------

def _numbers_to_grid(numbers: List[int]) -> np.ndarray:
    """Map numbers 1–49 onto a 7×7 binary grid (row = (n-1)//7, col = (n-1)%7)."""
    grid = np.zeros((7, 7), dtype=np.float32)
    for num in numbers:
        r, c = (num - 1) // 7, (num - 1) % 7
        grid[r, c] = 1.0
    return grid


def _toroidal_neighbor_count(grid: np.ndarray) -> np.ndarray:
    """
    For each cell, count how many of its 8 toroidal neighbours are active.
    Returns a (7, 7) array.
    """
    rolled = np.zeros((7, 7), dtype=np.float32)
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            rolled += np.roll(np.roll(grid, dr, axis=0), dc, axis=1)
    return rolled


def _torus_conv3x3(grid: np.ndarray) -> np.ndarray:
    """
    Local 3×3 average convolution on the toroidal grid.
    Returns a (7, 7) array.
    """
    out = np.zeros((7, 7), dtype=np.float32)
    for dr in range(-1, 2):
        for dc in range(-1, 2):
            out += np.roll(np.roll(grid, dr, axis=0), dc, axis=1)
    return out / 9.0


def torus_embedding_features(df: pd.DataFrame) -> np.ndarray:
    """
    Returns (n, 7+7+49+49) = (n, 112) array:
        row_sums(7) + col_sums(7) + neighbor_counts(49) + conv3x3(49)
    """
    n = len(df)
    parts = []
    for nums in df["numbers"]:
        grid = _numbers_to_grid(nums)
        row_sums = grid.sum(axis=1)      # (7,)
        col_sums = grid.sum(axis=0)      # (7,)
        neigh = _toroidal_neighbor_count(grid).ravel()  # (49,)
        conv = _torus_conv3x3(grid).ravel()             # (49,)
        parts.append(np.concatenate([row_sums, col_sums, neigh, conv]))
    return np.stack(parts, axis=0).astype(np.float32)   # (n, 112)


# ---------------------------------------------------------------------------
# 5. Theta features (experimental)
# ---------------------------------------------------------------------------

def _theta3_approx(z: float, alpha: float = 0.5, N: int = 20) -> float:
    """
    Truncated Jacobi theta-like function:
        θ₃(z) ≈ Σ_{n=0}^{N} exp(-alpha * n²) * cos(2π n z)
    Normalised so the n=0 term contributes 1.
    """
    total = 0.0
    for n_val in range(N + 1):
        total += np.exp(-alpha * n_val * n_val) * np.cos(2 * np.pi * n_val * z)
    return total


# Pre-compute theta values for all 49 numbers once.
_THETA_VALUES: np.ndarray | None = None
_THETA_ALPHA = 0.5
_THETA_N = 20


def _get_theta_values() -> np.ndarray:
    global _THETA_VALUES
    if _THETA_VALUES is None:
        vals = np.array(
            [_theta3_approx(k / 49.0, alpha=_THETA_ALPHA, N=_THETA_N) for k in range(1, 50)],
            dtype=np.float32,
        )
        # Normalise to [0, 1]
        v_min, v_max = vals.min(), vals.max()
        if v_max > v_min:
            vals = (vals - v_min) / (v_max - v_min)
        _THETA_VALUES = vals
    return _THETA_VALUES


def theta_features(df: pd.DataFrame) -> np.ndarray:
    """
    Experimental truncated-theta projection of each draw.

    For each draw returns a (49,) vector where position k receives
    the pre-computed θ₃(k/49) weight if number k+1 was drawn, else 0,
    plus a (49,) vector of element-wise products of the draw binary
    vector with the theta weights (capturing the 'theta-weighted' draw).

    Returns (n, 98) array  — marked experimental.
      [:49]  theta_vals[k] if number k+1 was drawn, else 0  (drawn-number projection)
      [49:]  theta_vals[k] if number k+1 was NOT drawn, else 0  (complement projection)
    """
    theta_vals = _get_theta_values()  # (49,)
    n = len(df)
    out = np.zeros((n, 98), dtype=np.float32)
    for i, nums in enumerate(df["numbers"]):
        binary = np.zeros(49, dtype=np.float32)
        for num in nums:
            binary[num - 1] = 1.0
        # theta projection of drawn numbers
        out[i, :49] = binary * theta_vals
        # theta projection of non-drawn numbers (complement)
        out[i, 49:] = (1.0 - binary) * theta_vals
    return out


# ---------------------------------------------------------------------------
# Combined feature builder
# ---------------------------------------------------------------------------

FEATURE_GROUPS = {
    "base": base_features,           # 49
    "time": complex_time_features,   # 13
    "winding": winding_history_features,  # 294
    "torus": torus_embedding_features,    # 112
    "theta": theta_features,         # 98  (experimental)
}

FEATURE_DIMS = {
    "base": 49,
    "time": 13,
    "winding": 294,
    "torus": 112,
    "theta": 98,
}


def build_features(
    df: pd.DataFrame,
    groups: List[str] | None = None,
    max_draw_index: float | None = None,
) -> np.ndarray:
    """
    Build a feature matrix from the given groups.

    Args:
        df:             DataFrame with columns [draw_index, date, numbers].
        groups:         Subset of FEATURE_GROUPS keys (default: all).
        max_draw_index: If provided, pass to :func:`complex_time_features` so
                        the draw-index is normalised by the global training
                        range rather than the per-split range.  This prevents
                        the time axis from being re-centred on each split.

    Returns:
        np.ndarray of shape (n_draws, total_features).
    """
    if groups is None:
        groups = list(FEATURE_GROUPS.keys())

    arrays = []
    for g in groups:
        if g == "time":
            arrays.append(complex_time_features(df, max_draw_index=max_draw_index))
        else:
            arrays.append(FEATURE_GROUPS[g](df))
    return np.concatenate(arrays, axis=1)


# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------

def load_sportka_csv(path: str) -> pd.DataFrame:
    """
    Load Sportka draw history from the CSV exported by sazka.cz.

    The CSV format has columns (semicolon-separated):
        date ; ? ; week ; weekday ; n1 ; n2 ; n3 ; n4 ; n5 ; n6 ; n7 ; ...

    Returns a DataFrame with columns:
        date         datetime64
        draw_index   int        (0-based, chronological)
        numbers      list[int]  (first 7 numbers)
    """
    raw = pd.read_csv(path, sep=";", header=0, encoding="latin-1")
    # Normalise column names
    raw.columns = [c.strip() for c in raw.columns]

    rows = []
    for _, row in raw.iterrows():
        try:
            vals = row.tolist()
            date_str = str(vals[0]).strip()
            date = pd.to_datetime(date_str, dayfirst=True)
            nums = [int(x) for x in vals[4:11]]
            rows.append({"date": date, "numbers": nums})
        except Exception:
            continue

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["draw_index"] = np.arange(len(df))
    return df


def generate_synthetic_data(n_draws: int = 500, seed: int = 42) -> pd.DataFrame:
    """
    Generate purely random Sportka-like draws for testing (null-hypothesis data).
    Each draw picks 7 numbers without replacement from 1–49.
    """
    rng = np.random.default_rng(seed)
    import datetime

    start = datetime.date(2000, 1, 5)
    rows = []
    for i in range(n_draws):
        date = start + datetime.timedelta(weeks=i // 2, days=(i % 2) * 3)
        nums = sorted(rng.choice(49, size=7, replace=False) + 1)
        rows.append({"date": pd.Timestamp(date), "numbers": list(nums)})

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    df["draw_index"] = np.arange(len(df))
    return df


def walk_forward_split(df: pd.DataFrame, train_frac: float = 0.70, val_frac: float = 0.15):
    """
    Chronological (walk-forward) train / validation / test split.

    Returns (train_df, val_df, test_df).
    """
    n = len(df)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    return (
        df.iloc[:train_end].reset_index(drop=True),
        df.iloc[train_end:val_end].reset_index(drop=True),
        df.iloc[val_end:].reset_index(drop=True),
    )
