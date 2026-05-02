"""
Evaluation utilities for the Sportka UBT/theta experiment.

Provides:
  - draw-level metrics: binary cross-entropy, top-k recall, KL divergence,
    average hits per draw
  - walk-forward evaluation loop
  - bootstrap confidence intervals (≥1000 resamples)
  - control tests: shuffled-labels and reversed-time
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Callable

from sportka.features import build_features, walk_forward_split


# ---------------------------------------------------------------------------
# Draw-level metrics
# ---------------------------------------------------------------------------

EPS = 1e-9  # numerical stability


def assert_probability_scale(p: np.ndarray) -> None:
    """
    Assert that a 49-element probability vector has the expected sum (~7).

    Each element should represent P(number appears in a draw).  Since exactly
    7 numbers are drawn each time, the sum over all 49 numbers must be ≈ 7.

    Args:
        p: 1-D array of length 49 (or 2-D (n, 49) where the row mean is checked).
    Raises:
        AssertionError if the sum is outside [6.0, 8.0].
    """
    arr = np.asarray(p, dtype=np.float64)
    if arr.ndim == 2:
        s = float(arr.mean(axis=0).sum())
    else:
        s = float(arr.sum())
    assert 6.0 < s < 8.0, f"Probability sum out of range: {s:.4f} (expected ~7)"


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean binary cross-entropy over all (draw, number) pairs.

    Args:
        y_true: (n, 49) binary ground-truth matrix
        y_pred: (n, 49) predicted probability matrix
    Returns:
        Scalar BCE (lower is better).
    """
    p = np.clip(y_pred, EPS, 1 - EPS)
    bce = -(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    return float(bce.mean())


def topk_recall(y_true: np.ndarray, y_pred: np.ndarray, k: int = 6) -> float:
    """
    Average recall@k: for each draw select the top-k predicted numbers,
    then compute the fraction of true positives recovered.

    Args:
        y_true: (n, 49) binary matrix
        y_pred: (n, 49) probability matrix
        k:      number of numbers to select per draw
    Returns:
        Mean recall@k ∈ [0, 1].
    """
    n = len(y_true)
    recalls = []
    for i in range(n):
        top_k_idx = np.argsort(y_pred[i])[-k:]
        true_set = np.where(y_true[i] == 1)[0]
        hits = len(set(top_k_idx) & set(true_set))
        recall = hits / max(len(true_set), 1)
        recalls.append(recall)
    return float(np.mean(recalls))


def kl_divergence_vs_uniform(y_pred: np.ndarray) -> float:
    """
    Mean KL divergence from the uniform distribution (1/49 for each number)
    to the predicted distribution, averaged over draws.

    KL(uniform || pred) = Σ_k (1/49) * log( (1/49) / pred_k )

    A model that is perfectly uniform has KL=0; one that concentrates mass
    will have KL > 0.  This measures how much predictions deviate from random.
    """
    p_uniform = np.full(49, 1.0 / 49.0, dtype=np.float64)
    # Normalise predictions to a probability distribution over the 49 numbers
    # (row-wise softmax normalisation)
    pred = y_pred.astype(np.float64)
    pred = np.clip(pred, EPS, None)
    pred = pred / pred.sum(axis=1, keepdims=True)

    kl_per_draw = np.sum(p_uniform * np.log(p_uniform / pred), axis=1)
    return float(kl_per_draw.mean())


def avg_hits(y_true: np.ndarray, y_pred: np.ndarray, k: int = 6) -> float:
    """
    Average number of correct hits when the top-k predicted numbers are chosen.
    """
    n = len(y_true)
    hits_list = []
    for i in range(n):
        top_k_idx = np.argsort(y_pred[i])[-k:]
        hits = int(y_true[i, top_k_idx].sum())
        hits_list.append(hits)
    return float(np.mean(hits_list))


def compute_all_metrics(
    y_true: np.ndarray, y_pred: np.ndarray
) -> Dict[str, float]:
    """Compute all evaluation metrics and return as a dict."""
    return {
        "bce": binary_cross_entropy(y_true, y_pred),
        "recall_at_6": topk_recall(y_true, y_pred, k=6),
        "recall_at_10": topk_recall(y_true, y_pred, k=10),
        "kl_vs_uniform": kl_divergence_vs_uniform(y_pred),
        "avg_hits_6": avg_hits(y_true, y_pred, k=6),
        "avg_hits_10": avg_hits(y_true, y_pred, k=10),
    }


# ---------------------------------------------------------------------------
# Walk-forward evaluation
# ---------------------------------------------------------------------------

def walk_forward_eval(
    model,
    df: pd.DataFrame,
    feature_groups: List[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Fit the model on the training split and evaluate on val and test splits.

    Uses temporal alignment: X[i] predicts draw i+1 (see _align_xy).
    Time features are normalised using the global training-set draw-index
    range to avoid per-split re-centring.

    Returns:
        (val_metrics, test_metrics) — dicts from compute_all_metrics.
    """
    train_df, val_df, test_df = walk_forward_split(df, train_frac, val_frac)

    # Determine global normalisation range from the training split only.
    # Always computed; only used when 'time' features are requested (build_features ignores it otherwise).
    max_draw_index = float(train_df["draw_index"].max())

    X_train, Y_train = _align_xy(train_df, feature_groups, max_draw_index=max_draw_index)
    X_val,   Y_val   = _align_xy(val_df,   feature_groups, max_draw_index=max_draw_index)
    X_test,  Y_test  = _align_xy(test_df,  feature_groups, max_draw_index=max_draw_index)

    model.fit(X_train, Y_train)

    val_pred  = model.predict_proba(X_val)
    test_pred = model.predict_proba(X_test)

    return (
        compute_all_metrics(Y_val,  val_pred),
        compute_all_metrics(Y_test, test_pred),
    )


def _binary_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert numbers column to (n, 49) binary matrix."""
    n = len(df)
    mat = np.zeros((n, 49), dtype=np.float32)
    for i, nums in enumerate(df["numbers"]):
        for num in nums:
            mat[i, num - 1] = 1.0
    return mat


def _align_xy(
    df: pd.DataFrame,
    feature_groups: List[str],
    max_draw_index: float | None = None,
):
    """
    Build temporally aligned X and Y matrices.

    X[i] = features of draw i   (used as predictor)
    Y[i] = binary vector of draw i+1  (the next draw to predict)

    This avoids data leakage: features of draw i cannot contain draw i+1's
    numbers.  The last draw has no target, so n-1 pairs are returned.

    Args:
        max_draw_index: Passed to build_features for consistent time
            normalisation across all splits (use the training-set max).
    """
    X_all = build_features(df, feature_groups, max_draw_index=max_draw_index)   # (n, d)
    Y_all = _binary_matrix(df)                                                    # (n, 49)
    return X_all[:-1], Y_all[1:]                                                  # (n-1, d), (n-1, 49)


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: Callable,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap confidence interval for a scalar metric.

    Args:
        y_true:      (n, 49) ground-truth
        y_pred:      (n, 49) predictions
        metric_fn:   function(y_true, y_pred) -> float
        n_resamples: number of bootstrap resamples
        ci:          confidence level (0.95 → 95%)
        seed:        RNG seed for reproducibility

    Returns:
        (point_estimate, lower_bound, upper_bound)
    """
    rng = np.random.default_rng(seed)
    n = len(y_true)
    point = metric_fn(y_true, y_pred)
    stats = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        stats.append(metric_fn(y_true[idx], y_pred[idx]))
    alpha = (1 - ci) / 2
    lower = float(np.percentile(stats, 100 * alpha))
    upper = float(np.percentile(stats, 100 * (1 - alpha)))
    return point, lower, upper


def bootstrap_ci_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_resamples: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Bootstrap CI for all metrics.  Returns dict:
        metric_name -> (point, lower, upper)
    """
    metric_fns = {
        "bce": binary_cross_entropy,
        "recall_at_6": lambda yt, yp: topk_recall(yt, yp, k=6),
        "recall_at_10": lambda yt, yp: topk_recall(yt, yp, k=10),
        "kl_vs_uniform": lambda yt, yp: kl_divergence_vs_uniform(yp),
        "avg_hits_6": lambda yt, yp: avg_hits(yt, yp, k=6),
        "avg_hits_10": lambda yt, yp: avg_hits(yt, yp, k=10),
    }
    results = {}
    for name, fn in metric_fns.items():
        results[name] = bootstrap_ci(
            y_true, y_pred, fn, n_resamples=n_resamples, ci=ci, seed=seed
        )
    return results


# ---------------------------------------------------------------------------
# Control tests
# ---------------------------------------------------------------------------

def shuffled_labels_test(
    model,
    df: pd.DataFrame,
    feature_groups: List[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 0,
) -> Dict[str, float]:
    """
    Shuffled-labels control: randomly permute the rows of Y_train to destroy
    any signal.  A model that truly learns should perform no better than
    random on this control.  Returns test-split metrics.
    """
    rng = np.random.default_rng(seed)

    train_df, val_df, test_df = walk_forward_split(df, train_frac, val_frac)

    X_train, Y_train = _align_xy(train_df, feature_groups)
    Y_train_shuffled = Y_train[rng.permutation(len(Y_train))]

    X_test, Y_test = _align_xy(test_df, feature_groups)

    model.fit(X_train, Y_train_shuffled)
    y_pred = model.predict_proba(X_test)
    return compute_all_metrics(Y_test, y_pred)


def reversed_time_test(
    model,
    df: pd.DataFrame,
    feature_groups: List[str],
    train_frac: float = 0.70,
    val_frac: float = 0.15,
) -> Dict[str, float]:
    """
    Reversed-time control: reverse the chronological order of the data
    (future is used to predict the past).  A well-specified causal model
    should perform worse in this setting; if it performs the same, the
    feature contains no real temporal signal.

    Returns test-split metrics on the reversed-time data.
    """
    df_rev = df.iloc[::-1].reset_index(drop=True)
    df_rev["draw_index"] = np.arange(len(df_rev))

    train_df, val_df, test_df = walk_forward_split(df_rev, train_frac, val_frac)

    X_train, Y_train = _align_xy(train_df, feature_groups)
    X_test,  Y_test  = _align_xy(test_df,  feature_groups)

    model.fit(X_train, Y_train)
    y_pred = model.predict_proba(X_test)
    return compute_all_metrics(Y_test, y_pred)


# ---------------------------------------------------------------------------
# Performance-over-time helper
# ---------------------------------------------------------------------------

def rolling_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    window: int = 50,
) -> pd.DataFrame:
    """
    Compute rolling-window metrics over the test set for time-series plots.

    Returns a DataFrame indexed by draw number with columns for each metric.
    """
    n = len(y_true)
    rows = []
    for i in range(n):
        start = max(0, i - window + 1)
        yt = y_true[start: i + 1]
        yp = y_pred[start: i + 1]
        row = compute_all_metrics(yt, yp)
        row["draw"] = i
        rows.append(row)
    return pd.DataFrame(rows).set_index("draw")
