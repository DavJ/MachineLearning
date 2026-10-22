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
from sportka.rules import MAIN_COUNT, POOL_SIZE


# ---------------------------------------------------------------------------
# Draw-level metrics
# ---------------------------------------------------------------------------

EPS = 1e-9  # numerical stability


def assert_probability_scale(p: np.ndarray) -> None:
    """
    Assert finite inclusion probabilities with each row summing to six.

    This checks each row separately; invalid rows cannot cancel one another.
    """
    arr = np.asarray(p, dtype=np.float64)
    assert arr.ndim in (1, 2) and arr.shape[-1] == POOL_SIZE and arr.size, "Expected (..., 49) probabilities"
    assert np.all(np.isfinite(arr)) and np.all((arr >= 0) & (arr <= 1)), "Probabilities must be finite and in [0, 1]"
    assert np.allclose(arr.sum(axis=-1), MAIN_COUNT, atol=1e-4), "Each probability row must sum to six"


def binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean binary cross-entropy over all (draw, number) pairs.

    Args:
        y_true: (n, 49) binary ground-truth matrix
        y_pred: (n, 49) predicted probability matrix
    Returns:
        Scalar BCE (lower is better).
    """
    # float32 rounds 1 - 1e-9 back to 1, causing log(0) and NaN even
    # for a perfect prediction. Clip in float64, then use stable log1p.
    p = np.clip(np.asarray(y_pred, dtype=np.float64), EPS, 1 - EPS)
    bce = -(y_true * np.log(p) + (1 - y_true) * np.log1p(-p))
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
    """Compute probability scores and six-number ticket hit rates.

    Hit rates do not represent profit: payout amounts and shared jackpots
    are absent. The bonus tier requires the separate bonus-ball column.
    """
    if y_true.shape != y_pred.shape or y_true.ndim != 2 or y_true.shape[1] != POOL_SIZE or not len(y_true):
        raise ValueError("Expected matching nonempty (n, 49) truth and prediction arrays")
    if not np.all(np.isfinite(y_pred)) or not np.all((y_pred >= 0) & (y_pred <= 1)):
        raise ValueError("Predictions must be finite and in [0, 1]")
    if not np.all((y_true == 0) | (y_true == 1)) or not np.all(y_true.sum(axis=1) == MAIN_COUNT):
        raise ValueError("Each target must contain exactly six main numbers")
    top = np.argsort(y_pred, axis=1)[:, -MAIN_COUNT:]
    hits = np.take_along_axis(y_true, top, axis=1).sum(axis=1)
    return {
        "bce": binary_cross_entropy(y_true, y_pred),
        "brier": float(np.mean((y_true - y_pred) ** 2)),
        "recall_at_6": topk_recall(y_true, y_pred, k=6),
        "recall_at_10": topk_recall(y_true, y_pred, k=10),
        "kl_vs_uniform": kl_divergence_vs_uniform(y_pred),
        "avg_hits_6": avg_hits(y_true, y_pred, k=6),
        "avg_hits_10": avg_hits(y_true, y_pred, k=10),
        "ticket_3plus_rate": float(np.mean(hits >= 3)),
        "ticket_4plus_rate": float(np.mean(hits >= 4)),
        "ticket_5plus_rate": float(np.mean(hits >= 5)),
        "ticket_6_rate": float(np.mean(hits == 6)),
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
    X_train, Y_train, X_val, Y_val, X_test, Y_test = temporal_feature_split(
        df, feature_groups, train_frac, val_frac
    )

    model.fit(X_train, Y_train)

    val_pred  = model.predict_proba(X_val)
    test_pred = model.predict_proba(X_test)

    return (
        compute_all_metrics(Y_val,  val_pred),
        compute_all_metrics(Y_test, test_pred),
    )


def _binary_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert numbers column to (n, 49) binary matrix."""
    from sportka.features import base_features
    return base_features(df)


def split_aligned_arrays(df, features, targets, train_frac=0.70, val_frac=0.15):
    """Split by target date, preserving prior history across split boundaries.

    features[i] describes draw i and predicts targets[i+1]. Validation and
    test include every target in those splits; only the first overall draw
    lacks history. Feature construction must itself be causal.
    """
    train, val, _ = walk_forward_split(df, train_frac, val_frac)
    if len(features) != len(df) or len(targets) != len(df):
        raise ValueError("Features and targets must match the draw history length")
    a, b = len(train) - 1, len(train) + len(val) - 1
    x, y = features[:-1], targets[1:]
    return x[:a], y[:a], x[a:b], y[a:b], x[b:], y[b:]


def temporal_feature_split(df, feature_groups, train_frac=0.70, val_frac=0.15):
    """Causal features with a time scale fitted on the training split only."""
    train, _, _ = walk_forward_split(df, train_frac, val_frac)
    features = build_features(df, feature_groups,
                              max_draw_index=float(train["draw_index"].max()))
    return split_aligned_arrays(df, features, _binary_matrix(df), train_frac, val_frac)


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
    # A fixed fallback keeps this helper invariant to appending future rows.
    if max_draw_index is None:
        max_draw_index = 1.0
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

    X_train, Y_train, _, _, X_test, Y_test = temporal_feature_split(
        df, feature_groups, train_frac, val_frac
    )
    Y_train_shuffled = Y_train[rng.permutation(len(Y_train))]

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

    X_train, Y_train, _, _, X_test, Y_test = temporal_feature_split(
        df_rev, feature_groups, train_frac, val_frac
    )

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
