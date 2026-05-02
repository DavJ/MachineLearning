#!/usr/bin/env python3
"""
Sportka UBT Null-Model Validation
===================================
Verifies whether UBT/theta features capture real structure or just overfit noise.

Experiments:
  1. Real data     — full UBT pipeline on actual (or synthetic) Sportka draws
  2. Shuffled data — same draws but with temporal order destroyed (row shuffle)
  3. Random data   — purely synthetic draws (7 unique numbers from 1–49)
  4. Reversed data — chronological order reversed (future predicts past)

For each condition the same UBT model and feature set are used without any
tuning.  Bootstrap CIs (1 000 resamples) and delta comparisons are computed.

Decision rule:
  - If UBT performs similarly on real and random/shuffled/reversed data
    → conclude "No detectable structure"
  - If UBT is significantly better on real data only
    → conclude "Potential structure detected"

Usage:
    python -m sportka.experiments.run_null_test [--csv PATH] [--seed INT] [--no-cnn]

Options:
    --csv PATH   Path to Sportka CSV from sazka.cz (omit → synthetic data).
    --seed INT   Random seed (default: 42).
    --no-cnn     Skip UBTCNNv2 (faster; only UBTMLPV2 is used).

Outputs:
    reports/sportka_ubt_null_test.md
    reports/sportka_null_comparison.png   (if matplotlib available)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from copy import deepcopy
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sportka.features import (
    load_sportka_csv,
    generate_synthetic_data,
    walk_forward_split,
)
from sportka.models import RandomPredictor
from sportka.multiscale_features import build_multiscale_tensors, flatten_multiscale_tensors
from sportka.model import UBTMLPV2
from sportka.evaluation import (
    _binary_matrix,
    compute_all_metrics,
    bootstrap_ci_all_metrics,
    binary_cross_entropy,
    topk_recall,
    avg_hits,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

# Multi-scale windows — must match the v2 experiment (do NOT change)
MS_WINDOWS = [1, 16, 52]
MS_ALPHA = 0.5
MS_N = 7
MS_INCLUDE_RAW = True

METRIC_COLS = ["bce", "recall_at_6", "recall_at_10", "kl_vs_uniform", "avg_hits_6", "avg_hits_10"]

# Theoretical random baseline under uniform draw (7 numbers, top-k selection)
RANDOM_RECALL_6 = 6 / 49   # ≈ 0.1224
RANDOM_HITS_6   = 7 * 6 / 49  # ≈ 0.857


# ---------------------------------------------------------------------------
# Feature helpers (do NOT change feature engineering)
# ---------------------------------------------------------------------------

def _build_binary_matrix(df: pd.DataFrame) -> np.ndarray:
    return _binary_matrix(df)


def _build_tensors(df: pd.DataFrame) -> np.ndarray:
    bm = _build_binary_matrix(df)
    return build_multiscale_tensors(
        bm,
        windows=MS_WINDOWS,
        alpha=MS_ALPHA,
        N=MS_N,
        include_raw=MS_INCLUDE_RAW,
    )


def _align_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Temporal alignment: X[i] (features of draw i) predicts Y[i+1].
    Returns tensors (n-1, C, 7, 7) and Y (n-1, 49).
    """
    tensors = _build_tensors(df)     # (n, C, 7, 7)
    Y = _build_binary_matrix(df)    # (n, 49)
    return tensors[:-1], Y[1:]


def _flat_xy(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    X_t, Y = _align_xy(df)
    return flatten_multiscale_tensors(X_t), Y


# ---------------------------------------------------------------------------
# Data variant builders
# ---------------------------------------------------------------------------

def _shuffled_df(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Randomly permute draw rows (destroy temporal order)."""
    rng = np.random.default_rng(seed)
    df_shuf = df.iloc[rng.permutation(len(df))].reset_index(drop=True)
    df_shuf["draw_index"] = np.arange(len(df_shuf))
    return df_shuf


def _random_df(n_draws: int, seed: int) -> pd.DataFrame:
    """Generate purely random draws (same size as real dataset)."""
    return generate_synthetic_data(n_draws=n_draws, seed=seed)


def _reversed_df(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse chronological order."""
    df_rev = df.iloc[::-1].reset_index(drop=True)
    df_rev["draw_index"] = np.arange(len(df_rev))
    return df_rev


# ---------------------------------------------------------------------------
# Single-condition evaluation
# ---------------------------------------------------------------------------

def evaluate_condition(
    df: pd.DataFrame,
    model_cls,
    model_kwargs: dict,
    seed: int,
    label: str,
) -> dict:
    """
    Run walk-forward train/test on a given data variant.

    Returns a dict with keys:
        metrics, ci, test_pred, Y_test, train_sec
    """
    train_df, val_df, test_df = walk_forward_split(df, TRAIN_FRAC, VAL_FRAC)

    X_tr, Y_tr = _flat_xy(train_df)
    X_te, Y_te = _flat_xy(test_df)

    model = model_cls(**model_kwargs)

    t0 = time.time()
    model.fit(X_tr, Y_tr)
    elapsed = time.time() - t0

    test_pred = model.predict_proba(X_te)
    metrics = compute_all_metrics(Y_te, test_pred)

    print(
        f"  {label:<22} — BCE={metrics['bce']:.4f}  "
        f"recall@6={metrics['recall_at_6']:.4f}  "
        f"avg_hits_6={metrics['avg_hits_6']:.4f}  "
        f"({elapsed:.1f}s)"
    )

    ci = bootstrap_ci_all_metrics(Y_te, test_pred, n_resamples=N_BOOTSTRAP, seed=seed)

    return {
        "metrics": metrics,
        "ci": ci,
        "test_pred": test_pred,
        "Y_test": Y_te,
        "train_sec": elapsed,
        "n_train": len(X_tr),
        "n_test": len(X_te),
    }


# ---------------------------------------------------------------------------
# Delta computation
# ---------------------------------------------------------------------------

def compute_deltas(real_metrics: dict, other_metrics: dict) -> dict:
    """Compute per-metric delta: other − real (positive = other > real)."""
    return {k: other_metrics[k] - real_metrics[k] for k in METRIC_COLS}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v: float, d: int = 4) -> str:
    return f"{v:.{d}f}"


def _ci_str(tup: Tuple[float, float, float], d: int = 4) -> str:
    pt, lo, hi = tup
    return f"{pt:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def _delta_str(v: float, d: int = 4) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{d}f}"


# ---------------------------------------------------------------------------
# Decision rule
# ---------------------------------------------------------------------------

def _apply_decision(results: dict) -> Tuple[str, str]:
    """
    Apply the decision rule from the problem spec.

    Returns (verdict, explanation) where verdict is one of:
        "NO_SIGNAL"  /  "POTENTIAL_SIGNAL"
    """
    real_r6  = results["real"]["metrics"]["recall_at_6"]
    real_ci  = results["real"]["ci"]["recall_at_6"]

    conditions = {k: results[k]["metrics"]["recall_at_6"]
                  for k in ["shuffled", "random", "reversed"]}

    # Random baseline theoretical CI (used for comparison)
    rnd_pred = results["real"]["Y_test"].copy()
    # theoretical: all conditions near random

    # Check whether UBT on real data is distinguishable from random/shuffled/reversed
    # Use bootstrap CI overlap as a proxy
    real_lo, real_hi = real_ci[1], real_ci[2]

    similar_count = 0
    for cond_name, cond_r6 in conditions.items():
        cond_ci = results[cond_name]["ci"]["recall_at_6"]
        cond_lo, cond_hi = cond_ci[1], cond_ci[2]
        # CIs overlap → performance is similar
        if real_lo <= cond_hi and cond_lo <= real_hi:
            similar_count += 1

    if similar_count >= 2:
        verdict = "NO_SIGNAL"
        explanation = (
            "UBT model performance on real data is statistically indistinguishable "
            "from at least two of the three null conditions (shuffled, random, reversed). "
            "This indicates **no detectable structure** in the Sportka data."
        )
    else:
        verdict = "POTENTIAL_SIGNAL"
        explanation = (
            "UBT model performance on real data is outside the confidence intervals "
            "of most null conditions.  This is consistent with **potential structure**, "
            "but must be replicated on independent data before any strong claim can be made."
        )

    return verdict, explanation


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def _generate_plots(results: dict, plots_dir: str) -> List[str]:
    generated = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        conditions = list(results.keys())
        colors = {"real": "#2196F3", "shuffled": "#FF9800", "random": "#F44336", "reversed": "#9C27B0"}

        # --- Recall@6 bar chart with 95% CI error bars ---
        r6_vals  = [results[c]["metrics"]["recall_at_6"] for c in conditions]
        r6_ci    = [results[c]["ci"]["recall_at_6"] for c in conditions]
        r6_lower = [v - ci[1] for v, ci in zip(r6_vals, r6_ci)]
        r6_upper = [ci[2] - v for v, ci in zip(r6_vals, r6_ci)]

        bce_vals  = [results[c]["metrics"]["bce"] for c in conditions]
        hits_vals = [results[c]["metrics"]["avg_hits_6"] for c in conditions]

        fig, axes = plt.subplots(1, 3, figsize=(14, 5))
        fig.suptitle("Sportka UBT Null-Model Validation", fontsize=13)

        # Recall@6
        ax = axes[0]
        bars = ax.bar(conditions, r6_vals,
                      color=[colors[c] for c in conditions],
                      yerr=[r6_lower, r6_upper], capsize=5, alpha=0.85)
        ax.axhline(RANDOM_RECALL_6, linestyle="--", color="gray", linewidth=1.2,
                   label=f"theoretical random ({RANDOM_RECALL_6:.4f})")
        ax.set_title("Recall@6 (95% CI)")
        ax.set_ylabel("recall@6")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # BCE
        ax = axes[1]
        ax.bar(conditions, bce_vals,
               color=[colors[c] for c in conditions], alpha=0.85)
        ax.set_title("Binary Cross-Entropy")
        ax.set_ylabel("BCE")
        ax.grid(axis="y", alpha=0.3)

        # avg_hits_6
        ax = axes[2]
        ax.bar(conditions, hits_vals,
               color=[colors[c] for c in conditions], alpha=0.85)
        ax.axhline(RANDOM_HITS_6, linestyle="--", color="gray", linewidth=1.2,
                   label=f"theoretical random ({RANDOM_HITS_6:.3f})")
        ax.set_title("Avg Hits@6")
        ax.set_ylabel("avg_hits_6")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        for ax in axes:
            ax.set_xticklabels(conditions, rotation=15, ha="right")

        plt.tight_layout()
        p = os.path.join(plots_dir, "sportka_null_comparison.png")
        plt.savefig(p, dpi=100)
        plt.close()
        generated.append(p)
        print(f"  Saved: {p}")

    except Exception as exc:
        print(f"  WARNING: Could not generate plots: {exc}")

    return generated


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(
    path: str,
    data_source: str,
    n_draws: int,
    results: dict,
    deltas: dict,
    verdict: str,
    explanation: str,
    plots_generated: List[str],
) -> None:
    lines: List[str] = []
    a = lines.append

    a("# Sportka UBT Null-Model Validation")
    a("")
    a("## Overview")
    a("")
    a("This experiment tests whether UBT/theta features capture real structure")
    a("or merely overfit noise.  The same UBT pipeline is run unmodified on four")
    a("data variants:")
    a("")
    a("| Condition | Description |")
    a("|-----------|-------------|")
    a("| **real**     | Original draw order (temporal structure intact) |")
    a("| **shuffled** | Draw rows randomly permuted (temporal order destroyed) |")
    a("| **random**   | Freshly generated synthetic draws (pure noise baseline) |")
    a("| **reversed** | Chronological order reversed (future predicts past) |")
    a("")
    a("**Constraints:** No changes to feature engineering or model hyperparameters.")
    a("This is an evaluation-only experiment.")
    a("")
    a("---")
    a("")
    a("## Data")
    a("")
    a(f"| Item | Value |")
    a(f"|------|-------|")
    a(f"| Source | {data_source} |")
    a(f"| Total draws | {n_draws} |")
    a(f"| Train (70%) | {results['real']['n_train']} |")
    a(f"| Test (15%) | {results['real']['n_test']} |")
    a(f"| Split | Walk-forward (chronological) |")
    a(f"| Bootstrap resamples | {N_BOOTSTRAP} |")
    a("")
    a("---")
    a("")
    a("## Metrics per Condition — Test Set")
    a("")
    header = "| Condition | " + " | ".join(METRIC_COLS) + " |"
    sep    = "|-----------|" + "|".join(["--------"] * len(METRIC_COLS)) + "|"
    a(header)
    a(sep)
    for cond in ["real", "shuffled", "random", "reversed"]:
        m = results[cond]["metrics"]
        row = [f"**{cond}**"] + [_fmt(m[k]) for k in METRIC_COLS]
        a("| " + " | ".join(row) + " |")
    a("")
    a("**Theoretical random baseline** (7 drawn, top-6 selected):")
    a(f"  `recall@6 ≈ {RANDOM_RECALL_6:.4f}`, `avg_hits_6 ≈ {RANDOM_HITS_6:.4f}`")
    a("")
    a("---")
    a("")
    a("## Bootstrap Confidence Intervals (95%, 1 000 resamples)")
    a("")
    for cond in ["real", "shuffled", "random", "reversed"]:
        ci = results[cond]["ci"]
        a(f"### {cond}")
        a("")
        a("| Metric | Estimate [95% CI] |")
        a("|--------|-------------------|")
        for met in METRIC_COLS:
            if met in ci:
                a(f"| {met} | {_ci_str(ci[met])} |")
        a("")
    a("---")
    a("")
    a("## Delta: Other Conditions vs Real")
    a("")
    a("Positive values indicate the null condition scored *higher* than real data.")
    a("For recall@6 and avg_hits, a positive delta means the null condition appears")
    a("*better* (or equally good) as the real condition — evidence against signal.")
    a("")
    header = "| Delta | " + " | ".join(METRIC_COLS) + " |"
    sep    = "|-------|" + "|".join(["--------"] * len(METRIC_COLS)) + "|"
    a(header)
    a(sep)
    for cond in ["shuffled", "random", "reversed"]:
        d = deltas[cond]
        row = [f"Δ real→{cond}"] + [_delta_str(d[k]) for k in METRIC_COLS]
        a("| " + " | ".join(row) + " |")
    a("")
    a("---")
    a("")
    a("## Plots")
    a("")
    if plots_generated:
        for p in plots_generated:
            base = os.path.basename(p)
            a(f"![Null comparison]({base})")
            a("")
            a("*Recall@6, BCE, and Avg Hits@6 across all four conditions.*")
            a("*Error bars on Recall@6 are 95% bootstrap CIs.*")
        a("")
    else:
        a("*(Plots not generated — matplotlib may be unavailable.)*")
        a("")
    a("---")
    a("")
    a("## Decision")
    a("")
    a(f"**Verdict: {verdict}**")
    a("")
    a(explanation)
    a("")
    a("### Decision rule applied")
    a("")
    a("| Condition pair | recall@6 real | recall@6 other | CIs overlap? |")
    a("|----------------|--------------|----------------|--------------|")
    real_ci = results["real"]["ci"]["recall_at_6"]
    for cond in ["shuffled", "random", "reversed"]:
        rv = results["real"]["metrics"]["recall_at_6"]
        cv = results[cond]["metrics"]["recall_at_6"]
        c_ci = results[cond]["ci"]["recall_at_6"]
        overlap = (real_ci[1] <= c_ci[2]) and (c_ci[1] <= real_ci[2])
        a(f"| real vs {cond} | {_fmt(rv)} [{_fmt(real_ci[1])}, {_fmt(real_ci[2])}] "
          f"| {_fmt(cv)} [{_fmt(c_ci[1])}, {_fmt(c_ci[2])}] "
          f"| {'Yes ✅' if overlap else 'No ⚠️'} |")
    a("")
    a("*'Yes' means the 95% CIs of recall@6 for real and null condition overlap.*")
    a("*Overlap in ≥ 2 conditions → NO_SIGNAL verdict.*")
    a("")
    a("---")
    a("")
    a("## Interpretation")
    a("")
    a("> **Null hypothesis:** Sportka draws are uniformly random; UBT/theta features")
    a("> capture no non-random structure.")
    a("")
    a("Under the null:")
    a(f"- Expected `recall@6 ≈ {RANDOM_RECALL_6:.4f}` (theoretical)")
    a(f"- Expected `avg_hits_6 ≈ {RANDOM_HITS_6:.4f}` (theoretical)")
    a(f"- All four conditions should produce statistically identical results.")
    a("")
    a("If performance on **real** data is statistically indistinguishable from")
    a("**shuffled**, **random**, and **reversed** data, the null hypothesis cannot")
    a("be rejected and UBT features are concluded to capture **no real structure**.")
    a("")
    a("---")
    a("")
    a("*Report generated automatically by `sportka/experiments/run_null_test.py`.*")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"  Report: {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_null_test(
    csv_path: Optional[str] = None,
    seed: int = RANDOM_SEED,
) -> dict:
    print("=" * 70)
    print("  Sportka UBT Null-Model Validation")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load real (or synthetic) data
    # ------------------------------------------------------------------
    print("\n[1/5] Loading data …")
    if csv_path and os.path.exists(csv_path):
        df_real = load_sportka_csv(csv_path)
        data_source = f"CSV file: {csv_path}"
        print(f"  Loaded {len(df_real)} draws from {csv_path}")
    else:
        if csv_path:
            print(f"  WARNING: {csv_path} not found — using synthetic random data.")
        else:
            print("  No CSV provided — using synthetic random data (null hypothesis).")
        df_real = generate_synthetic_data(n_draws=800, seed=seed)
        data_source = "Synthetic random draws (null hypothesis)"
        print(f"  Generated {len(df_real)} synthetic draws.")

    n_draws = len(df_real)

    # ------------------------------------------------------------------
    # 2. Build data variants
    # ------------------------------------------------------------------
    print("\n[2/5] Building data variants …")
    df_shuffled = _shuffled_df(df_real, seed=seed)
    df_random   = _random_df(n_draws=n_draws, seed=seed + 1)
    df_reversed = _reversed_df(df_real)
    print(f"  real: {len(df_real)}, shuffled: {len(df_shuffled)}, "
          f"random: {len(df_random)}, reversed: {len(df_reversed)}")

    # ------------------------------------------------------------------
    # 3. Run experiments
    # ------------------------------------------------------------------
    print("\n[3/5] Running UBT model on all conditions …")

    model_cls = UBTMLPV2
    model_kwargs = {"hidden_layer_sizes": (128, 64), "max_iter": 300, "random_state": seed}

    results: Dict[str, dict] = {}

    for label, df_variant in [
        ("real",     df_real),
        ("shuffled", df_shuffled),
        ("random",   df_random),
        ("reversed", df_reversed),
    ]:
        results[label] = evaluate_condition(
            df_variant, model_cls, model_kwargs, seed=seed, label=label
        )

    # ------------------------------------------------------------------
    # 4. Compute deltas and decision
    # ------------------------------------------------------------------
    print("\n[4/5] Computing deltas and decision …")
    deltas = {
        cond: compute_deltas(results["real"]["metrics"], results[cond]["metrics"])
        for cond in ["shuffled", "random", "reversed"]
    }

    verdict, explanation = _apply_decision(results)
    print(f"\n  Verdict: {verdict}")
    print(f"  {explanation}")

    # ------------------------------------------------------------------
    # 5. Generate plots and report
    # ------------------------------------------------------------------
    print("\n[5/5] Generating plots and report …")
    plots_dir = os.path.join(_REPO_ROOT, "reports")
    os.makedirs(plots_dir, exist_ok=True)
    plots_generated = _generate_plots(results, plots_dir)

    report_path = os.path.join(plots_dir, "sportka_ubt_null_test.md")
    _write_report(
        report_path,
        data_source=data_source,
        n_draws=n_draws,
        results=results,
        deltas=deltas,
        verdict=verdict,
        explanation=explanation,
        plots_generated=plots_generated,
    )

    print("\n✓ Null-model validation complete.\n")
    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sportka UBT Null-Model Validation")
    parser.add_argument("--csv", default=None, help="Path to Sportka CSV file")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()
    run_null_test(csv_path=args.csv, seed=args.seed)
