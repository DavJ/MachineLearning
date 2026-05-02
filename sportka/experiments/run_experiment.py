#!/usr/bin/env python3
"""
Sportka UBT/Theta Experiment v1
================================
End-to-end experiment runner.

Usage:
    python -m sportka.experiments.run_experiment [--csv PATH] [--seed INT]
    python sportka/experiments/run_experiment.py [--csv PATH] [--seed INT]

If --csv is omitted the script generates synthetic (purely random) draws so
the experiment can run out-of-the-box without network access.  Real draws can
be downloaded from sazka.cz and passed via --csv.

Outputs:
    reports/sportka_ubt_experiment.md
    reports/sportka_perf_over_time.png  (if matplotlib available)
    reports/sportka_dist_divergence.png (if matplotlib available)
"""

from __future__ import annotations

import argparse
import os
import sys
import json
import time
import warnings
from copy import deepcopy
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# Suppress sklearn convergence warnings to keep output clean
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Add repo root to path so `sportka` package is importable when run directly
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from sportka.features import (
    build_features,
    load_sportka_csv,
    generate_synthetic_data,
    walk_forward_split,
    FEATURE_GROUPS,
)
from sportka.models import (
    RandomPredictor,
    GlobalFreqPredictor,
    RollingFreqPredictor,
    LogisticRegressionModel,
    MLPModel,
    UBTModel,
)
from sportka.evaluation import (
    _binary_matrix,
    _align_xy,
    compute_all_metrics,
    walk_forward_eval,
    bootstrap_ci_all_metrics,
    shuffled_labels_test,
    reversed_time_test,
    rolling_metrics,
    topk_recall,
    binary_cross_entropy,
    avg_hits,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
N_BOOTSTRAP = 1000
RANDOM_SEED = 42

FEATURE_CONFIGS = {
    "base_only":       ["base"],
    "base_time":       ["base", "time"],
    "base_winding":    ["base", "time", "winding"],
    "ubt_full":        ["base", "time", "winding", "torus", "theta"],
}

MODELS = {
    "random_uniform":    RandomPredictor(),
    "global_frequency":  GlobalFreqPredictor(),
    "rolling_frequency": RollingFreqPredictor(),
    "logistic_reg":      LogisticRegressionModel(C=0.1, max_iter=200),
    "mlp_small":         MLPModel(hidden_layer_sizes=(64,), max_iter=200),
    "ubt_mlp":           UBTModel(hidden_layer_sizes=(128, 64), max_iter=300),
}

# Which feature config each model uses
MODEL_FEATURES = {
    "random_uniform":    "base_only",
    "global_frequency":  "base_only",
    "rolling_frequency": "base_winding",
    "logistic_reg":      "base_time",
    "mlp_small":         "base_winding",
    "ubt_mlp":           "ubt_full",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt(v: float, decimals: int = 4) -> str:
    return f"{v:.{decimals}f}"


def _ci_str(ci_tuple: Tuple[float, float, float], decimals: int = 4) -> str:
    pt, lo, hi = ci_tuple
    return f"{pt:.{decimals}f} [{lo:.{decimals}f}, {hi:.{decimals}f}]"


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(csv_path: str | None = None, seed: int = RANDOM_SEED) -> dict:
    print("=" * 70)
    print("  Sportka UBT/Theta Experiment v1")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print("\n[1/6] Loading data …")
    if csv_path and os.path.exists(csv_path):
        df = load_sportka_csv(csv_path)
        data_source = f"CSV file: {csv_path}"
        print(f"  Loaded {len(df)} draws from {csv_path}")
    else:
        if csv_path:
            print(f"  WARNING: {csv_path} not found — using synthetic random data.")
        else:
            print("  No CSV provided — using synthetic random data (null hypothesis).")
        df = generate_synthetic_data(n_draws=800, seed=seed)
        data_source = "Synthetic random draws (null hypothesis)"
        print(f"  Generated {len(df)} synthetic draws.")

    train_df, val_df, test_df = walk_forward_split(df, TRAIN_FRAC, VAL_FRAC)
    print(f"  Split: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

    # ------------------------------------------------------------------
    # 2. Build feature matrices for each config
    # ------------------------------------------------------------------
    print("\n[2/6] Building feature matrices …")
    feat_matrices: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for cfg_name, groups in FEATURE_CONFIGS.items():
        X_tr, Y_tr = _align_xy(train_df, groups)
        X_vl, Y_vl = _align_xy(val_df,   groups)
        X_te, Y_te = _align_xy(test_df,  groups)
        feat_matrices[cfg_name] = (X_tr, Y_tr, X_vl, Y_vl, X_te, Y_te)
        print(f"  {cfg_name:<20} → {X_tr.shape[1]} features  ({X_tr.shape[0]} train pairs)")

    # Canonical Y arrays (from the ubt_full config; all configs share the same Y)
    _, Y_train, _, Y_val, _, Y_test = feat_matrices["ubt_full"]

    # ------------------------------------------------------------------
    # 3. Train and evaluate all models
    # ------------------------------------------------------------------
    print("\n[3/6] Training & evaluating models …")
    results: Dict[str, Dict] = {}

    for model_name, model in MODELS.items():
        cfg_name = MODEL_FEATURES[model_name]
        X_tr, Y_tr, X_vl, Y_vl, X_te, Y_te = feat_matrices[cfg_name]

        t0 = time.time()
        model_clone = deepcopy(model)
        model_clone.fit(X_tr, Y_tr)
        elapsed = time.time() - t0

        val_pred  = model_clone.predict_proba(X_vl)
        test_pred = model_clone.predict_proba(X_te)

        val_m  = compute_all_metrics(Y_vl,  val_pred)
        test_m = compute_all_metrics(Y_te, test_pred)

        # Bootstrap CI on test set
        print(f"  {model_name:<20} ({cfg_name}) — bootstrapping CI …")
        ci = bootstrap_ci_all_metrics(Y_te, test_pred, n_resamples=N_BOOTSTRAP, seed=seed)

        results[model_name] = {
            "val":      val_m,
            "test":     test_m,
            "ci":       ci,
            "features": cfg_name,
            "n_features": X_tr.shape[1],
            "train_sec": elapsed,
            "test_pred": test_pred,   # kept for plots
            "Y_test":   Y_te,         # kept for plots
        }

        print(
            f"    BCE={_fmt(test_m['bce'])}  "
            f"recall@6={_fmt(test_m['recall_at_6'])}  "
            f"avg_hits_6={_fmt(test_m['avg_hits_6'])}  "
            f"({elapsed:.1f}s)"
        )

    # ------------------------------------------------------------------
    # 4. Control tests (UBT model only)
    # ------------------------------------------------------------------
    print("\n[4/6] Running control tests (ubt_mlp) …")
    ubt_model = MODELS["ubt_mlp"]
    ubt_cfg   = FEATURE_CONFIGS["ubt_full"]

    shuffle_m = shuffled_labels_test(deepcopy(ubt_model), df, ubt_cfg, seed=seed)
    revtime_m = reversed_time_test(deepcopy(ubt_model), df, ubt_cfg)
    print(f"  Shuffled labels — BCE={_fmt(shuffle_m['bce'])}  recall@6={_fmt(shuffle_m['recall_at_6'])}")
    print(f"  Reversed time   — BCE={_fmt(revtime_m['bce'])}  recall@6={_fmt(revtime_m['recall_at_6'])}")

    # ------------------------------------------------------------------
    # 5. Generate plots (best-effort)
    # ------------------------------------------------------------------
    print("\n[5/6] Generating plots …")
    plots_dir = os.path.join(_REPO_ROOT, "reports")
    os.makedirs(plots_dir, exist_ok=True)

    perf_plot_path = os.path.join(plots_dir, "sportka_perf_over_time.png")
    dist_plot_path = os.path.join(plots_dir, "sportka_dist_divergence.png")
    plots_generated = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # --- Performance over time (rolling recall@6) ---
        fig, ax = plt.subplots(figsize=(12, 5))
        for mname in ["random_uniform", "rolling_frequency", "ubt_mlp"]:
            tp = results[mname]["test_pred"]
            yt = results[mname]["Y_test"]
            rm = rolling_metrics(yt, tp, window=30)
            ax.plot(rm.index, rm["recall_at_6"], label=mname)
        ax.set_xlabel("Draw (test set)")
        ax.set_ylabel("Recall@6 (rolling 30-draw window)")
        ax.set_title("Performance over time — rolling Recall@6")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(perf_plot_path, dpi=100)
        plt.close()
        plots_generated.append(perf_plot_path)
        print(f"  Saved: {perf_plot_path}")

        # --- Distribution divergence (KL vs uniform) ---
        model_names = list(results.keys())
        kl_values   = [results[m]["test"]["kl_vs_uniform"] for m in model_names]
        fig, ax = plt.subplots(figsize=(10, 4))
        bars = ax.bar(model_names, kl_values)
        ax.set_ylabel("KL(uniform || predicted)")
        ax.set_title("KL Divergence from Uniform Distribution (test set)")
        ax.set_xticklabels(model_names, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(dist_plot_path, dpi=100)
        plt.close()
        plots_generated.append(dist_plot_path)
        print(f"  Saved: {dist_plot_path}")

    except Exception as exc:
        print(f"  WARNING: Could not generate plots: {exc}")

    # ------------------------------------------------------------------
    # 6. Write report
    # ------------------------------------------------------------------
    print("\n[6/6] Writing report …")
    report_path = os.path.join(plots_dir, "sportka_ubt_experiment.md")
    _write_report(
        report_path,
        data_source=data_source,
        n_draws=len(df),
        train_n=len(train_df),
        val_n=len(val_df),
        test_n=len(test_df),
        results=results,
        shuffle_m=shuffle_m,
        revtime_m=revtime_m,
        plots_generated=plots_generated,
    )
    print(f"  Report: {report_path}")

    print("\n✓ Experiment complete.\n")
    return results


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def _write_report(
    path: str,
    data_source: str,
    n_draws: int,
    train_n: int,
    val_n: int,
    test_n: int,
    results: dict,
    shuffle_m: dict,
    revtime_m: dict,
    plots_generated: list,
) -> None:
    lines = []
    a = lines.append

    a("# Sportka UBT/Theta Experiment v1")
    a("")
    a("## Overview")
    a("")
    a("This report presents a rigorous, null-hypothesis-first evaluation of whether")
    a("UBT/theta-inspired features capture any non-random structure in Sportka draw")
    a("history.  All models are compared against strong random baselines.")
    a("")
    a("**Guiding principles:**")
    a("- Sportka draws are treated as a null-hypothesis random system.")
    a("- Statistical validation is the primary goal; number prediction is not.")
    a("- Overfitting is avoided through strict walk-forward (chronological) splitting.")
    a("- All results are compared to a uniform-random baseline.")
    a("")
    a("---")
    a("")
    a("## Data")
    a("")
    a(f"| Item | Value |")
    a(f"|------|-------|")
    a(f"| Source | {data_source} |")
    a(f"| Total draws | {n_draws} |")
    a(f"| Train (first 70%) | {train_n} |")
    a(f"| Validation (next 15%) | {val_n} |")
    a(f"| Test (last 15%) | {test_n} |")
    a(f"| Split method | Walk-forward (chronological) |")
    a("")
    a("---")
    a("")
    a("## Feature Groups")
    a("")
    a("| Group | Dimensions | Description |")
    a("|-------|-----------|-------------|")
    a("| `base` | 49 | Binary indicator vector for each draw |")
    a("| `time` | 13 | Normalised draw index + cyclical sin/cos encodings |")
    a("| `winding` | 294 | Rolling & exponential-decay frequency vectors |")
    a("| `torus` | 112 | 7×7 grid: row/col sums, toroidal neighbour counts, 3×3 conv |")
    a("| `theta` (exp.) | 98 | Truncated Jacobi-theta projection (experimental) |")
    a("")
    a("---")
    a("")
    a("## Model Comparison — Test Set")
    a("")
    # Build comparison table
    metric_cols = ["bce", "recall_at_6", "recall_at_10", "kl_vs_uniform", "avg_hits_6"]
    header = "| Model | Features | n_feat | " + " | ".join(metric_cols) + " |"
    sep    = "|-------|----------|--------|" + "|".join(["--------"] * len(metric_cols)) + "|"
    a(header)
    a(sep)
    for mname, res in results.items():
        m = res["test"]
        row_parts = [
            mname,
            res["features"],
            str(res["n_features"]),
        ] + [_fmt(m[k]) for k in metric_cols]
        a("| " + " | ".join(row_parts) + " |")
    a("")
    a("**Metrics:**")
    a("- `bce`: binary cross-entropy (lower = better)")
    a("- `recall_at_6`: fraction of drawn numbers recovered in top-6 predictions (higher = better)")
    a("- `recall_at_10`: same for top-10")
    a("- `kl_vs_uniform`: KL divergence of predictions from uniform distribution")
    a("- `avg_hits_6`: average number of correct hits when selecting top 6")
    a("")
    a("---")
    a("")
    a("## Bootstrap Confidence Intervals (95%, 1000 resamples)")
    a("")
    a("Test-set metrics for selected models with 95% bootstrap CIs:")
    a("")
    for mname in ["random_uniform", "rolling_frequency", "ubt_mlp"]:
        if mname not in results:
            continue
        ci = results[mname]["ci"]
        a(f"### {mname}")
        a("")
        a("| Metric | Estimate [95% CI] |")
        a("|--------|-------------------|")
        for met, tup in ci.items():
            a(f"| {met} | {_ci_str(tup)} |")
        a("")

    a("---")
    a("")
    a("## Control Tests (UBT model — `ubt_mlp`)")
    a("")
    a("These tests verify whether the model learns genuine signal or overfits noise.")
    a("")
    a("### Shuffled-labels test")
    a("Training labels are permuted randomly, destroying any signal.")
    a("If the model truly learns, it should perform no better than random here.")
    a("")
    a("| Metric | Value |")
    a("|--------|-------|")
    for k, v in shuffle_m.items():
        a(f"| {k} | {_fmt(v)} |")
    a("")
    a("### Reversed-time test")
    a("Data is reversed chronologically (future predicts past).")
    a("A causal model should perform worse; similar performance implies no temporal signal.")
    a("")
    a("| Metric | Value |")
    a("|--------|-------|")
    for k, v in revtime_m.items():
        a(f"| {k} | {_fmt(v)} |")
    a("")
    a("---")
    a("")
    a("## Plots")
    a("")
    if plots_generated:
        for p in plots_generated:
            basename = os.path.basename(p)
            if "perf" in basename:
                a(f"![Performance over time]({basename})")
                a("")
                a("*Rolling Recall@6 over the test set (30-draw window).*")
            elif "dist" in basename:
                a(f"![Distribution divergence]({basename})")
                a("")
                a("*KL divergence from uniform distribution per model.*")
        a("")
    else:
        a("*(Plots not generated — matplotlib may be unavailable.)*")
        a("")

    a("---")
    a("")
    a("## Statistical Interpretation")
    a("")
    a("> **Null hypothesis:** Sportka draws are uniformly random; no feature captures")
    a("> non-random structure.")
    a("")
    a("**How to read the results:**")
    a("- If `recall_at_6` for UBT/ML models is *within* the bootstrap CI of the random")
    a("  baseline, we cannot reject the null hypothesis.")
    a("- Under the null (uniform random) with 7 drawn numbers and top-6 predictions:")
    a("  `E[hits] = 7 × 6/49 = 42/49 ≈ 0.8571`, so")
    a("  `recall@6 = E[hits] / 7 = 6/49 ≈ 0.1224`.")
    a("")
    ubt_test = results.get("ubt_mlp", {}).get("test", {})
    rnd_test = results.get("random_uniform", {}).get("test", {})
    ubt_r6   = ubt_test.get("recall_at_6", float("nan"))
    rnd_r6   = rnd_test.get("recall_at_6", float("nan"))
    ubt_ci   = results.get("ubt_mlp", {}).get("ci", {}).get("recall_at_6", (float("nan"),)*3)
    rnd_ci   = results.get("random_uniform", {}).get("ci", {}).get("recall_at_6", (float("nan"),)*3)

    a(f"- Random baseline recall@6: **{_fmt(rnd_r6)}** "
      f"(95% CI [{_fmt(rnd_ci[1])}, {_fmt(rnd_ci[2])}])")
    a(f"- UBT model recall@6:       **{_fmt(ubt_r6)}** "
      f"(95% CI [{_fmt(ubt_ci[1])}, {_fmt(ubt_ci[2])}])")

    significant = (
        ubt_r6 > rnd_ci[2] or ubt_r6 < rnd_ci[1]
    )
    if significant:
        a("")
        a("⚠️  The UBT model's performance is **outside** the random baseline CI.")
        a("    This is a *statistically unusual* result and warrants further")
        a("    investigation with independent data before any claims are made.")
    else:
        a("")
        a("✅  The UBT model's performance is **within** the random baseline CI.")
        a("    We **cannot** reject the null hypothesis.  No statistically significant")
        a("    predictive power has been demonstrated.")
    a("")
    a("**Conclusion:** Results must be replicated on independent real draw data before")
    a("any claim of predictive power can be made.")
    a("")
    a("---")
    a("")
    a("## Constraints and Caveats")
    a("")
    a("- No predictive power is claimed without statistical significance.")
    a("- Theta features are marked **experimental** and lack theoretical justification")
    a("  beyond exploratory interest.")
    a("- Torus embedding is a deterministic projection that may encode number")
    a("  proximity artefacts without capturing true lottery structure.")
    a("- All models use scikit-learn and are intentionally small to avoid overfitting.")
    a("- If run on synthetic data, all results trivially confirm the null hypothesis.")
    a("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sportka UBT/Theta Experiment v1")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to Sportka CSV file (sazka.cz export). "
             "Omit to run on synthetic random data.",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()
    run_experiment(csv_path=args.csv, seed=args.seed)


if __name__ == "__main__":
    main()
