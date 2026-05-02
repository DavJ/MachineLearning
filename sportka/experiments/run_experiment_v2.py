#!/usr/bin/env python3
"""
Sportka UBT/Theta Experiment v2
================================
End-to-end walk-forward experiment using the UBT-native theta pipeline.

What is new vs v1:
  - Feature extraction via toroidal embedding + theta transform on 7×7 grids.
  - Multi-scale temporal context (windows: 1, 16, 52 draws).
  - Two new models: UBTMLPV2 (shallow MLP) and UBTCNNv2 (2-layer toroidal CNN).
  - All v1 baselines retained for direct comparison.

Usage:
    python -m sportka.experiments.run_experiment_v2 [--csv PATH] [--seed INT]
    python sportka/experiments/run_experiment_v2.py

Options:
    --csv PATH   Path to Sportka CSV from sazka.cz (omit → synthetic data).
    --seed INT   Random seed (default: 42).
    --no-cnn     Skip UBTCNNv2 (faster run, useful for quick checks).

Outputs:
    reports/sportka_ubt_experiment_v2.md
    reports/sportka_v2_perf_over_time.png   (if matplotlib available)
    reports/sportka_v2_dist_divergence.png  (if matplotlib available)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from copy import deepcopy
from typing import Dict, List, Tuple

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
from sportka.models import (
    RandomPredictor,
    GlobalFreqPredictor,
    RollingFreqPredictor,
    UBTModel,
)
from sportka.multiscale_features import build_multiscale_tensors, flatten_multiscale_tensors
from sportka.model import UBTMLPV2, UBTCNNv2
from sportka.evaluation import (
    _binary_matrix,
    compute_all_metrics,
    bootstrap_ci_all_metrics,
    shuffled_labels_test,
    reversed_time_test,
    rolling_metrics,
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

# Multi-scale windows (short, medium, long — from spec)
MS_WINDOWS = [1, 16, 52]
MS_ALPHA = 0.5
MS_N = 7
MS_INCLUDE_RAW = True


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _build_binary_matrix(df: pd.DataFrame) -> np.ndarray:
    """Convert df['numbers'] → (T, 49) binary float32 matrix."""
    return _binary_matrix(df)


def _build_v2_features(df: pd.DataFrame) -> np.ndarray:
    """
    Build (T, C, 7, 7) multi-scale theta tensors from a draw DataFrame.
    """
    bm = _build_binary_matrix(df)
    return build_multiscale_tensors(
        bm,
        windows=MS_WINDOWS,
        alpha=MS_ALPHA,
        N=MS_N,
        include_raw=MS_INCLUDE_RAW,
    )


def _align_v2_xy(df: pd.DataFrame):
    """
    Temporal alignment: X[i] (features of draw i) predicts Y[i+1].

    Returns:
        X_tensors: (n-1, C, 7, 7)
        Y:         (n-1, 49)
    """
    tensors = _build_v2_features(df)   # (n, C, 7, 7)
    Y = _build_binary_matrix(df)       # (n, 49)
    return tensors[:-1], Y[1:]


def _align_flat_xy(df: pd.DataFrame):
    """Flat version of _align_v2_xy: (n-1, C*49), (n-1, 49)."""
    X_tensors, Y = _align_v2_xy(df)
    return flatten_multiscale_tensors(X_tensors), Y


def _align_random_xy(df: pd.DataFrame):
    """Minimal X for baseline models: (n-1, 49) binary, (n-1, 49) Y."""
    bm = _build_binary_matrix(df)
    return bm[:-1], bm[1:]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt(v: float, d: int = 4) -> str:
    return f"{v:.{d}f}"


def _ci_str(tup: Tuple[float, float, float], d: int = 4) -> str:
    pt, lo, hi = tup
    return f"{pt:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------

def run_experiment(
    csv_path: str | None = None,
    seed: int = RANDOM_SEED,
    include_cnn: bool = True,
) -> dict:
    print("=" * 70)
    print("  Sportka UBT/Theta Experiment v2")
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
    # 2. Build feature matrices
    # ------------------------------------------------------------------
    print("\n[2/6] Building v2 multi-scale theta features …")
    n_channels = len(MS_WINDOWS) + (1 if MS_INCLUDE_RAW else 0)
    print(
        f"  windows={MS_WINDOWS}, alpha={MS_ALPHA}, N={MS_N}, "
        f"include_raw={MS_INCLUDE_RAW}  →  {n_channels} channels × 49 = {n_channels * 49} features"
    )

    X_tr_t, Y_tr = _align_v2_xy(train_df)
    X_vl_t, Y_vl = _align_v2_xy(val_df)
    X_te_t, Y_te = _align_v2_xy(test_df)

    X_tr_f = flatten_multiscale_tensors(X_tr_t)
    X_vl_f = flatten_multiscale_tensors(X_vl_t)
    X_te_f = flatten_multiscale_tensors(X_te_t)

    # Baseline models use the same simple binary matrix
    X_tr_raw, _ = _align_random_xy(train_df)
    X_vl_raw, _ = _align_random_xy(val_df)
    X_te_raw, _ = _align_random_xy(test_df)

    print(f"  Tensor shape: {X_tr_t.shape}  →  flat: {X_tr_f.shape}")

    # ------------------------------------------------------------------
    # 3. Define models
    # ------------------------------------------------------------------
    models_cfg = {
        "random_uniform": {
            "model": RandomPredictor(),
            "X_tr": X_tr_raw, "X_vl": X_vl_raw, "X_te": X_te_raw,
        },
        "global_frequency": {
            "model": GlobalFreqPredictor(),
            "X_tr": X_tr_raw, "X_vl": X_vl_raw, "X_te": X_te_raw,
        },
        "rolling_frequency": {
            "model": RollingFreqPredictor(rolling_freq_feature_index=0),
            "X_tr": X_tr_raw, "X_vl": X_vl_raw, "X_te": X_te_raw,
        },
        "ubt_mlp_v1": {
            "model": UBTModel(hidden_layer_sizes=(128, 64), max_iter=300),
            "X_tr": X_tr_f, "X_vl": X_vl_f, "X_te": X_te_f,
        },
        "ubt_mlp_v2": {
            "model": UBTMLPV2(hidden_layer_sizes=(128, 64), max_iter=300),
            "X_tr": X_tr_t, "X_vl": X_vl_t, "X_te": X_te_t,
        },
    }
    if include_cnn:
        models_cfg["ubt_cnn_v2"] = {
            "model": UBTCNNv2(
                n_filters_1=16, n_filters_2=8,
                hidden_mlp=64, max_iter=300, random_state=seed,
            ),
            "X_tr": X_tr_t, "X_vl": X_vl_t, "X_te": X_te_t,
        }

    # ------------------------------------------------------------------
    # 4. Train and evaluate
    # ------------------------------------------------------------------
    print("\n[3/6] Training & evaluating models …")
    results: Dict[str, dict] = {}

    for mname, cfg in models_cfg.items():
        model = deepcopy(cfg["model"])
        Xtr, Xvl, Xte = cfg["X_tr"], cfg["X_vl"], cfg["X_te"]

        t0 = time.time()
        fitted = model.fit(Xtr, Y_tr)

        if fitted is None:
            print(f"  {mname:<20} — skipped (sklearn not available)")
            continue

        elapsed = time.time() - t0

        val_pred = model.predict_proba(Xvl)
        test_pred = model.predict_proba(Xte)

        val_m = compute_all_metrics(Y_vl, val_pred)
        test_m = compute_all_metrics(Y_te, test_pred)

        print(f"  {mname:<20} — bootstrapping CI …")
        ci = bootstrap_ci_all_metrics(Y_te, test_pred, n_resamples=N_BOOTSTRAP, seed=seed)

        results[mname] = {
            "val": val_m,
            "test": test_m,
            "ci": ci,
            "train_sec": elapsed,
            "test_pred": test_pred,
            "Y_test": Y_te,
        }

        print(
            f"    BCE={_fmt(test_m['bce'])}  "
            f"recall@6={_fmt(test_m['recall_at_6'])}  "
            f"avg_hits_6={_fmt(test_m['avg_hits_6'])}  "
            f"({elapsed:.1f}s)"
        )

    # ------------------------------------------------------------------
    # 5. Control tests (ubt_mlp_v2)
    # ------------------------------------------------------------------
    print("\n[4/6] Running control tests (ubt_mlp_v2) …")

    def _shuffled_test(model, seed_):
        """Shuffle training labels and evaluate on test set."""
        rng = np.random.default_rng(seed_)
        m = deepcopy(model)
        Y_shuf = Y_tr[rng.permutation(len(Y_tr))]
        m.fit(X_tr_t, Y_shuf)
        return compute_all_metrics(Y_te, m.predict_proba(X_te_t))

    def _reversed_test(model_):
        """Reverse time order and evaluate."""
        df_rev = df.iloc[::-1].reset_index(drop=True)
        df_rev["draw_index"] = np.arange(len(df_rev))
        tr_r, vl_r, te_r = walk_forward_split(df_rev, TRAIN_FRAC, VAL_FRAC)
        Xtr_r, Ytr_r = _align_v2_xy(tr_r)
        Xte_r, Yte_r = _align_v2_xy(te_r)
        m = deepcopy(model_)
        m.fit(Xtr_r, Ytr_r)
        return compute_all_metrics(Yte_r, m.predict_proba(Xte_r))

    v2_model = models_cfg["ubt_mlp_v2"]["model"]
    shuffle_m = _shuffled_test(v2_model, seed)
    revtime_m = _reversed_test(v2_model)
    print(
        f"  Shuffled labels — BCE={_fmt(shuffle_m['bce'])}  "
        f"recall@6={_fmt(shuffle_m['recall_at_6'])}"
    )
    print(
        f"  Reversed time   — BCE={_fmt(revtime_m['bce'])}  "
        f"recall@6={_fmt(revtime_m['recall_at_6'])}"
    )

    # ------------------------------------------------------------------
    # 6. Generate plots (best-effort)
    # ------------------------------------------------------------------
    print("\n[5/6] Generating plots …")
    plots_dir = os.path.join(_REPO_ROOT, "reports")
    os.makedirs(plots_dir, exist_ok=True)
    plots_generated = []

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # --- Rolling recall@6 over time ---
        fig, ax = plt.subplots(figsize=(12, 5))
        for mname in ["random_uniform", "rolling_frequency", "ubt_mlp_v2"]:
            if mname not in results:
                continue
            rm = rolling_metrics(results[mname]["Y_test"], results[mname]["test_pred"], window=30)
            ax.plot(rm.index, rm["recall_at_6"], label=mname)
        ax.set_xlabel("Draw (test set)")
        ax.set_ylabel("Recall@6 (rolling 30-draw window)")
        ax.set_title("v2 Pipeline — Performance over time (rolling Recall@6)")
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        p = os.path.join(plots_dir, "sportka_v2_perf_over_time.png")
        plt.savefig(p, dpi=100)
        plt.close()
        plots_generated.append(p)
        print(f"  Saved: {p}")

        # --- KL divergence bar chart ---
        mnames = list(results.keys())
        kl_vals = [results[m]["test"]["kl_vs_uniform"] for m in mnames]
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.bar(mnames, kl_vals)
        ax.set_ylabel("KL(uniform || predicted)")
        ax.set_title("v2 Pipeline — KL Divergence from Uniform (test set)")
        ax.set_xticklabels(mnames, rotation=20, ha="right")
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        p = os.path.join(plots_dir, "sportka_v2_dist_divergence.png")
        plt.savefig(p, dpi=100)
        plt.close()
        plots_generated.append(p)
        print(f"  Saved: {p}")

    except Exception as exc:
        print(f"  WARNING: Could not generate plots: {exc}")

    # ------------------------------------------------------------------
    # 7. Write report
    # ------------------------------------------------------------------
    print("\n[6/6] Writing report …")
    report_path = os.path.join(plots_dir, "report_v2.md")
    _write_report(
        report_path,
        data_source=data_source,
        n_draws=len(df),
        train_n=len(train_df),
        val_n=len(val_df),
        test_n=len(test_df),
        n_channels=n_channels,
        results=results,
        shuffle_m=shuffle_m,
        revtime_m=revtime_m,
        plots_generated=plots_generated,
    )
    print(f"  Report: {report_path}")

    print("\n✓ Experiment v2 complete.\n")
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
    n_channels: int,
    results: dict,
    shuffle_m: dict,
    revtime_m: dict,
    plots_generated: list,
) -> None:
    lines = []
    a = lines.append

    a("# Sportka UBT/Theta Experiment v2")
    a("")
    a("## Overview")
    a("")
    a("Version 2 introduces a UBT-native prediction pipeline built on three")
    a("interacting components:")
    a("")
    a("1. **Toroidal embedding** — maps each 49-dim binary draw vector onto a")
    a("   7×7 grid with periodic (torus) boundary conditions.")
    a("2. **Theta transform** — applies a truncated Jacobi-theta-like spectral")
    a("   transform element-wise over the 7×7 grid.")
    a("3. **Multi-scale temporal dynamics** — computes rolling-average grids")
    a("   over three time scales (short=1, medium=16, long=52 draws) and stacks")
    a("   the theta-transformed feature maps as separate channels.")
    a("")
    a("Two learning models are evaluated:")
    a("- `ubt_mlp_v2`: shallow 2-layer MLP on flattened (C×49) features.")
    a("- `ubt_cnn_v2` *(optional)*: 2-layer toroidal CNN + MLP head.")
    a("")
    a("All models are compared to v1 baselines under strict walk-forward")
    a("(chronological) evaluation.  The null hypothesis — that draws are")
    a("uniformly random — is the primary reference point.")
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
    a("## v2 Pipeline Details")
    a("")
    a("### Step 1 — Torus Embedding")
    a("")
    a("Each draw's 49 numbers are mapped onto a 7×7 binary grid:")
    a("")
    a("```")
    a("  row = (number - 1) // 7")
    a("  col = (number - 1) % 7")
    a("```")
    a("")
    a("Toroidal (periodic) boundary conditions are enforced via `np.roll`.")
    a("")
    a("### Step 2 — Theta Transform")
    a("")
    a("Applied element-wise to each 7×7 grid value v ∈ [0, 1]:")
    a("")
    a("```")
    a("  θ(v; α, N) = Σ_{n=0}^{N} exp(-α · n²) · cos(2π · n · v)")
    a("```")
    a("")
    a(f"Hyperparameters: **N={MS_N}**, **α={MS_ALPHA}**.")
    a("")
    a("### Step 3 — Multi-scale History")
    a("")
    a(f"| Channel | Window | Description |")
    a(f"|---------|--------|-------------|")
    for i, w in enumerate(MS_WINDOWS):
        label = {1: "short", 16: "medium", 52: "long"}.get(w, str(w))
        a(f"| ch {i} | {w} draws | θ(rolling avg grid over last {w} draw(s)) |")
    if MS_INCLUDE_RAW:
        a(f"| ch {len(MS_WINDOWS)} | 1 (current) | Raw binary grid (no transform) |")
    a("")
    a(f"**Total channels: {n_channels}  →  feature dimension: {n_channels} × 49 = {n_channels * 49}**")
    a("")
    a("---")
    a("")
    a("## Model Comparison — Test Set")
    a("")
    metric_cols = ["bce", "recall_at_6", "recall_at_10", "kl_vs_uniform", "avg_hits_6"]
    header = "| Model | " + " | ".join(metric_cols) + " | Train(s) |"
    sep    = "|-------|" + "|".join(["--------"] * len(metric_cols)) + "|--------|"
    a(header)
    a(sep)
    for mname, res in results.items():
        m = res["test"]
        row = [mname] + [_fmt(m[k]) for k in metric_cols] + [_fmt(res["train_sec"], 1)]
        a("| " + " | ".join(row) + " |")
    a("")
    a("**Metrics:**")
    a("- `bce`: binary cross-entropy (lower = better)")
    a("- `recall_at_6`: fraction of drawn numbers recovered in top-6 predictions (higher = better)")
    a("- `recall_at_10`: same for top-10")
    a("- `kl_vs_uniform`: KL divergence of predictions from uniform distribution")
    a("- `avg_hits_6`: average correct hits when selecting top-6")
    a("")
    a("---")
    a("")
    a("## Bootstrap Confidence Intervals (95%, 1000 resamples)")
    a("")
    for mname in ["random_uniform", "rolling_frequency", "ubt_mlp_v2"]:
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
    a("## Control Tests (`ubt_mlp_v2`)")
    a("")
    a("### Shuffled-labels test")
    a("Training labels permuted randomly.  A model that truly learns signal should")
    a("perform no better than random on this control.")
    a("")
    a("| Metric | Value |")
    a("|--------|-------|")
    for k, v in shuffle_m.items():
        a(f"| {k} | {_fmt(v)} |")
    a("")
    a("### Reversed-time test")
    a("Data reversed chronologically (future predicts past).")
    a("Similar performance to the forward direction implies no genuine temporal signal.")
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
            base = os.path.basename(p)
            if "perf" in base:
                a(f"![Performance over time]({base})")
                a("")
                a("*Rolling Recall@6 over the test set (30-draw window).*")
            elif "dist" in base:
                a(f"![Distribution divergence]({base})")
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
    a("- Under the null (uniform random, 7 drawn, top-6 selected):")
    a("  `E[hits] = 7 × 6/49 ≈ 0.857`, so `recall@6 = 6/49 ≈ 0.1224`.")
    a("- If the v2 model recall@6 falls within the random baseline CI, the null")
    a("  hypothesis cannot be rejected.")
    a("")

    rnd_r6 = results.get("random_uniform", {}).get("test", {}).get("recall_at_6", float("nan"))
    v2_r6  = results.get("ubt_mlp_v2",    {}).get("test", {}).get("recall_at_6", float("nan"))
    rnd_ci = results.get("random_uniform", {}).get("ci",   {}).get("recall_at_6", (float("nan"),)*3)
    v2_ci  = results.get("ubt_mlp_v2",    {}).get("ci",   {}).get("recall_at_6", (float("nan"),)*3)

    a(f"- Random baseline recall@6: **{_fmt(rnd_r6)}** "
      f"(95% CI [{_fmt(rnd_ci[1])}, {_fmt(rnd_ci[2])}])")
    a(f"- UBT v2 model recall@6:    **{_fmt(v2_r6)}** "
      f"(95% CI [{_fmt(v2_ci[1])}, {_fmt(v2_ci[2])}])")

    outside_ci = v2_r6 > rnd_ci[2] or v2_r6 < rnd_ci[1]
    if outside_ci:
        a("")
        a("⚠️  The v2 model's performance is **outside** the random baseline CI.")
        a("    This is statistically unusual and requires independent data replication.")
    else:
        a("")
        a("✅  The v2 model's performance is **within** the random baseline CI.")
        a("    We **cannot** reject the null hypothesis.  No statistically significant")
        a("    predictive power has been demonstrated.")
    a("")
    a("**Conclusion:** Results must be replicated on independent real draw data before")
    a("any claim of predictive power can be made.  Theta features are marked")
    a("**experimental** and carry no theoretical guarantee.")
    a("")
    a("---")
    a("")
    a("## Constraints and Caveats")
    a("")
    a("- No predictive power is claimed without statistical significance.")
    a("- Theta transform is a deterministic spectral projection; it does not")
    a("  encode temporal causality.")
    a("- Toroidal boundary conditions impose an artificial spatial topology that")
    a("  may not reflect real lottery structure.")
    a("- CNN filters are fixed random projections (random kitchen sinks),")
    a("  not back-propagated through the convolution layers.")
    a("- All models use scikit-learn and are intentionally small to avoid overfitting.")
    a("- When run on synthetic data, all results trivially confirm the null hypothesis.")
    a("")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Sportka UBT/Theta Experiment v2")
    parser.add_argument(
        "--csv", type=str, default=None,
        help="Path to Sportka CSV (sazka.cz export). Omit for synthetic data.",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--no-cnn", action="store_true",
        help="Skip UBTCNNv2 model (faster run).",
    )
    args = parser.parse_args()
    run_experiment(csv_path=args.csv, seed=args.seed, include_cnn=not args.no_cnn)


if __name__ == "__main__":
    main()
