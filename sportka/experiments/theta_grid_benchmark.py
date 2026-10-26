"""Toroidal theta-grid ablation on the exact theta_mobius benchmark split.

This is a fixed spatial feature extractor + MLP, NOT an adaptive Theta Grid,
nor proof that the artificial 7x7 topology describes the lottery mechanism.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sportka.data import load_sportka_csv
from sportka.experiments import theta_mobius as benchmark

GRID_VERSION = "fixed-toroidal-theta-v1"


def toroidal_neighborhood(channel):
    """Fixed nearest-neighbor averaging, periodic in both grid axes."""
    grid = np.asarray(channel, dtype=np.float32).reshape(7, 7)
    return (grid + np.roll(grid, 1, 0) + np.roll(grid, -1, 0)
            + np.roll(grid, 1, 1) + np.roll(grid, -1, 1)).reshape(49) / 5


def grid_dataset(df):
    """147 inputs: raw last draw; toroidally mixed and theta mapped 16/52 averages.

    Every input X[t] uses only data from draws [0,t). Targets Y[t] are
    inaccessible at feature construction, including at split boundaries.
    """
    from sportka.features import base_features
    history = base_features(df)
    out = []
    for t in range(52, len(history)):
        previous = history[t - 1]
        medium = history[t - 16:t].mean(axis=0)
        long = history[t - 52:t].mean(axis=0)
        out.append(np.concatenate((
            previous,
            benchmark.spectral(toroidal_neighborhood(medium), "theta"),
            benchmark.spectral(toroidal_neighborhood(long), "theta"),
        )))
    return np.asarray(out, dtype=np.float32).reshape(-1, 147), history[52:]


def run(csv, game=1, cache_dir="models/sportka_research",
        *, hidden=32, max_iter=150, seed=42, fresh=False):
    import hashlib

    source = Path(csv)
    source_bytes = source.read_bytes()
    df = load_sportka_csv(source, game=game)
    if source.read_bytes() != source_bytes:
        raise ValueError("History changed while loading")
    if len(df) < 150:
        raise ValueError("Need at least 150 draw dates")
    X, Y = grid_dataset(df)
    n = len(Y)
    train_end, test_start = int(n * .7), int(n * .85)
    # Explicit new feature version and training-array signature in cache identity.
    key = benchmark.cache_key(source_bytes, game, GRID_VERSION, train_end,
                              hidden=hidden, max_iter=max_iter, seed=seed)
    model, reused = benchmark.train_or_load(
        X[:train_end], Y[:train_end], key=key, cache_dir=cache_dir,
        kind=GRID_VERSION, hidden=hidden, max_iter=max_iter,
        seed=seed, fresh=fresh)
    uniform_val = np.full_like(Y[train_end:test_start], 6 / 49)
    uniform_test = np.full_like(Y[test_start:], 6 / 49)
    dates = df["date"].iloc[52:].reset_index(drop=True)
    return {
        "method": GRID_VERSION,
        "meaning": "Fixed toroidal neighborhood, theta transformed temporal frequencies, MLP",
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "game": game, "feature_count": 147,
        "splits": {"train": train_end, "validation": test_start - train_end, "test": n - test_start},
        "test_dates": [str(dates.iloc[test_start].date()), str(dates.iloc[-1].date())],
        "checkpoint_reused": reused,
        "iteration_limit_reached": bool(model.n_iter_ >= max_iter),
        "theta_grid": {
            "validation": benchmark.metrics(Y[train_end:test_start], model.predict_proba(X[train_end:test_start])),
            "test": benchmark.metrics(Y[test_start:], model.predict_proba(X[test_start:])),
        },
        "uniform": {
            "validation": benchmark.metrics(Y[train_end:test_start], uniform_val),
            "test": benchmark.metrics(Y[test_start:], uniform_test),
        },
        "warning": "Exploratory: historical test data have been analyzed already; no demonstrated lottery edge."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--game", type=int, choices=(1, 2), default=1)
    parser.add_argument("--cache-dir", default="models/sportka_research")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fresh", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.csv, args.game, args.cache_dir,
                         hidden=args.hidden, max_iter=args.max_iter,
                         seed=args.seed, fresh=args.fresh), indent=2))


if __name__ == "__main__":
    main()
