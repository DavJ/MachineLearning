"""Exploratory theta/Möbius/Mertens comparison with trusted local checkpoints.

No prediction edge is established. Each variant uses the same past-only
feature dimensions, target rows, model settings and chronological split.
"""
from __future__ import annotations

import argparse
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from sportka.data import load_sportka_csv
from sportka.features import base_features

WINDOWS = (1, 16, 52)
ALPHA = 0.5
ORDER = 7
SEED = 42
VERS = "theta-mobius-v2"
KINDS = ("raw", "theta", "mobius", "mertens")


def mobius(n: int) -> int:
    """Number-theoretic Möbius function, mu(1)=1."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError("n must be a positive integer")
    factors = 0
    p = 2
    while p * p <= n:
        if n % p == 0:
            n //= p
            if n % p == 0:
                return 0
            factors += 1
        p += 1
    return (-1 if factors % 2 else 1) * (-1 if n > 1 else 1)


def spectral(v: np.ndarray, kind: str) -> np.ndarray:
    """Damped cosine features; Möbius/Mertens coefficients are hypotheses.

    This is a finite theta-like cosine sum, not a derived physical model or
    an implementation of the separate adaptive Theta Grid architecture.
    """
    if kind not in ("theta", "mobius", "mertens"):
        raise ValueError(kind)
    coefficients = np.array([mobius(n) for n in range(1, ORDER + 1)])
    if kind == "theta":
        coefficients[:] = 1
    elif kind == "mertens":
        coefficients = np.cumsum(coefficients)
    n = np.arange(1, ORDER + 1, dtype=float)
    weights = coefficients * np.exp(-ALPHA * n * n)
    phase = 2 * np.pi * np.asarray(v, dtype=float)[..., None] * n
    return np.sum(weights * np.cos(phase), axis=-1).astype(np.float32)


def dataset(df, kind: str):
    """X[t] uses only draws before target t, with equal inputs across variants.

    The last draw stays raw for every variant. Applying cos(2*pi*n*x) to
    binary x makes zero and one indistinguishable, so only the 16/52-draw
    frequency channels receive a spectral transform. All variants have
    3*49 features and the same 52-draw warmup.
    """
    if kind not in KINDS:
        raise ValueError(f"Unknown feature kind: {kind}")
    rows = base_features(df)
    warmup = max(WINDOWS)
    features = []
    for t in range(warmup, len(rows)):
        channels = []
        for window in WINDOWS:
            channel = rows[t - window:t].mean(axis=0)
            if kind != "raw" and window != 1:
                channel = spectral(channel, kind)
            channels.append(channel)
        features.append(np.concatenate(channels))
    return (np.asarray(features, dtype=np.float32).reshape(-1, len(WINDOWS) * 49),
            rows[warmup:])


def model_config(hidden=32, max_iter=150, seed=SEED):
    if hidden < 1 or max_iter < 1:
        raise ValueError("hidden and max_iter must be positive")
    return {"hidden_layer_sizes": (hidden,), "alpha": 0.1,
            "early_stopping": False, "max_iter": max_iter,
            "random_state": seed, "solver": "adam", "shuffle": True,
            "learning_rate_init": 0.001}


def library_versions():
    return {name: version(name) for name in ("numpy", "scikit-learn", "joblib")}


def cache_key(csv_bytes, game, kind, train_end, *, hidden=32, max_iter=150, seed=SEED):
    """Include source, features, all configured model parameters and libraries."""
    h = hashlib.sha256(csv_bytes)
    configuration = {"version": VERS, "game": game, "kind": kind,
                     "train_end": train_end, "windows": WINDOWS, "alpha": ALPHA,
                     "order": ORDER, "model": model_config(hidden, max_iter, seed),
                     "libraries": library_versions()}
    h.update(json.dumps(configuration, sort_keys=True).encode())
    return h.hexdigest()[:24]


def training_signature(X, Y):
    """Protect against a caller accidentally reusing a key for other arrays."""
    h = hashlib.sha256()
    for array in (X, Y):
        array = np.ascontiguousarray(array)
        h.update(str((array.shape, array.dtype.str)).encode())
        h.update(array.tobytes())
    return h.hexdigest()


def train_or_load(X, Y, *, key, cache_dir, kind, hidden=32, max_iter=150,
                  seed=SEED, fresh=False):
    """Reuse identical training only. This does not resume on a changed dataset.

    joblib checkpoints may execute code: use only a directory you control.
    A changed source/configuration creates a new key; --fresh bypasses reuse.
    """
    from joblib import dump, load
    from sklearn.neural_network import MLPClassifier
    configuration = model_config(hidden, max_iter, seed)
    libraries = library_versions()
    signature = training_signature(X, Y)
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}-{key}.joblib"
    if path.exists() and not fresh:
        record = load(path)
        expected = {"key": key, "train_signature": signature, "kind": kind,
                    "configuration": configuration, "libraries": libraries,
                    "version": VERS}
        if not isinstance(record, dict) or any(record.get(k) != v for k, v in expected.items()):
            raise ValueError("Checkpoint signature mismatch; rerun with --fresh")
        return record["model"], True
    model = MLPClassifier(**configuration)
    model.fit(X, Y)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".joblib", delete=False) as f:
        tmp = Path(f.name)
    try:
        dump({"model": model, "key": key, "train_signature": signature,
              "train_rows": len(Y), "version": VERS, "kind": kind,
              "configuration": configuration, "libraries": libraries}, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return model, False


def metrics(Y, scores):
    from sportka.evaluation import compute_all_metrics
    result = compute_all_metrics(Y, np.asarray(scores))
    result["n"] = len(Y)
    return result


def run(csv, game=1, cache_dir="models/sportka_research", *, hidden=32,
        max_iter=150, seed=SEED, fresh=False):
    source = Path(csv)
    source_bytes = source.read_bytes()
    df = load_sportka_csv(source, game=game)
    if source.read_bytes() != source_bytes:
        raise ValueError("CSV changed during loading; retry with a stable file")
    if len(df) < 150:
        raise ValueError("Need at least 150 chronological draws")
    warmup = max(WINDOWS)
    n = len(df) - warmup
    train_end, test_start = int(n * .7), int(n * .85)
    dates = df["date"].iloc[warmup:].reset_index(drop=True)
    result = {"source_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "game": game, "version": VERS,
              "note": "Exploratory; no confirmed predictive edge or profit estimate",
              "comparison": "Same feature dimensions, target dates and model configuration",
              "cache_behavior": "Reuse identical training; changed data/config retrains, without incremental resume",
              "model_configuration": model_config(hidden, max_iter, seed),
              "libraries": library_versions(), "source_rows": len(df),
              "warmup_rows": warmup, "feature_count": len(WINDOWS) * 49,
              "split_samples": {"training": train_end, "validation": test_start - train_end,
                                "test": n - test_start},
              "validation_dates": [str(dates.iloc[train_end].date()), str(dates.iloc[test_start - 1].date())],
              "test_dates": [str(dates.iloc[test_start].date()), str(dates.iloc[-1].date())],
              "models": {}}
    for kind in KINDS:
        X, Y = dataset(df, kind)
        key = cache_key(source_bytes, game, kind, train_end, hidden=hidden,
                        max_iter=max_iter, seed=seed)
        model, reused = train_or_load(X[:train_end], Y[:train_end], key=key,
                                     cache_dir=cache_dir, kind=kind, hidden=hidden,
                                     max_iter=max_iter, seed=seed, fresh=fresh)
        result["models"][kind] = {
            "validation": metrics(Y[train_end:test_start], model.predict_proba(X[train_end:test_start])),
            "test": metrics(Y[test_start:], model.predict_proba(X[test_start:])),
            "cache_reused": reused, "checkpoint_key": key,
            "training_iterations": int(model.n_iter_),
            "reached_iteration_limit": bool(model.n_iter_ >= max_iter),
            "final_training_loss": float(model.loss_)}
    result["models"]["uniform"] = {
        "validation": metrics(Y[train_end:test_start], np.full_like(Y[train_end:test_start], 6 / 49)),
        "test": metrics(Y[test_start:], np.full_like(Y[test_start:], 6 / 49))}
    result["interpretation"] = (
        "Point estimates only. Comparing several feature variants on one test period "
        "does not demonstrate significance or justify choosing a betting model. "
        "Replicate a prespecified choice on untouched future draws.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--game", type=int, choices=(1, 2), default=1)
    parser.add_argument("--cache-dir", default="models/sportka_research",
                        help="Trusted local checkpoint directory")
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--fresh", action="store_true", help="Retrain even if a matching checkpoint exists")
    args = parser.parse_args()
    try:
        result = run(args.csv, args.game, args.cache_dir, hidden=args.hidden,
                     max_iter=args.max_iter, seed=args.seed, fresh=args.fresh)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
