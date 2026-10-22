"""Research-only Sportka comparison with opt-in cached models.

No betting advice: a fair lottery has no learnable temporal signal.
Usage:
 python -m sportka.experiments.theta_mobius --csv data/sportka.csv --game 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np

from sportka.data import load_sportka_csv

WINDOWS = (1, 16, 52)
ALPHA = 0.5
ORDER = 7
SEED = 42
VERS = "theta-mobius-v1"


def mobius(n: int) -> int:
    """Number-theoretic Möbius function, mu(1)=1."""
    if n < 1:
        raise ValueError("n must be >= 1")
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
    """Theta-like Fourier features; Möbius/Mertens weights are hypotheses."""
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
    """X[t] uses draws strictly before t, Y[t] is the next unknown draw."""
    rows = np.zeros((len(df), 49), dtype=np.float32)
    for t, numbers in enumerate(df["numbers"]):
        rows[t, np.array(numbers, dtype=int) - 1] = 1
    features = []
    for t in range(52, len(rows)):
        channels = [rows[t - w:t].mean(axis=0) for w in WINDOWS]
        if kind == "raw":
            features.append(np.concatenate(channels))
        else:
            features.append(np.concatenate([spectral(c, kind) for c in channels] + [rows[t - 1]]))
    return np.asarray(features, dtype=np.float32), rows[52:]


def cache_key(csv_bytes, game, kind, train_end):
    h = hashlib.sha256()
    h.update(csv_bytes)
    h.update(json.dumps([VERS, game, kind, train_end, WINDOWS, ALPHA, ORDER, SEED]).encode())
    return h.hexdigest()[:24]


def train_or_load(X, Y, *, key, cache_dir, kind, hidden=32, max_iter=150):
    from joblib import dump, load
    from sklearn.neural_network import MLPClassifier
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{kind}-{key}.joblib"
    if path.exists():
        record = load(path)  # Only load checkpoints from a trusted directory.
        if record.get("key") != key:
            raise ValueError("Checkpoint signature mismatch")
        return record["model"], True
    model = MLPClassifier(hidden_layer_sizes=(hidden,), alpha=0.1,
                          early_stopping=False, max_iter=max_iter, random_state=SEED)
    model.fit(X, Y)
    with tempfile.NamedTemporaryFile(dir=directory, suffix=".joblib", delete=False) as f:
        tmp = Path(f.name)
    try:
        dump({"model": model, "key": key, "train_rows": len(Y), "version": VERS}, tmp)
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return model, False


def metrics(Y, scores):
    from sklearn.metrics import log_loss
    scores = np.clip(np.asarray(scores), 1e-7, 1 - 1e-7)
    picks = np.argsort(-scores, axis=1)[:, :6]
    hits = np.take_along_axis(Y, picks, axis=1).sum(axis=1)
    bce = -np.mean(Y * np.log(scores) + (1 - Y) * np.log(1 - scores))
    return {"bce": float(bce), "mean_hits": float(hits.mean()), "n": len(Y)}


def run(csv, game=1, cache_dir="models/sportka_research"):
    source = Path(csv)
    df = load_sportka_csv(source, game=game)
    if len(df) < 150:
        raise ValueError("Need at least 150 chronological draws")
    result = {"source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
              "game": game, "note": "Exploratory, no confirmed predictive edge",
              "models": {}}
    for kind in ("raw", "theta", "mobius", "mertens"):
        X, Y = dataset(df, kind)
        n = len(Y)
        train_end = int(n * .7)
        test_start = int(n * .85)
        key = cache_key(source.read_bytes(), game, kind, train_end)
        model, reused = train_or_load(X[:train_end], Y[:train_end], key=key,
                                     cache_dir=cache_dir, kind=kind)
        pred = model.predict_proba(X[test_start:])
        result["models"][kind] = {**metrics(Y[test_start:], pred), "cache_reused": reused,
                                  "training_samples": train_end}
    _, Y = dataset(df, "raw")
    scores = np.full_like(Y[int(len(Y) * .85):], 6 / 49)
    result["models"]["uniform"] = metrics(Y[int(len(Y) * .85):], scores)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--game", type=int, choices=(1, 2), default=1)
    parser.add_argument("--cache-dir", default="models/sportka_research")
    args = parser.parse_args()
    print(json.dumps(run(args.csv, args.game, args.cache_dir), indent=2))


if __name__ == "__main__":
    main()
