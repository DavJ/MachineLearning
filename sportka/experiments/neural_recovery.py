"""Frozen neural pipeline tested on real history and known artificial signals.

Historical scores are exploratory. Synthetic recovery is not a lottery edge.
Use only trusted local joblib checkpoint directories.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from math import comb, log
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from sportka.data import load_sportka_csv
from sportka.features import base_features
from sportka.fixed_cardinality import FixedCardinality
from sportka.experiments.signal_audit import sample_planted, sample_uniform, validate_rows, wilson_interval
from sportka.experiments.theta_mobius import (
    KINDS, WINDOWS, cache_key, dataset, library_versions, model_config, train_or_load,
)

VERSION = "sportka-neural-recovery-v1"
WEIGHTS = (0., .01, .03, .1, .3, 1.)
LEVELS = (("uniform", 0.), ("number", .01), ("number", .05),
          ("number", .15), ("pair", .25), ("temporal", .1), ("temporal", .25))
LOG_UNIFORM_NLL = log(comb(49, 6))
# One selected forecast per game, alpha=.05 / 2. No game independence needed.
E_THRESHOLD = 40.


def blend_gain(log_ratio, weight):
    """log(((1-weight) U + weight Q) / U), stably and with full support."""
    values = np.asarray(log_ratio, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Need nonempty finite per-draw log ratios")
    if not np.isfinite(weight) or not 0 <= weight <= 1:
        raise ValueError("weight must be in [0,1]")
    if weight == 0:
        return np.zeros_like(values)
    if weight == 1:
        return values.copy()
    return np.logaddexp(np.log1p(-weight), log(weight) + values)


def evidence(gains):
    """Terminal likelihood-ratio evidence. Logs remain usable at extreme sizes."""
    gains = np.asarray(gains, dtype=float)
    if gains.ndim != 1 or not len(gains) or not np.isfinite(gains).all():
        raise ValueError("Need finite nonempty gains")
    total = float(gains.sum())
    return {"mean_joint_gain": float(gains.mean()),
            "joint_nll": LOG_UNIFORM_NLL - float(gains.mean()),
            "terminal_log_e": total,
            "terminal_e": float(np.exp(total)) if -700 < total < 700 else None,
            "threshold": E_THRESHOLD, "terminal_crossing": total >= log(E_THRESHOLD)}


def pipeline(rows, cache_dir, *, fresh=False):
    """Train four fixed MLPs, select on validation, then score one frozen choice.

    Every family/weight choice precedes any test prediction. Uniform is the
    first candidate and wins ties. FixedCardinality specifies Q for each MLP;
    the optional validation weight mixes complete-set laws, not raw marginals.
    """
    rows = validate_rows(rows)
    if len(rows) < 200:
        raise ValueError("Need at least 200 draws")
    frame = pd.DataFrame({"numbers": [(np.flatnonzero(row) + 1).tolist() for row in rows]})
    n = len(rows) - max(WINDOWS)
    train_end, test_start = int(.7 * n), int(.85 * n)
    best = {"kind": "uniform", "weight": 0., "validation_mean_joint_gain": 0.}
    fitted, diagnostics = {}, {}
    identity = VERSION.encode() + rows.tobytes()
    for kind in KINDS:
        X, Y = dataset(frame, kind)
        key = cache_key(identity, 0, kind, train_end)
        model, reused = train_or_load(X[:train_end], Y[:train_end], key=key,
                                     cache_dir=cache_dir, kind=kind, fresh=fresh)
        val = FixedCardinality(model.predict_proba(X[train_end:test_start])).log_prob(Y[train_end:test_start]) + LOG_UNIFORM_NLL
        options = [float(blend_gain(val, weight).mean()) for weight in WEIGHTS]
        chosen = 0
        for i in range(1, len(options)):
            if options[i] > options[chosen] + 1e-12:
                chosen = i
        if options[chosen] > best["validation_mean_joint_gain"] + 1e-12:
            best = {"kind": kind, "weight": WEIGHTS[chosen],
                    "validation_mean_joint_gain": options[chosen]}
        diagnostics[kind] = {"validation_gains_by_weight": options,
                             "validation_best_weight": WEIGHTS[chosen],
                             "cache_reused": reused, "checkpoint_key": key,
                             "iterations": int(model.n_iter_),
                             "iteration_limit_reached": bool(model.n_iter_ >= 150)}
        fitted[kind] = (model, X, Y)
    if best["kind"] == "uniform":
        test = np.zeros(n - test_start)
    else:
        model, X, Y = fitted[best["kind"]]
        test = FixedCardinality(model.predict_proba(X[test_start:])).log_prob(Y[test_start:]) + LOG_UNIFORM_NLL
        test = blend_gain(test, best["weight"])
    return {"selected": best, "test": evidence(test), "fit_diagnostics": diagnostics,
            "split_samples": {"training": train_end, "validation": test_start - train_end,
                              "test": n - test_start},
            "test_source_start": max(WINDOWS) + test_start}


def oracle_gain(rows, kind, strength, start):
    """Known planted complete-set law, never used for fitting or selection."""
    y, previous = rows[start:], rows[start - 1:-1]
    if kind == "uniform":
        return np.zeros(len(y))
    if kind == "number":
        ratio = (49 / 6) * y[:, 0]
    elif kind == "pair":
        count = y[:, :2].sum(axis=1)
        ratio = np.where(count == 2, 48 / 5, np.where(count == 0, 8 / 7, 0.))
    elif kind == "temporal":
        ratio = comb(49, 6) * np.all(y == previous, axis=1)
    else:
        raise ValueError(kind)
    return np.log1p(strength * (ratio - 1))


def _control(task):
    level, trial, n, seed, cache_dir, fresh = task
    kind, strength = LEVELS[level]
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, 901, level, trial]))
    rows = sample_uniform(n, rng) if kind == "uniform" else sample_planted(n, kind, strength, rng)
    fitted = pipeline(rows, cache_dir, fresh=fresh)
    return level, trial, {"selected": fitted["selected"], "test": fitted["test"],
                          "fits_at_iteration_limit": sum(d["iteration_limit_reached"] for d in fitted["fit_diagnostics"].values()),
                          "oracle": evidence(oracle_gain(rows, kind, strength, fitted["test_source_start"]))}


def sensitivity(n, cache_dir, *, trials=16, seed=42, workers=2, fresh=False, progress=None):
    if (isinstance(trials, bool) or not isinstance(trials, int) or trials < 1 or
            isinstance(workers, bool) or not isinstance(workers, int) or not 1 <= workers <= 4 or
            isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 or n < 200):
        raise ValueError("Need n>=200, positive integer trials, nonnegative seed, workers in 1..4")
    tasks = [(i, j, n, seed, str(cache_dir), fresh) for i in range(len(LEVELS)) for j in range(trials)]
    results = {}
    def collect(record):
        i, j, result = record
        results[i, j] = result
        if progress:
            progress(f"Neural controls {len(results)}/{len(tasks)}")
    if workers == 1:
        for task in tasks:
            collect(_control(task))
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_control, task) for task in tasks]
            for future in as_completed(futures):
                collect(future.result())
    levels = []
    for i, (kind, strength) in enumerate(LEVELS):
        records = [results[i, j] for j in range(trials)]
        count = sum(r["test"]["terminal_crossing"] for r in records)
        levels.append({"kind": kind, "strength": strength, "crossings": count,
                       "trials": trials, "wilson95": wilson_interval(count, trials),
                       "oracle_crossings": sum(r["oracle"]["terminal_crossing"] for r in records),
                       "selected_counts": {k: sum(r["selected"]["kind"] == k for r in records)
                                           for k in ("uniform", *KINDS)},
                       "trials_detail": records})
    return {"rows_per_history": n, "seed": seed, "trials_per_level": trials,
            "threshold_per_game": E_THRESHOLD, "levels": levels,
            "note": "Synthetic recovery of the entire frozen pipeline. These finite matched controls do not establish lottery power or profit; oracle laws never participate in training/selection."}


def run(csv, cache_dir, *, trials=16, seed=42, workers=2, power=True, fresh=False, progress=None):
    source = Path(csv)
    original = source.read_bytes()
    games = {}
    for game in (1, 2):
        frame = load_sportka_csv(source, game=game)
        result = pipeline(base_features(frame), cache_dir, fresh=fresh)
        result["source_rows"] = len(frame)
        result["test_dates"] = [str(frame.date.iloc[result["test_source_start"]].date()), str(frame.date.iloc[-1].date())]
        games[str(game)] = result
        if progress:
            progress(f"Real game {game} complete: {result['selected']}")
    if source.read_bytes() != original:
        raise ValueError("CSV changed during loading")
    report = {"version": VERSION, "source_sha256": hashlib.sha256(original).hexdigest(),
              "libraries": library_versions(), "model_configuration": model_config(),
              "weights": list(WEIGHTS), "feature_kinds": list(KINDS), "games": games,
              "prediction_edge": "NOT_ESTABLISHED",
              "historical_note": "Already-explored history: evidence values are exploratory, not confirmatory p-values. No prospective ledger is written.",
              "null_control_note": "For genuinely fresh uniform conditional draws, each frozen selected-law terminal likelihood ratio has mean one. Markov gives P(E>=40)<=.025 per game, hence <=.05 for either of two games; this does not cover project-wide historical searches."}
    if power:
        report["sensitivity"] = sensitivity(len(frame), cache_dir, trials=trials, seed=seed, workers=workers, fresh=fresh, progress=progress)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--cache-dir", default="models/neural_recovery", help="Trusted local joblib files only")
    parser.add_argument("--trials", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--no-power", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="Explicitly rebuild even matching or unreadable checkpoints")
    args = parser.parse_args()
    try:
        result = run(args.csv, args.cache_dir, trials=args.trials, seed=args.seed,
                     workers=args.workers, power=not args.no_power, fresh=args.fresh,
                     progress=lambda text: print(text, file=sys.stderr, flush=True))
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
