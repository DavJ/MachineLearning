"""Causal recovery of artificial six-set signals, including fair-marginal pairs.

Fit on the first 70% of targets, select on the next 15%, score the final 15%.
Uniform simulations rerun this entire procedure, including all feature searches.
This experiment does not change the registered prospective forecast protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from sportka.data import load_sportka_csv
from sportka.features import base_features
from sportka.experiments.signal_audit import (
    PAIR_I, PAIR_J, mc_pvalue, sample_planted, sample_uniform, validate_rows,
    wilson_interval,
)

VERSION = "sportka-joint-recovery-v1"
MAX_STRENGTH = .95  # Keep positive probability for every valid six-set.
KINDS = ("uniform", "number", "pair", "copy")
LEVELS = (("uniform", 0.), ("number", .01), ("number", .05), ("number", .15),
          ("pair", .05), ("pair", .1), ("pair", .25),
          ("temporal", .005), ("temporal", .01), ("temporal", .025))


@dataclass(frozen=True)
class Model:
    kind: str
    balls: tuple[int, ...] = ()  # zero-based internally; one-based in reports
    strength: float = 0.

    def __post_init__(self):
        expected = {"uniform": 0, "number": 1, "pair": 2, "copy": 0}
        if (self.kind not in expected or len(self.balls) != expected[self.kind] or
                len(set(self.balls)) != len(self.balls) or
                any(isinstance(b, bool) or not isinstance(b, (int, np.integer)) or not 0 <= b < 49
                    for b in self.balls) or not np.isfinite(self.strength) or
                not 0 <= self.strength <= MAX_STRENGTH or
                (self.kind == "uniform" and self.strength != 0)):
            raise ValueError("Invalid registered joint model")

    def record(self):
        return {"kind": self.kind, "balls": [int(b) + 1 for b in self.balls],
                "strength": float(self.strength)}


def _geometry(rows):
    rows = np.asarray(rows)
    if rows.ndim != 2 or not len(rows) or not np.all((rows == 0) | (rows == 1)):
        raise ValueError("Need nonempty binary draw rows")
    count, pool = int(rows[0].sum()), rows.shape[1]
    if not 2 <= count <= pool - 2 or not np.all(rows.sum(axis=1) == count):
        raise ValueError("Rows must contain a constant number of distinct balls")
    return rows, pool, count


def log_ratios(targets, previous, model):
    """log(q(S | past) / uniform(S)); the probability is for the whole set."""
    targets, pool, count = _geometry(targets)
    previous, past_pool, past_count = _geometry(previous)
    if (previous.shape != targets.shape or (past_pool, past_count) != (pool, count) or
            any(b >= pool for b in model.balls)):
        raise ValueError("Forecast/target dimensions differ")
    if model.kind == "uniform":
        return np.zeros(len(targets), dtype=np.float64)
    if model.kind == "number":
        ratio = targets[:, model.balls[0]] * (pool / count)
    elif model.kind == "pair":
        category = targets[:, model.balls].sum(axis=1)
        ratio = np.where(category == 2, (pool - 1) / (count - 1),
                         np.where(category == 0, (pool - 1) / (pool - count - 1), 0.))
    else:
        ratio = np.all(targets == previous, axis=1) * math.comb(pool, count)
    return np.log1p(model.strength * (ratio - 1))


def marginals(previous, model):
    """Individual-ball probabilities implied by the coherent set distribution."""
    previous, pool, count = _geometry(previous)
    p, s = count / pool, model.strength
    if any(b >= pool for b in model.balls):
        raise ValueError("Model ball is outside the pool")
    if model.kind == "copy":
        return (1 - s) * p + s * previous
    probabilities = np.full(previous.shape, p, dtype=np.float64)
    if model.kind == "number":
        probabilities[:] = (1 - s) * p + s * (count - 1) / (pool - 1)
        probabilities[:, model.balls[0]] = (1 - s) * p + s
    # The coupled-pair law has exactly uniform single-ball marginals for any s.
    return probabilities


def map_sets(previous, model):
    """One deterministic maximum-probability set; ties use ascending numbers."""
    previous, pool, count = _geometry(previous)
    if any(b >= pool for b in model.balls):
        raise ValueError("Model ball is outside the pool")
    if model.kind == "copy" and model.strength > 0:
        return previous.copy()
    chosen = list(model.balls) if model.strength > 0 else []
    chosen.extend(n for n in range(pool) if n not in chosen)
    result = np.zeros_like(previous, dtype=np.uint8)
    result[:, chosen[:count]] = 1
    return result


def metrics(targets, previous, model):
    gains = log_ratios(targets, previous, model)
    probabilities = marginals(previous, model)
    bce = -(targets * np.log(probabilities) + (1 - targets) * np.log1p(-probabilities)).mean()
    return {"joint_gain_nats_per_draw": float(gains.mean()),
            "joint_nll": float(math.log(math.comb(targets.shape[1], int(targets[0].sum()))) - gains.mean()),
            "marginal_bce": float(bce),
            "marginal_brier": float(np.mean((probabilities - targets) ** 2)),
            "mean_main_hits": float(np.sum(map_sets(previous, model) * targets, axis=1).mean())}


def fit_models(targets, previous):
    """Training-only MLE: search 49 positive number biases and all 1,176 pairs."""
    targets = validate_rows(targets)
    previous = validate_rows(previous)
    if targets.shape != previous.shape:
        raise ValueError("Training targets and past rows must align")
    n, pool = targets.shape
    count = 6
    p = count / pool
    values = targets.astype(np.float64)
    singles = values.sum(axis=0)
    s_number = np.clip((singles / n - p) / (1 - p), 0, MAX_STRENGTH)
    number_gain = singles * np.log1p(s_number * (1 / p - 1)) + (n - singles) * np.log1p(-s_number)
    number = int(np.argmax(number_gain))

    both = (values.T @ values)[PAIR_I, PAIR_J]
    single = singles[PAIR_I] + singles[PAIR_J] - 2 * both
    neither = n - both - single
    a, c = (pool - 1) / (count - 1) - 1, (pool - 1) / (pool - count - 1) - 1
    # Every profile likelihood is concave. Vectorized bisection solves all pairs
    # without a data-dependent optimizer or validation/test parameter fitting.
    lo, hi = np.zeros(len(both)), np.full(len(both), MAX_STRENGTH)
    def derivative(s):
        return a * both / (1 + a * s) - single / (1 - s) + c * neither / (1 + c * s)
    for _ in range(52):
        mid = (lo + hi) / 2
        increasing = derivative(mid) > 0
        lo, hi = np.where(increasing, mid, lo), np.where(increasing, hi, mid)
    s_pair = np.where(derivative(0.) <= 0, 0.,
                      np.where(derivative(MAX_STRENGTH) >= 0, MAX_STRENGTH, (lo + hi) / 2))
    pair_gain = both * np.log1p(a * s_pair) + single * np.log1p(-s_pair) + neither * np.log1p(c * s_pair)
    pair = int(np.argmax(pair_gain))

    repeats = float(np.all(targets == previous, axis=1).mean())
    chance = 1 / math.comb(pool, count)
    s_copy = float(np.clip((repeats - chance) / (1 - chance), 0, MAX_STRENGTH))
    return [Model("uniform"), Model("number", (number,), float(s_number[number])),
            Model("pair", (int(PAIR_I[pair]), int(PAIR_J[pair])), float(s_pair[pair])),
            Model("copy", strength=s_copy)]


def pipeline(rows, *, details=True):
    rows = validate_rows(rows)
    if len(rows) < 150:
        raise ValueError("Need at least 150 historical dates")
    previous, targets = rows[:-1], rows[1:]
    n = len(targets)
    train_end, test_start = int(.70 * n), int(.85 * n)
    models = fit_models(targets[:train_end], previous[:train_end])
    validation = [float(log_ratios(targets[train_end:test_start], previous[train_end:test_start], model).mean())
                  for model in models]
    selected = models[int(np.argmax(validation))]  # Uniform is first and wins exact ties.
    test_targets, test_previous = targets[test_start:], previous[test_start:]
    result = {"splits": {"train": train_end, "validation": test_start - train_end, "test": n - test_start},
              "fitted_models": {model.kind: model.record() for model in models},
              "validation_joint_gains": dict(zip(KINDS, validation)),
              "selected_model": selected.kind,
              "selected_test_joint_gain": float(log_ratios(test_targets, test_previous, selected).mean())}
    if details:
        result["test_scores"] = {model.kind: metrics(test_targets, test_previous, model) for model in models}
    return result


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def calibrate_null(n, resamples=999, seed=42, progress=None):
    _positive_integer(resamples, "resamples")
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, 7321, 0]))
    scores, selected = np.empty(resamples), Counter()
    for i in range(resamples):
        result = pipeline(sample_uniform(n, rng), details=False)
        scores[i] = result["selected_test_joint_gain"]
        selected[result["selected_model"]] += 1
        if progress and (i + 1) % max(1, resamples // 10) == 0:
            progress(f"Whole-pipeline null: {i + 1}/{resamples}")
    return scores, {kind: selected[kind] for kind in KINDS}


def sensitivity(n, reference, *, trials=64, seed=42, family_count=2, alpha=.05, progress=None):
    _positive_integer(trials, "trials")
    levels = []
    test_start = int(.85 * (n - 1))
    for index, (kind, strength) in enumerate(LEVELS):
        detections, positive, recovered = 0, 0, 0
        selected, gains, oracle_gains, oracle_bce_gains = Counter(), [], [], []
        model_kind = "copy" if kind == "temporal" else kind
        balls = (0,) if kind == "number" else ((0, 1) if kind == "pair" else ())
        oracle = Model(model_kind, balls, strength)
        for trial in range(trials):
            rng = np.random.default_rng(np.random.SeedSequence([seed, n, 7321, 1, index, trial]))
            rows = sample_uniform(n, rng) if kind == "uniform" else sample_planted(n, kind, strength, rng)
            result = pipeline(rows, details=False)
            score = result["selected_test_joint_gain"]
            detections += score > 0 and family_count * mc_pvalue(score, reference) <= alpha
            positive += score > 0
            gains.append(score)
            chosen = result["selected_model"]
            selected[chosen] += 1
            recovered += chosen == model_kind and result["fitted_models"][chosen]["balls"] == [b + 1 for b in balls]
            target, previous = rows[1:][test_start:], rows[:-1][test_start:]
            oracle_score = metrics(target, previous, oracle)
            uniform_score = metrics(target, previous, Model("uniform"))
            oracle_gains.append(oracle_score["joint_gain_nats_per_draw"])
            oracle_bce_gains.append(uniform_score["marginal_bce"] - oracle_score["marginal_bce"])
        levels.append({"alternative": kind, "strength": strength, "trials": trials,
                       "detections": int(detections), "detection_rate": detections / trials,
                       "conditional_wilson_95_interval": wilson_interval(detections, trials),
                       "positive_test_gain_trials": int(positive),
                       "correct_structure_selected_trials": int(recovered),
                       "selected_model_counts": {name: selected[name] for name in KINDS},
                       "mean_selected_test_joint_gain": float(np.mean(gains)),
                       "oracle_mean_test_joint_gain": float(np.mean(oracle_gains)),
                       "oracle_mean_marginal_bce_gain": float(np.mean(oracle_bce_gains))})
        if progress:
            progress(f"Recovery {kind}, strength={strength}: {detections}/{trials}")
    return {"draw_count": n, "trials_per_level": trials,
            "meaning": "Independent synthetic histories; oracle knows the planted law and is never a fitted candidate",
            "note": "One-game detection with the same across-game correction as the real run. Wilson intervals condition on the shared Monte Carlo reference and exclude its uncertainty. Controls do not describe every possible signal.",
            "levels": levels}


def run(csv, *, games=(1, 2), resamples=999, trials=64, seed=42,
        alpha=.05, power=True, progress=None):
    _positive_integer(resamples, "resamples")
    if (not games or len(set(games)) != len(games) or any(type(g) is not int or g not in (1, 2) for g in games) or
            not 0 < alpha < 1 or type(seed) is not int or seed < 0):
        raise ValueError("Need distinct games 1/2, alpha in (0,1), and a nonnegative integer seed")
    if len(games) / (resamples + 1) > alpha:
        raise ValueError("More resamples needed for the corrected significance threshold")
    if power:
        _positive_integer(trials, "trials")
    source = Path(csv)
    source_bytes = source.read_bytes()
    result = {"version": VERSION, "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "seed": seed, "numpy": np.__version__, "resamples": resamples, "alpha": alpha,
              "max_strength": MAX_STRENGTH, "training_fraction": .70, "test_start_fraction": .85,
              "correction": "Rerun training searches and validation selection under uniform whole-set histories, then Bonferroni across games",
              "prediction_edge": "NOT_ESTABLISHED", "games": {}, "null_calibrations": {}, "sensitivity": {},
              "limitations": ["Historical and exploratory: these dates have already been examined in project research",
                              "Simple matched mixture families, not a test of all neural/theta architectures or all possible signals",
                              "Correction covers this fixed pipeline and requested games, not all earlier or later research choices",
                              "Positive joint log score is not a measured jackpot or profitability advantage",
                              "No prospective forecast, registered alpha budget, betting integration or pending ledger is changed"]}
    references = {}
    for game in games:
        df = load_sportka_csv(source, game=game)
        rows = validate_rows(base_features(df))
        fitted = pipeline(rows)
        n = len(rows)
        if n not in references:
            reference, counts = calibrate_null(n, resamples, seed, progress)
            references[n] = reference
            result["null_calibrations"][str(n)] = {"selected_model_counts": counts,
                                                   "selected_test_gains": reference.tolist()}
        pvalue = mc_pvalue(fitted["selected_test_joint_gain"], references[n])
        adjusted = min(1., len(games) * pvalue)
        train_end = fitted["splits"]["train"]
        test_start = train_end + fitted["splits"]["validation"]
        dates = df["date"].iloc[1:]
        fitted.update({"draw_count": n, "dates": [str(df.date.min().date()), str(df.date.max().date())],
                       "target_date_ranges": {name: [str(dates.iloc[a].date()), str(dates.iloc[b - 1].date())]
                                              for name, a, b in (("train", 0, train_end), ("validation", train_end, test_start),
                                                                 ("test", test_start, len(dates)))},
                       "pipeline_pvalue": pvalue, "across_games_adjusted_pvalue": adjusted,
                       "historical_rejection": bool(fitted["selected_test_joint_gain"] > 0 and adjusted <= alpha)})
        result["games"][str(game)] = fitted
        if progress:
            progress(f"Game {game}: selected={fitted['selected_model']}, test gain={fitted['selected_test_joint_gain']:.6g}, adjusted p={adjusted:.4g}")
    if power:
        for n, reference in references.items():
            result["sensitivity"][str(n)] = sensitivity(n, reference, trials=trials, seed=seed,
                                                       family_count=len(games), alpha=alpha, progress=progress)
    if source.read_bytes() != source_bytes:
        raise ValueError("Source CSV changed during the experiment")
    result["verdict"] = ("EXPLORATORY_JOINT_DEVIATION_REQUIRES_REPLICATION"
                         if any(g["historical_rejection"] for g in result["games"].values())
                         else "NO_CORRECTED_JOINT_PREDICTIVE_GAIN")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--game", choices=("both", "1", "2"), default="both")
    parser.add_argument("--resamples", type=int, default=999)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-power", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(args.csv, games=(1, 2) if args.game == "both" else (int(args.game),),
                     resamples=args.resamples, trials=args.trials, seed=args.seed, power=not args.no_power,
                     progress=(lambda msg: print(msg, file=sys.stderr, flush=True)) if args.progress else None)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
