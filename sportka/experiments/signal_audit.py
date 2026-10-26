"""Exploratory 6/49 randomness audit and planted-signal sensitivity checks.

Numbers/pairs use iid uniform subsets without replacement. Temporal scores use
permutations of the observed rows, preserving all marginal/pair frequencies.
Max-statistic calibration controls searches within a family; Bonferroni also
accounts for every family and draw requested in this invocation. Results are
historical diagnostics, never proof of a prospective prediction advantage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile

import numpy as np

from sportka.data import load_sportka_csv
from sportka.features import base_features
from sportka.rules import MAIN_COUNT, POOL_SIZE

VERSION = "sportka-signal-audit-v1"
LAGS = (1, 2, 5, 10)
PAIR_I, PAIR_J = np.triu_indices(POOL_SIZE, 1)
P_NUMBER = MAIN_COUNT / POOL_SIZE
P_PAIR = MAIN_COUNT * (MAIN_COUNT - 1) / (POOL_SIZE * (POOL_SIZE - 1))


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def validate_rows(rows):
    rows = np.asarray(rows)
    if (rows.ndim != 2 or rows.shape[1] != POOL_SIZE or len(rows) < 2 or
            not np.all((rows == 0) | (rows == 1)) or
            not np.all(rows.sum(axis=1) == MAIN_COUNT)):
        raise ValueError("Expected (n, 49) binary rows with six main numbers, n >= 2")
    return rows.astype(np.uint8)


def _subsets(n, count, pool, rng):
    # Six smallest independent continuous ranks give a uniform unordered subset.
    # rng.choice(pool, size=(n, 6), replace=False) would wrongly sample globally.
    ranks = rng.random((n, pool))
    return np.argpartition(ranks, count - 1, axis=1)[:, :count]


def sample_uniform(n, rng):
    _positive_integer(n, "n")
    indices = _subsets(n, MAIN_COUNT, POOL_SIZE, rng)
    rows = np.zeros((n, POOL_SIZE), dtype=np.uint8)
    rows[np.arange(n)[:, None], indices] = 1
    return rows


def sample_planted(n, kind, strength, rng):
    """Known artificial alternatives, with exactly six distinct balls per row.

    number: force ball 1 in a fraction strength of otherwise fair rows.
    pair: mix fair rows with a law where balls 1/2 appear together with p=6/49,
          or both are absent. Every individual marginal remains exactly 6/49.
    temporal: copy the previous row with probability strength; stationary
              single-draw marginals and within-draw pair probabilities stay fair.
    These controls are never mixed into the user's real history.
    """
    if kind not in ("number", "pair", "temporal"):
        raise ValueError(f"Unknown planted alternative: {kind}")
    if not np.isfinite(strength) or not 0 <= strength <= 1:
        raise ValueError("strength must be finite and in [0, 1]")
    rows = sample_uniform(n, rng)
    selected = rng.random(n) < strength
    if kind == "number":
        for index in np.flatnonzero(selected & (rows[:, 0] == 0)):
            removed = rng.choice(np.flatnonzero(rows[index]))
            rows[index, removed] = 0
            rows[index, 0] = 1
    elif kind == "pair":
        both = rng.random(n) < P_NUMBER
        for include_pair in (False, True):
            indices = np.flatnonzero(selected & (both == include_pair))
            count = MAIN_COUNT - (2 if include_pair else 0)
            choices = _subsets(len(indices), count, POOL_SIZE - 2, rng) + 2
            rows[indices] = 0
            rows[indices[:, None], choices] = 1
            if include_pair:
                rows[indices, :2] = 1
    else:
        for index in range(1, n):
            if selected[index]:
                rows[index] = rows[index - 1]
    return rows


def uniform_components(rows):
    """Counts, exact uniform expectations and scaled deviations for numbers/pairs."""
    n = len(rows)
    values = np.asarray(rows, dtype=np.float64)
    counts = values.sum(axis=0)
    pairs = (values.T @ values)[PAIR_I, PAIR_J]
    result = {}
    for name, observed, probability in (("numbers", counts, P_NUMBER),
                                        ("pairs", pairs, P_PAIR)):
        expected = np.full(len(observed), n * probability)
        scale = np.sqrt(n * probability * (1 - probability))
        result[name] = (observed, expected, np.abs(observed - expected) / scale)
    return result


def temporal_context(rows, lags=LAGS):
    n = len(rows)
    if (not lags or len(set(lags)) != len(lags) or
            any(isinstance(lag, bool) or not isinstance(lag, (int, np.integer)) or
                not 1 <= lag < n for lag in lags)):
        raise ValueError("lags must be distinct positive integers smaller than the history")
    counts = rows.sum(axis=0, dtype=np.float64)
    # Exact expected co-occurrence of a number in two distinct randomly ordered
    # rows, conditional on the observed history. No independence variance claim.
    q = counts * (counts - 1) / (n * (n - 1))
    expected, labels = [], []
    for lag in lags:
        means = (n - lag) * q
        expected.extend(means)
        expected.append(float(means.sum()))
        labels.extend({"lag": int(lag), "number": number} for number in range(1, 50))
        labels.append({"lag": int(lag), "aggregate_overlap": True})
    expected = np.asarray(expected, dtype=np.float64)
    # Pearson-style scaling only. Permutations, rather than a normal/chi-square
    # approximation, calibrate the complete maximum statistic.
    return expected, np.sqrt(np.maximum(expected, 1.0)), labels


def temporal_components(rows, lags=LAGS, context=None):
    expected, scale, _ = context if context is not None else temporal_context(rows, lags)
    observed = []
    for lag in lags:
        counts = np.bitwise_and(rows[:-lag], rows[lag:]).sum(axis=0, dtype=np.float64)
        observed.extend(counts)
        observed.append(float(counts.sum()))
    observed = np.asarray(observed)
    return observed, expected, np.abs(observed - expected) / scale


def mc_pvalue(score, maxima):
    """Conservative Monte Carlo rank p-value, including ties and never zero."""
    values = np.asarray(maxima, dtype=np.float64)
    if (values.ndim != 1 or not len(values) or not np.isfinite(values).all() or
            not np.isfinite(score)):
        raise ValueError("Need a finite score and nonempty finite reference maxima")
    tolerance = 100 * np.finfo(np.float64).eps * max(1.0, abs(float(score)))
    return (1 + int(np.count_nonzero(values >= score - tolerance))) / (len(values) + 1)


def calibrate_uniform(n, resamples=1999, seed=42, cache_dir=None, progress=None):
    _positive_integer(n, "n")
    _positive_integer(resamples, "resamples")
    metadata = {"version": VERSION, "n": int(n), "resamples": int(resamples),
                "seed": int(seed), "numpy": np.__version__}
    identity = hashlib.sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()[:24]
    path = Path(cache_dir) / f"uniform-{identity}.npz" if cache_dir is not None else None
    if path is not None and path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if json.loads(str(saved["metadata"].item())) != metadata:
                raise ValueError("Uniform calibration metadata mismatch")
            references = {name: saved[name].copy() for name in ("numbers", "pairs")}
        if any(a.shape != (resamples,) or not np.isfinite(a).all() for a in references.values()):
            raise ValueError("Invalid uniform calibration cache")
        return references, {**metadata, "cache_reused": True, "identity": identity}
    rng = np.random.default_rng(np.random.SeedSequence([seed, n, 0]))
    references = {name: np.empty(resamples, dtype=np.float64) for name in ("numbers", "pairs")}
    for index in range(resamples):
        components = uniform_components(sample_uniform(n, rng))
        for name in references:
            references[name][index] = components[name][2].max()
        if progress and (index + 1) % max(1, resamples // 10) == 0:
            progress(f"Uniform calibration {index + 1}/{resamples}")
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as handle:
                temporary = Path(handle.name)
            np.savez_compressed(temporary, metadata=json.dumps(metadata, sort_keys=True), **references)
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return references, {**metadata, "cache_reused": False, "identity": identity}


def permutation_maxima(rows, resamples=1999, seed=42, lags=LAGS):
    _positive_integer(resamples, "resamples")
    context = temporal_context(rows, lags)
    rng = np.random.default_rng(seed)
    maxima = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        shuffled = rows[rng.permutation(len(rows))]
        maxima[index] = temporal_components(shuffled, lags, context)[2].max()
    return maxima


def summarize_family(components, references, labels, family_count, alpha=0.05, top=5):
    observed, expected, scores = components
    p = mc_pvalue(float(scores.max()), references)
    adjusted = min(1.0, family_count * p)
    chosen = np.argsort(-scores, kind="stable")[:top]
    return {"hypotheses": len(scores), "maximum_scaled_deviation": float(scores.max()),
            "maxstat_pvalue": p, "overall_adjusted_pvalue": adjusted,
            "rejected_after_correction": adjusted <= alpha,
            "top_features": [{"feature": labels[int(index)],
                              "count": int(observed[index]), "expected_count": float(expected[index]),
                              "scaled_deviation": float(scores[index]),
                              "overall_maxstat_adjusted_pvalue": min(1.0, family_count * mc_pvalue(scores[index], references))}
                             for index in chosen]}


def audit_rows(rows, references, *, resamples=1999, seed=42, family_count=6,
               alpha=0.05, lags=LAGS):
    rows = validate_rows(rows)
    uniform = uniform_components(rows)
    families = {
        "numbers": summarize_family(uniform["numbers"], references["numbers"],
                                    list(range(1, 50)), family_count, alpha),
        "pairs": summarize_family(uniform["pairs"], references["pairs"],
                                  [[int(a + 1), int(b + 1)] for a, b in zip(PAIR_I, PAIR_J)], family_count, alpha),
    }
    temporal = temporal_components(rows, lags)
    time_reference = permutation_maxima(rows, resamples, seed, lags)
    families["ordering"] = summarize_family(temporal, time_reference,
                                            temporal_context(rows, lags)[2], family_count, alpha)
    return families


def wilson_interval(successes, trials):
    _positive_integer(trials, "trials")
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    radius = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return [max(0.0, float(center - radius)), min(1.0, float(center + radius))]


def sensitivity(n, references, *, trials=64, permutations=1999, seed=42,
                family_count=6, alpha=0.05, lags=LAGS, progress=None):
    """Independent planted alternatives; intervals condition on shared calibration.

    Uses the same max-statistic and across-family correction as the real audit.
    Temporal controls use their own conditional permutation tests. Numerical
    Monte Carlo resolution and finite trial counts limit sensitivity estimates.
    """
    _positive_integer(trials, "trials")
    _positive_integer(permutations, "permutations")
    if family_count / (permutations + 1) > alpha:
        raise ValueError("More power permutations are needed for the requested corrected significance level")
    levels = {"number": (0.0, 0.01, 0.025, 0.05),
              "pair": (0.0, 0.025, 0.05, 0.10),
              "temporal": (0.0, 0.005, 0.01, 0.025)}
    results = []
    for kind_index, (kind, strengths) in enumerate(levels.items()):
        for level_index, strength in enumerate(strengths):
            successes = 0
            for trial in range(trials):
                rng = np.random.default_rng(np.random.SeedSequence([seed, n, 2, kind_index, level_index, trial]))
                rows = sample_planted(n, kind, strength, rng)
                if kind == "temporal":
                    score = temporal_components(rows, lags)[2].max()
                    null = permutation_maxima(rows, permutations,
                                              np.random.SeedSequence([seed, n, 3, level_index, trial]), lags)
                else:
                    family = "numbers" if kind == "number" else "pairs"
                    score = uniform_components(rows)[family][2].max()
                    null = references[family]
                successes += family_count * mc_pvalue(score, null) <= alpha
                if progress and kind == "temporal" and (trial + 1) % max(1, trials // 4) == 0:
                    progress(f"Sensitivity {kind}, strength={strength}: {trial + 1}/{trials} trials")
            details = {"alternative": kind, "mixture_strength": strength, "trials": trials,
                       "detections": int(successes), "detection_rate": successes / trials,
                       "conditional_wilson_95_interval": wilson_interval(successes, trials)}
            if kind == "number":
                details["number_1_inclusion_probability"] = P_NUMBER + strength * (1 - P_NUMBER)
            elif kind == "pair":
                details["pair_1_2_probability"] = P_PAIR + strength * (P_NUMBER - P_PAIR)
                details["all_number_marginals"] = P_NUMBER
            else:
                details["copy_previous_probability"] = strength
            results.append(details)
            if progress:
                progress(f"Sensitivity {kind}, strength={strength}: {successes}/{trials}")
    return {"draw_count": n, "trials_per_level": trials,
            "target": "Audit detectors, not neural-network training or ticket profitability",
            "ordering_permutations_per_trial": permutations,
            "note": "Synthetic controls only; zero strength estimates size for that family, not forecasting skill. Wilson intervals condition on the shared uniform calibration and do not include its Monte Carlo uncertainty.",
            "levels": results}


def run(csv, *, games=(1, 2), resamples=1999, trials=64, power_permutations=None,
        seed=42, alpha=0.05, lags=LAGS, cache_dir="models/sportka_audit", power=True, progress=None):
    if not games or len(set(games)) != len(games) or any(game not in (1, 2) for game in games):
        raise ValueError("Choose distinct games from 1 and 2")
    _positive_integer(resamples, "resamples")
    if not 0 < alpha < 1 or seed < 0:
        raise ValueError("alpha must be in (0,1) and seed nonnegative")
    family_count = 3 * len(games)
    if family_count / (resamples + 1) > alpha:
        raise ValueError("More resamples are needed for the requested corrected significance level")
    if power:
        _positive_integer(trials, "trials")
        power_permutations = resamples if power_permutations is None else power_permutations
        _positive_integer(power_permutations, "power_permutations")
        if family_count / (power_permutations + 1) > alpha:
            raise ValueError("More power permutations are needed for the requested corrected significance level")
    source = Path(csv)
    source_bytes = source.read_bytes()
    report = {"version": VERSION, "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
              "seed": seed, "alpha": alpha, "resamples": resamples, "lags": list(lags),
              "families_in_correction": family_count,
              "minimum_overall_adjusted_pvalue": family_count / (resamples + 1),
              "nulls": {"numbers_pairs": "Independent uniform six-element subsets of 1..49; main balls only",
                        "ordering": "Conditional row exchangeability, preserving the complete observed draw multiset"},
              "correction": "Maximum absolute scaled deviation within each family; Bonferroni across all requested draw/family combinations",
              "limitations": ["Historical diagnostic, not a prediction model or proof of a betting advantage",
                              "Pair deviations may be explained by marginal bias; rejection does not identify a mechanism",
                              "Ordering rejection can reflect regime changes, data artifacts or dependence; it is not automatically useful for prediction",
                              "Correction covers only the prespecified families/lags in this invocation, not all prior or future analyses",
                              "Non-rejection does not prove randomness or exclude weak, different or time-varying signals"],
              "games": {}, "sensitivity": {}}
    calibrations = {}
    for game in games:
        df = load_sportka_csv(source, game=game)
        rows = validate_rows(base_features(df))
        temporal_context(rows, lags)  # validate before expensive calibration
        n = len(rows)
        if n not in calibrations:
            calibrations[n] = calibrate_uniform(n, resamples, seed, cache_dir, progress)
        references, calibration = calibrations[n]
        if progress:
            progress(f"Auditing game {game}: {n} draws")
        families = audit_rows(rows, references, resamples=resamples,
                              seed=np.random.SeedSequence([seed, n, 1, game]),
                              family_count=family_count, alpha=alpha, lags=lags)
        report["games"][str(game)] = {"draw_count": n,
            "date_range": [str(df.date.min().date()), str(df.date.max().date())],
            "uniform_calibration": calibration, "families": families}
        if power and str(n) not in report["sensitivity"]:
            report["sensitivity"][str(n)] = sensitivity(
                n, references, trials=trials, permutations=power_permutations,
                seed=seed, family_count=family_count, alpha=alpha, lags=lags, progress=progress)
    if source.read_bytes() != source_bytes:
        raise ValueError("CSV changed during the audit; rerun with a stable source")
    rejected = any(family["rejected_after_correction"]
                   for game in report["games"].values() for family in game["families"].values())
    report["verdict"] = "HISTORICAL_DEVIATION_REQUIRES_REPLICATION" if rejected else "NO_CORRECTED_DEVIATION_DETECTED"
    report["prediction_edge"] = "NOT_ESTABLISHED"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--game", choices=("both", "1", "2"), default="both")
    parser.add_argument("--resamples", type=int, default=1999)
    parser.add_argument("--trials", type=int, default=64)
    parser.add_argument("--power-permutations", type=int,
                        help="Permutation count for sensitivity trials (default: same as --resamples)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--lags", type=int, nargs="+", default=list(LAGS))
    parser.add_argument("--cache-dir", default="models/sportka_audit")
    parser.add_argument("--no-power", action="store_true")
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()
    try:
        report = run(args.csv, games=(1, 2) if args.game == "both" else (int(args.game),),
                     resamples=args.resamples, trials=args.trials, power_permutations=args.power_permutations,
                     seed=args.seed, alpha=args.alpha, lags=tuple(args.lags), cache_dir=args.cache_dir,
                     power=not args.no_power,
                     progress=(lambda message: print(message, file=sys.stderr, flush=True)) if args.progress else None)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
