"""Offline tickets, exact odds, and chronological baseline evaluation."""

import argparse
import hashlib
import json
from pathlib import Path

from sportka.rules import TOTAL_COMBINATIONS, tier_probabilities
from sportka.tickets import generate_tickets, jackpot_coverage, ticket_cost


NOTICE = ("Fair independent draws cannot be predicted from past results. "
          "These tickets have no demonstrated predictive edge. "
          "Jackpot odds below exclude Šance and Superjackpot.")


def backtest(args):
    """Evaluate fixed baselines on untouched chronological validation/test sets."""
    from sportka.evaluation import compute_all_metrics, temporal_feature_split
    from sportka.fixed_cardinality import PROBABILITY_CLIP, VERSION
    from sportka.features import generate_synthetic_data, load_sportka_csv
    from sportka.models import GlobalFreqPredictor, RandomPredictor, RollingFreqPredictor

    if args.csv:
        df = load_sportka_csv(args.csv, game=args.game)
        source = {"kind": "csv", "sha256": hashlib.sha256(Path(args.csv).read_bytes()).hexdigest(),
                  "game": args.game}
    else:
        df = generate_synthetic_data(args.synthetic, seed=args.seed)
        source = {"kind": "synthetic_null", "seed": args.seed}
    groups = ["base", "time", "winding"]
    xtr, ytr, xv, yv, xt, yt = temporal_feature_split(df, groups)
    models = [RandomPredictor(), GlobalFreqPredictor(), RollingFreqPredictor()]
    if args.include_ml:
        from sportka.models import LogisticRegressionModel, SKLEARN_AVAILABLE
        if not SKLEARN_AVAILABLE:
            raise ValueError("--include-ml requires scikit-learn")
        models.append(LogisticRegressionModel())
    results = {}
    for model in models:
        model.fit(xtr, ytr)
        results[model.name] = {
            "validation": compute_all_metrics(yv, model.predict_proba(xv)),
            "test": compute_all_metrics(yt, model.predict_proba(xt)),
        }
    # Difference from the fair-draw probability baseline, on the same held-out
    # draws. A lower BCE alone does not establish a reproducible betting edge.
    baseline = results["random_uniform"]["test"]["bce"]
    for result in results.values():
        result["test_bce_minus_uniform"] = result["test"]["bce"] - baseline
    return {
        "notice": NOTICE,
        "source": source,
        "draw_count": len(df),
        "date_range": [str(df.date.min().date()), str(df.date.max().date())],
        "split_pairs": {"train": len(ytr), "validation": len(yv), "test": len(yt)},
        "expected_random_hits_per_line": 36 / 49,
        "expected_random_jackpot_rate": 1 / TOTAL_COMBINATIONS,
        "selection": "Fixed models; no test-set tuning or automatic betting recommendation",
        "conditional_score_protocol": {
            "version": VERSION,
            "law": "Independent Bernoulli scores conditioned on exactly six main balls",
            "probability_clip": PROBABILITY_CLIP,
            "set_score_units": "Natural log units per complete draw; lower NLL is better",
            "gain": "Uniform set NLL minus model set NLL; positive is descriptive, not evidence of an edge",
            "marginals": "conditional_bce/brier use implied marginals; bce/brier retain raw scores",
        },
        "tie_policy": "Equal probability scores use a fixed numerical ordering for top-k metrics",
        "profit": "Not estimated: actual payouts, costs, and jackpot sharing are not supplied",
        "results": results,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description="Sportka / Allwyn offline tools")
    commands = parser.add_subparsers(dest="command", required=True)
    for command in ("predict", "tickets", "odds"):
        p = commands.add_parser(command, help="Valid tickets and/or exact fair-draw odds")
        p.add_argument("--lines", type=int, default=1)
        p.add_argument("--draws", type=int, default=2, help="Independent main draws (default: both Sportka draws)")
        p.add_argument("--seed", type=int, help="Reproducible test seed; omitted uses OS randomness")
        p.add_argument("--budget", help="Maximum total amount in the same currency as the supplied price")
        p.add_argument("--cost-per-line", help="User-supplied total cost per column, including the intended schedule")
        p.add_argument("--json", action="store_true")
    p = commands.add_parser("backtest", help="Chronological evaluation; no automatic downloads")
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument("--csv", help="Local Allwyn/Sazka semicolon CSV export")
    source.add_argument("--synthetic", type=int, metavar="DRAWS", help="Explicit synthetic null test")
    p.add_argument("--game", type=int, choices=(1, 2), default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--include-ml", action="store_true", help="Also evaluate experimental logistic regression")
    p = commands.add_parser("download", help="Explicitly download and validate Allwyn's public CSV")
    p.add_argument("--directory", default="/tmp")
    p.add_argument("--timeout", type=float, default=30)
    for name in ("gui", "config", "test"):
        commands.add_parser(name)
    p = commands.add_parser("train", help="Optional TensorFlow experiment using a local CSV")
    p.add_argument("--csv", default="/tmp/sportka.csv")
    p.add_argument("--date")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--no-biquaternion", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.command in ("predict", "tickets", "odds"):
            if (args.budget is None) != (args.cost_per_line is None):
                raise ValueError("Supply --budget and --cost-per-line together")
            cost = ticket_cost(args.lines, args.cost_per_line, args.budget) if args.budget is not None else None
            lines = generate_tickets(args.lines, seed=args.seed)
            result = {"notice": NOTICE, "coverage": jackpot_coverage(lines, args.draws),
                      "per_line_per_draw_tier_probabilities": tier_probabilities(),
                      "total_cost": cost}
            if args.command != "odds":
                result["tickets"] = lines
            if args.json:
                print(json.dumps(result, indent=2, allow_nan=False))
            else:
                print(NOTICE)
                for i, line in enumerate(result.get("tickets", []), start=1):
                    print(f"{i:2d}: " + " ".join(f"{n:02d}" for n in line))
                coverage = result["coverage"]
                print(f"Six-main-ball jackpot: 1 in {coverage['one_in']:,.2f} "
                      f"({coverage['unique_lines']} unique lines, {args.draws} independent draws)")
                if cost is not None:
                    print(f"Total supplied cost: {cost}; budget: {args.budget}")
        elif args.command == "backtest":
            print(json.dumps(backtest(args), indent=2, allow_nan=False))
        elif args.command == "download":
            from sportka.download import download_data_from_sazka
            if not 0 < args.timeout <= 300:
                raise ValueError("timeout must be between 0 and 300 seconds")
            print(download_data_from_sazka(args.directory, args.timeout))
        elif args.command == "gui":
            from sportka.gui import main as gui_main
            return gui_main()
        elif args.command == "config":
            from sportka.config import create_default_config
            create_default_config()
        elif args.command == "train":
            from sportka.learn_improved import main as train_main
            options = ["--csv", args.csv, "--epochs", str(args.epochs)]
            if args.date:
                options += ["--date", args.date]
            if args.no_biquaternion:
                options += ["--no-biquaternion"]
            return train_main(options)
        elif args.command == "test":
            from importlib.util import find_spec
            required = ("numpy", "pandas", "sklearn", "requests")
            missing = [name for name in required if find_spec(name) is None]
            for name in required + ("tensorflow", "ephem", "tkinter", "reportlab"):
                suffix = " (optional)" if name not in required else ""
                print(f"{name}{suffix}: {'available' if find_spec(name) else 'missing'}")
            return int(bool(missing))
    except (ValueError, OSError, ImportError) as exc:
        parser.error(str(exc))
    return 0
