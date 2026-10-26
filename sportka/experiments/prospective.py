"""Seal future Sportka forecasts and evaluate an unreset, anytime-valid test.

Only the three registered baselines are included. Local hashes check internal
consistency; publish each forecast in external version control before its date.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import date, datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from zoneinfo import ZoneInfo

from sportka.data import read_history
from sportka.rules import MAIN_COUNT, POOL_SIZE, validate_numbers

MODELS = ("frequency_all", "frequency_52", "previous_draw")
GAMES = ("1", "2")
TILTS = (.01, .025, .05, .1, .25, .5, 1.)
PRAGUE = ZoneInfo("Europe/Prague")
PROTOCOL = {
    "version": "sportka-prospective-v1",
    "models": list(MODELS), "games": list(GAMES), "tilts": list(TILTS),
    "pool_size": POOL_SIZE, "main_count": MAIN_COUNT, "alpha": .05,
    "combination": "equal mixture of 42 persistent game/model/tilt processes",
    "selection": "descending count; ties by ascending number; no tuning",
    "timezone": "Europe/Prague", "minimum_history": 52,
}


def digest(value):
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def hit_probabilities(pool=POOL_SIZE, count=MAIN_COUNT):
    """Exact intersection law of two count-subsets of a pool."""
    if not (type(pool) is int and type(count) is int and 1 <= count <= pool):
        raise ValueError("Require integer 1 <= count <= pool")
    denominator = math.comb(pool, count)
    return [math.comb(count, h) * math.comb(pool - count, count - h) / denominator
            if 0 <= count - h <= pool - count else 0.
            for h in range(count + 1)]


def log_mean_exp(values):
    high = max(values)
    return high + math.log(math.fsum(math.exp(x - high) for x in values) / len(values))


def log_factor(hits, tilt, pool=POOL_SIZE, count=MAIN_COUNT):
    """An exponential likelihood factor with null conditional mean exactly 1."""
    if isinstance(hits, bool) or not isinstance(hits, int) or not 0 <= hits <= count:
        raise ValueError("Invalid main-ball hit count")
    if not math.isfinite(tilt) or tilt <= 0:
        raise ValueError("The positive tilt must be finite")
    terms = [math.log(p) + tilt * h for h, p in enumerate(hit_probabilities(pool, count))
             if p > 0]
    high = max(terms)
    log_mgf = high + math.log(math.fsum(math.exp(x - high) for x in terms))
    return tilt * hits - log_mgf


def _clock(now=None):
    now = datetime.now(timezone.utc) if now is None else now
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("An aware UTC timestamp is required")
    return now.astimezone(timezone.utc)


def _parse_time(value):
    instant = datetime.fromisoformat(value)
    if instant.tzinfo is None or instant.utcoffset() != timezone.utc.utcoffset(instant):
        raise ValueError("Ledger timestamps must include UTC offset +00:00")
    return instant


def _main(record):
    if record["second"] is None:
        raise ValueError("Both main draws are required")
    return {"1": list(validate_numbers(record["first"])),
            "2": list(validate_numbers(record["second"]))}


def _prefix(records):
    return [{"date": r["date"].isoformat(), "main": _main(r),
             "bonus": [r["bonus_first"], r["bonus_second"]]} for r in records]


def forecasts(records):
    """Deterministic past-only recipes, separately for each main draw."""
    if len(records) < PROTOCOL["minimum_history"]:
        raise ValueError("At least 52 complete historical dates are required")
    output = {model: {} for model in MODELS}
    for game in GAMES:
        field = "first" if game == "1" else "second"
        for model, history in (("frequency_all", records), ("frequency_52", records[-52:])):
            counts = [0] * POOL_SIZE
            for record in history:
                for number in validate_numbers(record[field]):
                    counts[number - 1] += 1
            ranked = sorted(range(1, POOL_SIZE + 1), key=lambda n: (-counts[n - 1], n))
            output[model][game] = sorted(ranked[:MAIN_COUNT])
        output["previous_draw"][game] = list(validate_numbers(records[-1][field]))
    return output


def _check_main(value):
    if set(value) != set(GAMES):
        raise ValueError("Both games must be present exactly once")
    for game in GAMES:
        if value[game] != list(validate_numbers(value[game])):
            raise ValueError("Main numbers must be sorted")


def _check_hash(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("Invalid SHA-256 digest")


def validate_ledger(ledger):
    """Check the complete chain and one-forecast/one-result state machine."""
    if set(ledger) != {"protocol", "events"} or ledger["protocol"] != PROTOCOL:
        raise ValueError("Unrecognised or modified prospective protocol")
    previous, pending, last_time, last_target = digest(PROTOCOL), None, None, None
    for event in ledger["events"]:
        payload = {k: v for k, v in event.items() if k != "hash"}
        if event["previous"] != previous or event["hash"] != digest(payload):
            raise ValueError("Ledger hash chain has been modified")
        instant = _parse_time(event["at_utc"])
        if last_time is not None and instant < last_time:
            raise ValueError("Ledger clock moved backwards")
        target = date.fromisoformat(event["target_date"])
        if event["type"] == "forecast":
            if pending is not None or (last_target is not None and target <= last_target):
                raise ValueError("Resolve the pending forecast; targets must increase")
            if target <= instant.astimezone(PRAGUE).date():
                raise ValueError("Forecasts must be sealed before the target calendar day")
            source = event["source"]
            if (type(source["count"]) is not int or source["count"] < 52 or
                    not date.fromisoformat(source["through"]) <= instant.astimezone(PRAGUE).date() < target):
                raise ValueError("Invalid historical source cutoff")
            for key in ("csv_sha256", "prefix_sha256"):
                _check_hash(source[key])
            _check_hash(event["predictor_code_sha256"])
            if set(event["predictions"]) != set(MODELS):
                raise ValueError("The registered models cannot be added or removed")
            for prediction in event["predictions"].values():
                _check_main(prediction)
            pending = event
        elif event["type"] == "result":
            if (pending is None or event["forecast_hash"] != pending["hash"] or
                    event["target_date"] != pending["target_date"] or
                    instant.astimezone(PRAGUE).date() < target):
                raise ValueError("Result does not resolve its sealed target")
            _check_main(event["observed"])
            _check_hash(event["source_csv_sha256"])
            pending = None
        else:
            raise ValueError("Unknown ledger event type")
        previous, last_time, last_target = event["hash"], instant, target
    return pending


def load_ledger(path):
    with Path(path).open(encoding="utf-8") as stream:
        ledger = json.load(stream)
    validate_ledger(ledger)
    return ledger


def _read_source(path, now):
    path = Path(path)
    source_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    records = read_history(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != source_hash:
        raise ValueError("Source CSV changed while it was being read")
    _prefix(records)  # Require complete, valid data for both games.
    if records[-1]["date"] > now.astimezone(PRAGUE).date():
        raise ValueError("Source CSV contains future results")
    return records, source_hash


def verify_source(ledger, records):
    """Reject historical revisions, target leakage or changes to a fixed recipe."""
    by_date = {r["date"].isoformat(): r for r in records}
    for event in ledger["events"]:
        if event["type"] == "forecast":
            source = event["source"]
            prefix = records[:source["count"]]
            if (len(prefix) != source["count"] or prefix[-1]["date"].isoformat() != source["through"] or
                    digest(_prefix(prefix)) != source["prefix_sha256"]):
                raise ValueError("Historical source prefix was revised or truncated")
            if forecasts(prefix) != event["predictions"]:
                raise ValueError("Sealed predictions differ from the registered recipe")
        elif event["target_date"] not in by_date or _main(by_date[event["target_date"]]) != event["observed"]:
            raise ValueError("A scored result was removed or revised")


def _append(ledger, payload):
    event = dict(payload, previous=ledger["events"][-1]["hash"] if ledger["events"] else digest(PROTOCOL))
    event["hash"] = digest(event)
    ledger["events"].append(event)
    validate_ledger(ledger)


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    try:
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as exc:
        raise ValueError(f"Ledger is locked: {lock}") from exc
    try:
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _save(ledger, path):
    validate_ledger(ledger)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(ledger, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def issue(csv_path, ledger_path, target_date, *, now=None):
    """Append one forecast for a future Prague calendar day; never replace one."""
    now, path = _clock(now), Path(ledger_path)
    target = date.fromisoformat(target_date)
    if target <= now.astimezone(PRAGUE).date():
        raise ValueError("Target must be after today's date in Europe/Prague")
    with _lock(path):
        ledger = load_ledger(path) if path.exists() else {"protocol": deepcopy(PROTOCOL), "events": []}
        if validate_ledger(ledger) is not None:
            raise ValueError("Resolve the pending forecast before issuing another")
        records, source_hash = _read_source(csv_path, now)
        verify_source(ledger, records)
        if records[-1]["date"] >= target:
            raise ValueError("Target result is already present in the source")
        _append(ledger, {
            "type": "forecast", "at_utc": now.isoformat(), "target_date": target.isoformat(),
            "source": {"count": len(records), "through": records[-1]["date"].isoformat(),
                       "csv_sha256": source_hash, "prefix_sha256": digest(_prefix(records))},
            "predictor_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "predictions": forecasts(records),
        })
        _save(ledger, path)
    return summarize(ledger)


def score(csv_path, ledger_path, *, now=None):
    """Resolve exactly the pending date; do not backfill unsealed observations."""
    now, path = _clock(now), Path(ledger_path)
    with _lock(path):
        ledger = load_ledger(path)
        pending = validate_ledger(ledger)
        if pending is None:
            raise ValueError("No pending forecast to score")
        if now < _parse_time(pending["at_utc"]):
            raise ValueError("Current clock precedes the sealed forecast")
        records, source_hash = _read_source(csv_path, now)
        verify_source(ledger, records)
        target = date.fromisoformat(pending["target_date"])
        actual = next((r for r in records if r["date"] == target), None)
        if actual is None:
            if records[-1]["date"] > target:
                raise ValueError("Target date is missing despite later source results")
            return summarize(ledger)
        _append(ledger, {
            "type": "result", "at_utc": now.isoformat(), "target_date": target.isoformat(),
            "forecast_hash": pending["hash"], "observed": _main(actual),
            "source_csv_sha256": source_hash,
        })
        _save(ledger, path)
    return summarize(ledger)


def summarize(ledger):
    pending = validate_ledger(ledger)
    log_values = {(model, game, tilt): 0. for model in MODELS for game in GAMES for tilt in TILTS}
    hits_total = {(model, game): 0 for model in MODELS for game in GAMES}
    max_log, trajectory, current = 0., [], None
    for event in ledger["events"]:
        if event["type"] == "forecast":
            current = event
            continue
        hits = {model: {} for model in MODELS}
        for model in MODELS:
            for game in GAMES:
                hit = len(set(current["predictions"][model][game]) & set(event["observed"][game]))
                hits[model][game] = hit
                hits_total[model, game] += hit
                for tilt in TILTS:
                    log_values[model, game, tilt] += log_factor(hit, tilt)
        log_e = log_mean_exp(list(log_values.values()))
        max_log = max(max_log, log_e)
        trajectory.append({"target_date": event["target_date"], "hits": hits, "log_e_value": log_e})
    log_e = log_mean_exp(list(log_values.values()))
    n = len(trajectory)
    return {
        "protocol": PROTOCOL["version"], "scored_dates": n, "expert_count": len(log_values),
        "null_expected_hits_per_game": MAIN_COUNT ** 2 / POOL_SIZE,
        "log_e_value": log_e, "e_value": math.exp(log_e) if -700 < log_e < 700 else None,
        "running_max_log_e_value": max_log, "log_anytime_p_bound": -max_log,
        "anytime_p_bound": math.exp(-max_log) if max_log < 700 else None,
        "rejection_threshold": 1 / PROTOCOL["alpha"],
        "verdict": "FAIR_DRAW_NULL_REJECTED" if max_log >= -math.log(PROTOCOL["alpha"]) else "NO_PROSPECTIVE_EVIDENCE",
        "prediction_edge": "NOT_ESTABLISHED",
        "external_publication": "MUST_BE_VERIFIED_IN_VERSION_HISTORY",
        "mean_hits": {model: {game: hits_total[model, game] / n if n else None for game in GAMES}
                      for model in MODELS},
        "trajectory": trajectory, "pending_forecast": pending,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("issue", "score", "status"):
        child = sub.add_parser(command)
        child.add_argument("--ledger", required=True)
        if command != "status":
            child.add_argument("--csv", required=True)
        if command == "issue":
            child.add_argument("--target-date", required=True, help="YYYY-MM-DD, after today in Prague")
    args = parser.parse_args(argv)
    try:
        if args.command == "issue":
            result = issue(args.csv, args.ledger, args.target_date)
        elif args.command == "score":
            result = score(args.csv, args.ledger)
        else:
            result = summarize(load_ledger(args.ledger))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    main()
