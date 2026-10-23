"""Exact null calibration and a source-anchored future-forecast lifecycle."""

from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import itertools
import json
import math
import subprocess
import sys

import pytest

from sportka.experiments import prospective as p
from tests.test_sportka_data import HEADER

UTC = timezone.utc
ISSUED = datetime(2025, 3, 1, 12, tzinfo=UTC)
SCORED = datetime(2025, 3, 3, 12, tzinfo=UTC)
TARGET = "2025-03-02"


def csv_text(rows):
    text = HEADER
    for day, first, second in rows:
        bonus1 = next(n for n in range(49, 0, -1) if n not in first)
        bonus2 = next(n for n in range(49, 0, -1) if n not in second)
        values = [day.strftime("%d.%m.%Y"), day.year, 1, day.weekday(),
                  *first, bonus1, *second, bonus2]
        text += ";".join(map(str, values)) + "\n"
    return text


@pytest.fixture
def history(tmp_path):
    rows = [(date(2025, 1, 1) + timedelta(days=i), list(range(1, 7)), list(range(10, 16)))
            for i in range(52)]
    csv = tmp_path / "history.csv"
    csv.write_text(csv_text(rows))
    return csv, tmp_path / "ledger.json", rows


def add_result(csv, rows, first=(1, 2, 3, 4, 5, 6), second=(10, 11, 12, 13, 14, 15)):
    csv.write_text(csv_text([*rows, (date.fromisoformat(TARGET), list(first), list(second))]))


def test_exact_hit_distribution_matches_enumerated_small_lottery():
    counts = Counter(len(set(draw) & {0, 1, 2}) for draw in itertools.combinations(range(8), 3))
    total = math.comb(8, 3)
    assert p.hit_probabilities(8, 3) == pytest.approx([counts[h] / total for h in range(4)])
    fair = p.hit_probabilities()
    assert sum(fair) == pytest.approx(1)
    assert sum(h * prob for h, prob in enumerate(fair)) == pytest.approx(36 / 49)
    assert fair[6] == 1 / math.comb(49, 6)


@pytest.mark.parametrize("tilt", p.TILTS)
def test_every_factor_has_unit_null_expectation_even_after_past_adaptation(tilt):
    # Pick a *different* second forecast after each possible first result.
    # Enumerate all two-round paths, not independent Bernoulli ball labels.
    draws = list(itertools.combinations(range(8), 3))
    products = []
    for first in draws:
        h1 = len(set(first) & {0, 1, 2})
        forecast2 = {(x + 1) % 8 for x in first}
        for second in draws:
            h2 = len(set(second) & forecast2)
            products.append(math.exp(p.log_factor(h1, tilt, 8, 3) + p.log_factor(h2, tilt, 8, 3)))
    # Python <3.12 sum accumulates enough rounding error across 3,136 paths to
    # exceed this tolerance. Use a precise reference sum on every interpreter.
    assert math.fsum(products) / len(products) == pytest.approx(1., abs=2e-14)


def test_mixture_is_valid_with_perfectly_dependent_games_and_models():
    # Identical outcomes across all six game/model streams: averaging remains
    # calibrated; multiplying those six streams would not be calibrated.
    expected = 0.
    for h, probability in enumerate(p.hit_probabilities()):
        components = [p.log_factor(h, tilt) for _ in range(6) for tilt in p.TILTS]
        expected += probability * math.exp(p.log_mean_exp(components))
    assert expected == pytest.approx(1.)


def test_full_lifecycle_keeps_the_forecast_immutable_and_scores_both_games(history):
    csv, ledger, rows = history
    result = p.issue(csv, ledger, TARGET, now=ISSUED)
    assert result["scored_dates"] == 0
    assert result["e_value"] == result["anytime_p_bound"] == 1
    assert result["expert_count"] == 42
    original = p.load_ledger(ledger)["events"][0]
    assert original["predictions"]["frequency_all"]["1"] == list(range(1, 7))
    # A skipped intervening draw does not get backfilled into the test.
    rows = [*rows, (date(2025, 2, 27), list(range(30, 36)), list(range(35, 41)))]
    add_result(csv, rows)
    result = p.score(csv, ledger, now=SCORED)
    assert p.load_ledger(ledger)["events"][0] == original
    assert result["pending_forecast"] is None
    assert result["scored_dates"] == 1
    assert result["trajectory"][0]["hits"]["previous_draw"] == {"1": 6, "2": 6}
    assert result["prediction_edge"] == "NOT_ESTABLISHED"
    assert result["verdict"] == "FAIR_DRAW_NULL_REJECTED"
    assert result["anytime_p_bound"] <= .05
    assert result["mean_hits"]["frequency_52"] == {"1": 6., "2": 6.}
    with pytest.raises(ValueError, match="No pending"):
        p.score(csv, ledger, now=SCORED)


def test_missing_result_waits_without_mutating_or_inventing_hits(history):
    csv, ledger, _ = history
    p.issue(csv, ledger, TARGET, now=ISSUED)
    before = ledger.read_bytes()
    result = p.score(csv, ledger, now=SCORED)
    assert ledger.read_bytes() == before
    assert result["scored_dates"] == 0
    assert result["pending_forecast"]["target_date"] == TARGET
    assert result["mean_hits"]["previous_draw"]["1"] is None


def test_pending_forecast_cannot_be_replaced_or_deleted_by_issue(history):
    csv, ledger, _ = history
    p.issue(csv, ledger, TARGET, now=ISSUED)
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match="pending"):
        p.issue(csv, ledger, "2025-03-04", now=ISSUED)
    assert ledger.read_bytes() == before


@pytest.mark.parametrize("target", ["2025-02-28", "2025-03-01"])
def test_same_day_and_past_forecasts_are_rejected_before_writing(history, target):
    csv, ledger, _ = history
    with pytest.raises(ValueError, match="after today"):
        p.issue(csv, ledger, target, now=ISSUED)
    assert not ledger.exists()


def test_deadline_uses_prague_day_not_utc_day(history):
    csv, ledger, _ = history
    # UTC is March 1, but Prague is already March 2.
    with pytest.raises(ValueError, match="after today"):
        p.issue(csv, ledger, TARGET, now=datetime(2025, 3, 1, 23, 30, tzinfo=UTC))


def test_historical_revision_is_rejected_without_changing_ledger(history):
    csv, ledger, rows = history
    p.issue(csv, ledger, TARGET, now=ISSUED)
    before = ledger.read_bytes()
    rows[0] = (rows[0][0], list(range(2, 8)), rows[0][2])
    add_result(csv, rows)
    with pytest.raises(ValueError, match="prefix was revised"):
        p.score(csv, ledger, now=SCORED)
    assert ledger.read_bytes() == before


def test_source_format_and_number_order_changes_do_not_break_semantic_prefix(history):
    csv, ledger, rows = history
    p.issue(csv, ledger, TARGET, now=ISSUED)
    rows = [(day, list(reversed(first)), list(reversed(second))) for day, first, second in reversed(rows)]
    add_result(csv, rows)
    assert p.score(csv, ledger, now=SCORED)["scored_dates"] == 1


def test_edited_forecast_or_protocol_is_rejected(history):
    csv, ledger, _ = history
    p.issue(csv, ledger, TARGET, now=ISSUED)
    saved = p.load_ledger(ledger)
    modified = deepcopy(saved)
    modified["events"][0]["predictions"]["previous_draw"]["1"][-1] = 7
    with pytest.raises(ValueError, match="hash chain"):
        p.validate_ledger(modified)
    modified = deepcopy(saved)
    modified["protocol"]["alpha"] = .99
    with pytest.raises(ValueError, match="protocol"):
        p.validate_ledger(modified)


def test_rehashed_non_recipe_forecast_fails_source_verification(history):
    # Local hashes alone cannot prevent a rewrite. Even a rehashed forgery is
    # rejected if it no longer matches the fixed recipe on the locked history.
    csv, ledger, _ = history
    p.issue(csv, ledger, TARGET, now=ISSUED)
    saved = p.load_ledger(ledger)
    event = saved["events"][0]
    event["predictions"]["frequency_all"]["1"][-1] = 7
    event["hash"] = p.digest({k: v for k, v in event.items() if k != "hash"})
    p.validate_ledger(saved)
    with pytest.raises(ValueError, match="registered recipe"):
        p.verify_source(saved, p.read_history(csv))


def test_atomic_write_failure_preserves_previous_ledger(history, monkeypatch):
    csv, ledger, rows = history
    p.issue(csv, ledger, TARGET, now=ISSUED)
    before = ledger.read_bytes()
    add_result(csv, rows)
    def fail_replace(*args):
        raise OSError("simulated disk failure")
    monkeypatch.setattr(p.os, "replace", fail_replace)
    with pytest.raises(OSError, match="disk failure"):
        p.score(csv, ledger, now=SCORED)
    assert ledger.read_bytes() == before
    assert not list(ledger.parent.glob("ledger.json.*"))


def test_concurrent_writer_is_rejected(history):
    csv, ledger, _ = history
    with p._lock(ledger):
        with pytest.raises(ValueError, match="locked"):
            p.issue(csv, ledger, TARGET, now=ISSUED)
    assert not ledger.exists()


def test_no_reset_after_losses_and_no_omitting_a_bad_target(history):
    csv, ledger, rows = history
    p.issue(csv, ledger, TARGET, now=ISSUED)
    add_result(csv, rows, range(30, 36), range(35, 41))
    first = p.score(csv, ledger, now=SCORED)
    assert first["log_e_value"] < 0
    assert first["anytime_p_bound"] == 1
    before = ledger.read_bytes()
    with pytest.raises(ValueError, match="after today"):
        p.issue(csv, ledger, TARGET, now=SCORED)
    assert ledger.read_bytes() == before
    p.issue(csv, ledger, "2025-03-05", now=SCORED)
    pending = p.load_ledger(ledger)
    assert len(pending["events"]) == 3
    assert p.summarize(pending)["log_e_value"] == first["log_e_value"]
    # Remove an already scored row from an otherwise extended source.
    csv.write_text(csv_text([*rows, (date(2025, 3, 6), list(range(1, 7)), list(range(10, 16)))]))
    with pytest.raises(ValueError, match="scored result was removed"):
        p.score(csv, ledger, now=datetime(2025, 3, 7, tzinfo=UTC))


def test_missing_target_among_later_dates_is_an_error(history):
    csv, ledger, rows = history
    p.issue(csv, ledger, TARGET, now=ISSUED)
    csv.write_text(csv_text([*rows, (date(2025, 3, 3), list(range(1, 7)), list(range(10, 16)))]))
    with pytest.raises(ValueError, match="Target date is missing"):
        p.score(csv, ledger, now=SCORED)


def test_source_with_future_results_is_rejected(history):
    csv, ledger, rows = history
    add_result(csv, rows)
    with pytest.raises(ValueError, match="future results"):
        p.issue(csv, ledger, "2025-03-04", now=ISSUED)
    assert not ledger.exists()


def test_running_evidence_bound_retains_a_crossing_after_a_later_loss(history):
    csv, ledger, rows = history
    p.issue(csv, ledger, TARGET, now=ISSUED)
    add_result(csv, rows)
    first = p.score(csv, ledger, now=SCORED)
    p.issue(csv, ledger, "2025-03-05", now=SCORED)
    rows += [(date(2025, 3, 2), list(range(1, 7)), list(range(10, 16))),
             (date(2025, 3, 5), list(range(30, 36)), list(range(35, 41)))]
    csv.write_text(csv_text(rows))
    second = p.score(csv, ledger, now=datetime(2025, 3, 6, tzinfo=UTC))
    assert second["scored_dates"] == 2
    assert second["e_value"] < 20 < first["e_value"]
    assert second["anytime_p_bound"] == first["anytime_p_bound"]
    assert second["verdict"] == "FAIR_DRAW_NULL_REJECTED"


def test_bonus_ball_is_never_scored_as_a_main_hit(history):
    csv, ledger, rows = history
    rows = [(day, [1, 2, 3, 4, 5, 49], second) for day, _, second in rows]
    csv.write_text(csv_text(rows))
    p.issue(csv, ledger, TARGET, now=ISSUED)
    # The result writer uses bonus 49, which is in the predicted first set.
    add_result(csv, rows, range(10, 16), range(30, 36))
    result = p.score(csv, ledger, now=SCORED)
    assert result["trajectory"][0]["hits"]["previous_draw"] == {"1": 0, "2": 0}


def test_exponentially_large_evidence_keeps_finite_logarithms(history):
    csv, ledger, _ = history
    p.issue(csv, ledger, TARGET, now=ISSUED)
    saved = p.load_ledger(ledger)
    source = saved["events"][0]
    # Build an internally consistent long chain without external publication;
    # this is a numeric stress fixture, not actual prospective evidence.
    for i in range(200):
        if i:
            payload = {k: v for k, v in source.items() if k not in {"hash", "previous"}}
            payload.update(at_utc=(ISSUED + timedelta(days=2*i)).isoformat(),
                           target_date=(date.fromisoformat(TARGET) + timedelta(days=2*i)).isoformat())
            p._append(saved, payload)
        current = saved["events"][-1]
        p._append(saved, {"type": "result", "at_utc": (SCORED + timedelta(days=2*i)).isoformat(),
                          "target_date": current["target_date"], "forecast_hash": current["hash"],
                          "observed": current["predictions"]["previous_draw"], "source_csv_sha256": "0" * 64})
    summary = p.summarize(saved)
    assert summary["log_e_value"] > 700
    assert summary["e_value"] is None
    assert summary["anytime_p_bound"] is None
    assert summary["log_anytime_p_bound"] < -700
    json.dumps(summary, allow_nan=False)


def test_cli_status_round_trip_and_missing_file_errors(history):
    csv, ledger, _ = history
    p.issue(csv, ledger, TARGET, now=ISSUED)
    result = subprocess.run([sys.executable, "-m", "sportka.experiments.prospective", "status",
                             "--ledger", str(ledger)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["scored_dates"] == 0
    result = subprocess.run([sys.executable, "-m", "sportka.experiments.prospective", "score",
                             "--csv", "/no/source.csv", "--ledger", str(ledger)],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode != 0
    assert "error:" in result.stderr
    assert not result.stdout
