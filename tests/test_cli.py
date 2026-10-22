"""Exercise executable offline flows and explicit error paths."""

import json
import subprocess
import sys

import pytest


def run_cli(*args):
    return subprocess.run([sys.executable, "sportka_launcher.py", *args],
                          text=True, capture_output=True, timeout=30)


def test_predict_is_offline_and_generates_six_number_tickets():
    result = run_cli("predict", "--lines", "8", "--seed", "1", "--json")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["coverage"]["unique_lines"] == 8
    assert all(len(line) == 6 for line in report["tickets"])


def test_explicit_synthetic_backtest_labels_source_and_all_targets():
    result = run_cli("backtest", "--synthetic", "40")
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["source"]["kind"] == "synthetic_null"
    assert sum(report["split_pairs"].values()) == 39
    assert "Not estimated" in report["profit"]


@pytest.mark.parametrize("args", [("backtest", "--csv", "/does/not/exist.csv"),
                                 ("predict", "--lines", "8", "--budget", "10", "--cost-per-line", "2"),
                                 ("predict", "--budget", "10"),
                                 ("predict", "--draws", "0"),
                                 ("backtest", "--synthetic", "3")])
def test_invalid_requests_fail_without_synthetic_fallback(args):
    result = run_cli(*args)
    assert result.returncode != 0
    assert "error:" in result.stderr
