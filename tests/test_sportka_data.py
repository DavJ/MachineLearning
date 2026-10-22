"""CSV regressions: bonus isolation, chronology, malformed data and imports."""

import subprocess
import sys

import numpy as np
import pytest

from sportka.data import draw_history, load_sportka_csv


HEADER = "date;year;week;weekday;n1;n2;n3;n4;n5;n6;bonus;m1;m2;m3;m4;m5;m6;bonus2\n"
ROW = "1. 1. 2025;2025;1;3;1;2;3;4;5;6;49;10;11;12;13;14;15;48\n"


@pytest.fixture
def history_csv(tmp_path):
    path = tmp_path / "sportka.csv"
    # Deliberately newest first, as exports may arrive in reverse order.
    path.write_text(HEADER + ROW.replace("1. 1. 2025", "8. 1. 2025") + ROW, encoding="utf-8-sig")
    return path


def test_csv_six_main_balls_and_separate_bonus(history_csv):
    first, second = load_sportka_csv(history_csv), load_sportka_csv(history_csv, game=2)
    assert first.numbers.tolist() == [[1, 2, 3, 4, 5, 6]] * 2
    assert first.bonus.tolist() == [49, 49]
    assert second.numbers.iloc[0] == [10, 11, 12, 13, 14, 15]
    assert second.bonus.iloc[0] == 48
    assert first.date.is_monotonic_increasing
    assert first.draw_index.tolist() == [0, 1]


def test_history_is_instance_local_and_binary(history_csv):
    a, b = draw_history(history_csv), draw_history(history_csv)
    assert a.draws is not b.draws
    assert len(a.draws) == len(b.draws) == 2
    assert a.draws[0].y_train_1.sum() == 6
    assert a.draws[0].y_train_1[48] == 0
    np.testing.assert_array_equal(a.draws[1].x_train_history_1, a.draws[0].y_train_1)
    assert a.draws[0].x_train_history_1.sum() == pytest.approx(6)


@pytest.mark.parametrize("contents", [HEADER, HEADER + ROW + ROW,
    HEADER + ROW.replace(";1;2;3;4;5;6;49", ";1;2;3;4;5;5;49"),
    HEADER + ROW.replace(";6;49", ";6;6"),
    HEADER + ROW.replace(";6;49", ";6;50"),
    HEADER + ROW.replace(";6;49", ";6;6.5"),
    HEADER + "broken;row\n"])
def test_bad_history_fails_instead_of_silently_skipping(tmp_path, contents):
    path = tmp_path / "bad.csv"
    path.write_text(contents)
    with pytest.raises(ValueError):
        load_sportka_csv(path)


def test_local_imports_do_not_download_train_or_exit():
    result = subprocess.run([sys.executable, "-c", "import sys; import sportka.learn; import test_core; "
                             "assert 'tensorflow' not in sys.modules; "
                             "assert 'selenium' not in sys.modules"],
                            capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    assert result.stdout == ""
