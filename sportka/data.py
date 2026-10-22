"""Validated local Sportka CSV loading, without network or training side effects."""

import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from sportka.rules import MAIN_COUNT, POOL_SIZE, UNIFORM_PROBABILITY, validate_numbers


def read_history(path):
    """Read Allwyn/Sazka's semicolon export in ascending date order.

    Positions: date, year, week, weekday, six main balls, bonus, optionally
    another six main balls and bonus. Invalid rows fail with a line number.
    """
    source = Path(path)
    try:
        text = source.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        text = source.read_text(encoding="cp1250")
    reader = csv.reader(text.splitlines(), delimiter=";")
    next(reader, None)  # export header
    records = []
    for line, row in enumerate(reader, start=2):
        if not row or not any(cell.strip() for cell in row):
            continue
        try:
            date_text = "".join(row[0].split())
            try:
                date = datetime.strptime(date_text, "%d.%m.%Y").date()
            except ValueError:
                date = datetime.strptime(date_text, "%Y-%m-%d").date()
            first = [int(x) for x in row[4:10]]
            bonus_first = int(row[10])
            validate_numbers(first)
            validate_numbers([*first, bonus_first], MAIN_COUNT + 1)
            second, bonus_second = None, None
            if len(row) > 11 and any(x.strip() for x in row[11:18]):
                second = [int(x) for x in row[11:17]]
                bonus_second = int(row[17])
                validate_numbers(second)
                validate_numbers([*second, bonus_second], MAIN_COUNT + 1)
            records.append(dict(date=date, first=first, second=second,
                                bonus_first=bonus_first, bonus_second=bonus_second))
        except (ValueError, IndexError) as exc:
            raise ValueError(f"{source}: invalid draw on CSV line {line}: {exc}") from exc
    if not records:
        raise ValueError(f"{source}: no draws found")
    records.sort(key=lambda r: r["date"])
    if len({r["date"] for r in records}) != len(records):
        raise ValueError(f"{source}: duplicate draw dates")
    return records


def load_sportka_csv(path, game=1):
    """Return date, draw_index, six main numbers and a separate bonus column."""
    if game not in (1, 2):
        raise ValueError("game must be 1 or 2")
    name = "first" if game == 1 else "second"
    rows = []
    for record in read_history(path):
        if record[name] is None:
            raise ValueError(f"{path}: missing second draw on {record['date']}")
        rows.append({"date": pd.Timestamp(record["date"]),
                     "numbers": record[name], "bonus": record[f"bonus_{name}"]})
    df = pd.DataFrame(rows)
    df["draw_index"] = np.arange(len(df))
    return df


def date_to_x(date):
    """Legacy five date features; ephem is needed only by the optional neural path."""
    import ephem
    previous = ephem.previous_new_moon(date)
    following = ephem.next_new_moon(date)
    lunation = (ephem.Date(date) - previous) / (following - previous)
    return np.array([date.day / 31, date.month / 12, date.year / 2019,
                     date.weekday() / 6, lunation])


class draw:
    """One date's two draws. Labels contain the six main numbers only."""

    def __init__(self, record, draw_history, index):
        self.__dict__.update(record)
        self.draw_history = draw_history
        self.index = index

    @staticmethod
    def _binary(numbers):
        out = np.zeros(POOL_SIZE, dtype=np.float32)
        out[np.array(numbers) - 1] = 1
        return out

    @property
    def x_train(self):
        return date_to_x(self.date)

    @property
    def y_train_1(self):
        return self._binary(self.first)

    @property
    def y_train_2(self):
        return self._binary(self.second)

    def _history(self, game):
        if self.index == 0:
            return np.full(POOL_SIZE, UNIFORM_PROBABILITY, dtype=np.float32)
        previous = self.draw_history.draws[self.index - 1]
        return previous.y_train_1 if game == 1 else previous.y_train_2

    @property
    def x_train_history_1(self):
        return self._history(1)

    @property
    def x_train_history_2(self):
        return self._history(2)


class draw_history:
    """Load an existing file once. Downloading is always an explicit action."""

    def __init__(self, path="/tmp/sportka.csv"):
        records = read_history(path)
        if any(r["second"] is None for r in records):
            raise ValueError("The legacy two-draw model requires both draws in the CSV")
        self.draws = [draw(record, self, i) for i, record in enumerate(records)]
