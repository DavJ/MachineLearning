"""Sportka's six main balls and separate bonus ball (one draw).

Rules: https://www.allwyn.cz/loterie/sportka/jak-hrat
Checked 2026-10-22. Šance and Superjackpot are separate from these odds.
"""

from math import comb
from numbers import Integral

POOL_SIZE = 49
MAIN_COUNT = 6
TOTAL_COMBINATIONS = comb(POOL_SIZE, MAIN_COUNT)
UNIFORM_PROBABILITY = MAIN_COUNT / POOL_SIZE


def validate_numbers(numbers, count=MAIN_COUNT):
    """Return a sorted valid combination; never truncate or coerce bad input."""
    values = tuple(numbers)
    if (len(values) != count or
            any(isinstance(n, bool) or not isinstance(n, Integral) or
                not 1 <= n <= POOL_SIZE for n in values) or
            len(set(values)) != count):
        raise ValueError(f"Expected {count} distinct integers from 1 to {POOL_SIZE}")
    return tuple(sorted(int(n) for n in values))


def prize_tier(ticket, main_numbers, bonus):
    """Return 6, 5+bonus, 5, 4, 3, or none; bonus never counts as a main hit."""
    ticket = validate_numbers(ticket)
    main = validate_numbers(main_numbers)
    validate_numbers([*main, bonus], MAIN_COUNT + 1)
    hits = len(set(ticket).intersection(main))
    if hits == 5 and bonus in ticket:
        return "5+bonus"
    return str(hits) if hits >= 3 else "none"


def tier_probabilities():
    """Exact marginal tier probabilities for one fixed six-number ticket."""
    hit = {k: comb(MAIN_COUNT, k) * comb(POOL_SIZE - MAIN_COUNT, MAIN_COUNT - k)
           / TOTAL_COMBINATIONS for k in range(MAIN_COUNT + 1)}
    return {
        "6": hit[6],
        "5+bonus": hit[5] / (POOL_SIZE - MAIN_COUNT),
        "5": hit[5] * (POOL_SIZE - MAIN_COUNT - 1) / (POOL_SIZE - MAIN_COUNT),
        "4": hit[4],
        "3": hit[3],
        "none": sum(hit[k] for k in range(3)),
    }
