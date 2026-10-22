"""Uniform, unique six-number combinations and exact jackpot coverage.

No ranking can improve a line's chance under a fair independent draw.
Distinct lines avoid spending twice on the same jackpot outcome.
"""

from decimal import Decimal, InvalidOperation
from math import comb, expm1, log1p
from random import Random, SystemRandom

from sportka.rules import MAIN_COUNT, POOL_SIZE, TOTAL_COMBINATIONS, validate_numbers


def _unrank(rank, pool_size=POOL_SIZE, count=MAIN_COUNT):
    """Map a lexicographic combination index to its distinct sorted numbers."""
    result = []
    start = 1
    for remaining in range(count, 0, -1):
        for number in range(start, pool_size + 1):
            block = comb(pool_size - number, remaining - 1)
            if rank < block:
                result.append(number)
                start = number + 1
                break
            rank -= block
    return tuple(result)


def generate_tickets(count=1, seed=None):
    """Sample combinations uniformly without replacement, with bounded work.

    A supplied seed is for reproducible experiments. Otherwise SystemRandom
    uses operating-system randomness; model scores are deliberately unused.
    """
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 10_000:
        raise ValueError("count must be an integer between 1 and 10000")
    rng = SystemRandom() if seed is None else Random(seed)
    return [_unrank(rank) for rank in rng.sample(range(TOTAL_COMBINATIONS), count)]


def jackpot_coverage(tickets, draws=1):
    """Exact probability of at least one six-main-ball hit over independent draws.

    Lines within one draw are mutually exclusive jackpot outcomes, so their
    probability is unique_count / C(49,6), not 1 - (1 - 1/C)^line_count.
    Duplicate tickets do not increase coverage. This is not Superjackpot odds.
    """
    if isinstance(draws, bool) or not isinstance(draws, int) or draws < 1:
        raise ValueError("draws must be a positive integer")
    lines = [validate_numbers(line) for line in tickets]
    unique = len(set(lines))
    per_draw = unique / TOTAL_COMBINATIONS
    probability = 1.0 if per_draw == 1 else -expm1(draws * log1p(-per_draw))
    return {"lines": len(lines), "unique_lines": unique,
            "duplicate_lines": len(lines) - unique, "independent_draws": draws,
            "jackpot_probability": probability,
            "one_in": 1 / probability if probability else None}


def ticket_cost(count, cost_per_line, budget):
    """Check a caller-supplied price and limit, without changing ticket count.

    No price is inferred. Extras such as Šance must be included by the caller
    or budgeted separately; this tool never purchases a ticket.
    """
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer")
    try:
        price, limit = Decimal(str(cost_per_line)), Decimal(str(budget))
    except InvalidOperation as exc:
        raise ValueError("Price and budget must be decimal amounts") from exc
    if not price.is_finite() or not limit.is_finite() or price <= 0 or limit < 0:
        raise ValueError("Price must be positive and budget nonnegative, both finite")
    total = count * price
    if total > limit:
        raise ValueError(f"Requested lines cost {total}, exceeding budget {limit}")
    return str(total)
