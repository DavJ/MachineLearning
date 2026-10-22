"""Independent combinatorial checks for main balls, bonus tiers and tickets."""

from itertools import combinations
from math import comb

import pytest

from sportka.rules import TOTAL_COMBINATIONS, prize_tier, tier_probabilities, validate_numbers
from sportka.tickets import _unrank, generate_tickets, jackpot_coverage, ticket_cost


def test_combination_unranking_matches_exhaustive_enumeration():
    expected = list(combinations(range(1, 10), 4))
    assert [_unrank(i, 9, 4) for i in range(len(expected))] == expected
    assert _unrank(0) == (1, 2, 3, 4, 5, 6)
    assert _unrank(TOTAL_COMBINATIONS - 1) == (44, 45, 46, 47, 48, 49)


def test_unique_valid_reproducible_tickets():
    tickets = generate_tickets(1000, seed=92)
    assert tickets == generate_tickets(1000, seed=92)
    assert len(set(tickets)) == 1000
    assert all(validate_numbers(line) == line for line in tickets)
    assert any(49 in line for line in tickets)


@pytest.mark.parametrize("count", [0, -1, 1.5, True, 10001])
def test_ticket_count_rejected(count):
    with pytest.raises(ValueError):
        generate_tickets(count)


@pytest.mark.parametrize("line", [[1, 2, 3, 4, 5, 5], [0, 2, 3, 4, 5, 6],
                                 [1, 2, 3, 4, 5, 50], [1, 2, 3, 4, 5, 6, 7],
                                 [True, 2, 3, 4, 5, 6], [1., 2, 3, 4, 5, 6]])
def test_invalid_ticket_rejected(line):
    with pytest.raises(ValueError):
        validate_numbers(line)


def test_bonus_changes_only_five_hit_tier():
    main, bonus = [1, 2, 3, 4, 5, 6], 7
    assert prize_tier(main, main, bonus) == "6"
    assert prize_tier([1, 2, 3, 4, 5, 7], main, bonus) == "5+bonus"
    assert prize_tier([1, 2, 3, 4, 5, 8], main, bonus) == "5"
    assert prize_tier([1, 2, 3, 4, 7, 8], main, bonus) == "4"
    assert prize_tier([1, 2, 7, 8, 9, 10], main, bonus) == "none"
    with pytest.raises(ValueError):
        prize_tier(main, main, 6)


def test_exact_tier_probabilities_and_partition():
    p = tier_probabilities()
    assert TOTAL_COMBINATIONS == 13_983_816
    assert p["6"] == 1 / comb(49, 6)
    assert p["5+bonus"] == pytest.approx(6 / comb(49, 6))
    assert p["5"] == pytest.approx(6 * 42 / comb(49, 6))
    assert sum(p.values()) == pytest.approx(1)
    assert p["3"] == pytest.approx(comb(6, 3) * comb(43, 3) / comb(49, 6))


def test_unique_coverage_is_additive_within_draw_and_duplicates_do_not_help():
    a, b = (1, 2, 3, 4, 5, 6), (1, 2, 3, 4, 5, 7)
    report = jackpot_coverage([a, b, a])
    assert report["duplicate_lines"] == 1
    assert report["jackpot_probability"] == pytest.approx(2 / comb(49, 6))
    assert jackpot_coverage([a, b], draws=2)["jackpot_probability"] == pytest.approx(
        2 * (2 / comb(49, 6)) - (2 / comb(49, 6)) ** 2, rel=1e-14
    )
    assert jackpot_coverage([])["one_in"] is None


def test_budget_uses_decimal_and_never_increases_lines():
    assert ticket_cost(3, "0.10", "0.30") == "0.30"
    with pytest.raises(ValueError, match="exceeding budget"):
        ticket_cost(3, "0.10", "0.29")
    for price, budget in [("nan", 100), (1, "inf"), (-1, 100), (1, -1)]:
        with pytest.raises(ValueError):
            ticket_cost(1, price, budget)
