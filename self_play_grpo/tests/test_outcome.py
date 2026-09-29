import math

import pytest

from self_play_grpo.rewards.outcome import (
    fixed_outcome_advantages,
    fractional_results_from_utilities,
    outcome_advantages,
)


def test_decisive_four_player_advantages_match_closed_form() -> None:
    expected = (math.sqrt(3), -1 / math.sqrt(3), -1 / math.sqrt(3), -1 / math.sqrt(3))
    actual = outcome_advantages((1.0, 0.0, 0.0, 0.0))
    assert actual == pytest.approx(expected)
    assert fixed_outcome_advantages((1.0, 0.0, 0.0, 0.0)) == pytest.approx(expected)


def test_uniform_draw_has_zero_advantage() -> None:
    assert outcome_advantages((0.25, 0.25, 0.25, 0.25)) == (0.0, 0.0, 0.0, 0.0)


def test_open_spiel_utility_conversion_preserves_canonical_vector_order() -> None:
    utilities = (-1 / 3, 1.0, -1 / 3, -1 / 3)
    assert fractional_results_from_utilities(utilities) == pytest.approx((0.0, 1.0, 0.0, 0.0))
    assert fractional_results_from_utilities((0.0, 0.0, 0.0, 0.0)) == (0.25,) * 4

