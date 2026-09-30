"""Four-player terminal result conversion and GRPO-style advantages."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


FOUR_PLAYER_OUTCOME_SCALE = math.sqrt(3.0) / 4.0


def fractional_results_from_utilities(utilities: Sequence[float]) -> tuple[float, ...]:
    """Map OpenSpiel winner/loser utilities to a unit-sum result vector."""

    n = len(utilities)
    if n < 2:
        raise ValueError("At least two utilities are required")
    if all(abs(float(value)) < 1e-12 for value in utilities):
        return tuple(1.0 / n for _ in utilities)
    results = tuple(((n - 1) * float(value) + 1.0) / n for value in utilities)
    if not math.isclose(sum(results), 1.0, abs_tol=1e-9):
        raise ValueError(f"Utilities do not map to a unit-sum result: {utilities}")
    return results


def outcome_advantages(results: Sequence[float]) -> tuple[float, ...]:
    """Population-standardize the seats within one complete match."""

    if len(results) != 4:
        raise ValueError("The experiment requires exactly four seat results")
    values = tuple(float(value) for value in results)
    mean = sum(values) / 4.0
    variance = sum((value - mean) ** 2 for value in values) / 4.0
    if variance <= 1e-24:
        return (0.0, 0.0, 0.0, 0.0)
    std = math.sqrt(variance)
    return tuple((value - mean) / std for value in values)


def fixed_outcome_advantages(results: Sequence[float]) -> tuple[float, ...]:
    """Equivalent closed form for sole-winner-or-uniform-draw outcomes."""

    if len(results) != 4:
        raise ValueError("The experiment requires exactly four seat results")
    return tuple((float(value) - 0.25) / FOUR_PLAYER_OUTCOME_SCALE for value in results)


def validate_result_vector(results: Iterable[float]) -> tuple[float, float, float, float]:
    values = tuple(float(value) for value in results)
    if len(values) != 4:
        raise ValueError("A match result must have four entries")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("Fractional results must be in [0, 1]")
    if not math.isclose(sum(values), 1.0, abs_tol=1e-9):
        raise ValueError("Fractional results must sum to one")
    return values  # type: ignore[return-value]

