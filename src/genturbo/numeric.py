"""Small shared numerical summaries used by deterministic result processors."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from statistics import fmean
from typing import Any


def pearson_correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    """Return Pearson correlation, or ``None`` for an undefined comparison."""

    if len(left) < 2 or len(left) != len(right):
        return None
    left_values = [float(value) for value in left]
    right_values = [float(value) for value in right]
    if not all(math.isfinite(value) for value in (*left_values, *right_values)):
        return None
    left_mean = fmean(left_values)
    right_mean = fmean(right_values)
    numerator = sum((a - left_mean) * (b - right_mean) for a, b in zip(left_values, right_values, strict=True))
    left_ss = sum((value - left_mean) ** 2 for value in left_values)
    right_ss = sum((value - right_mean) ** 2 for value in right_values)
    denominator = math.sqrt(left_ss * right_ss)
    return numerator / denominator if denominator > 0.0 else None


def finite_mean_std(values: Iterable[Any], *, ddof: int = 0) -> tuple[float | None, float | None]:
    """Return mean/std after ignoring non-numeric and non-finite values."""

    if ddof not in {0, 1}:
        raise ValueError("ddof must be 0 or 1")
    numbers = [
        number
        for value in values
        if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(number := float(value))
    ]
    if not numbers:
        return None, None
    mean = fmean(numbers)
    if len(numbers) <= ddof:
        return mean, 0.0
    variance = sum((number - mean) ** 2 for number in numbers) / (len(numbers) - ddof)
    return mean, math.sqrt(variance)


__all__ = ["finite_mean_std", "pearson_correlation"]
