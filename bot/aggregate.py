"""
Aggregation of several model runs into one forecast. Pure functions, no library imports.

- binary: median of the runs, then clipped away from 0 and 1
- multiple choice: mean per option, floored, renormalised
- numeric / discrete / date: equal-weight mixture of the runs' CDFs (bot/numeric.py)
"""

from __future__ import annotations

import math
import statistics
from typing import Mapping, Sequence


def aggregate_binary(predictions: Sequence[float], *, low: float, high: float) -> float:
    usable = [float(p) for p in predictions if p is not None and math.isfinite(p)]
    if not usable:
        raise ValueError("no binary predictions to aggregate")
    if any(p < 0 or p > 1 for p in usable):
        raise ValueError("binary predictions must be within [0, 1]")
    return clip_probability(statistics.median(usable), low=low, high=high)


def clip_probability(value: float, *, low: float, high: float) -> float:
    if not math.isfinite(value):
        raise ValueError("probability is not finite")
    return float(min(max(value, low), high))


def floor_and_normalise(probabilities: Mapping[str, float], *, floor: float) -> dict[str, float]:
    """
    Returns probabilities that sum to exactly 1 with every option >= `floor` (the floor is
    lowered automatically when there are so many options that it could not be met).
    Mass is taken proportionally from the options that sit above the floor.
    """
    names = list(probabilities)
    if len(names) < 2:
        raise ValueError("need at least two options")
    values = [float(probabilities[n]) for n in names]
    if any((not math.isfinite(v)) or v < 0 for v in values):
        raise ValueError("option probabilities must be finite and non-negative")
    total = sum(values)
    if total <= 0:
        raise ValueError("option probabilities sum to zero")
    values = [v / total for v in values]

    floor = min(floor, 0.5 / len(names))
    fixed: set[int] = set()  # options pinned at the floor
    result = list(values)
    for _ in range(len(names)):
        free = [i for i in range(len(names)) if i not in fixed]
        free_mass = 1.0 - floor * len(fixed)
        free_total = sum(values[i] for i in free)
        for i in free:
            result[i] = values[i] * free_mass / free_total
        below = [i for i in free if result[i] < floor]
        if not below:
            break
        fixed.update(below)
        for i in below:
            result[i] = floor
    # absorb floating point residue in the largest option
    residue = 1.0 - sum(result)
    result[result.index(max(result))] += residue
    return dict(zip(names, result))


def aggregate_options(
    predictions: Sequence[Mapping[str, float]], options: Sequence[str], *, floor: float
) -> dict[str, float]:
    """Mean per option across runs (every run must cover exactly `options`), then floored."""
    if not predictions:
        raise ValueError("no multiple-choice predictions to aggregate")
    expected = set(options)
    for prediction in predictions:
        if set(prediction) != expected:
            raise ValueError(f"prediction options {sorted(prediction)} do not match the question's {sorted(expected)}")
    means = {
        option: sum(float(p[option]) for p in predictions) / len(predictions) for option in options
    }
    return floor_and_normalise(means, floor=floor)


def validate_option_payload(payload: Mapping[str, float], options: Sequence[str]) -> list[str]:
    """Mirrors the server's multiple-choice check. Empty list = acceptable."""
    problems: list[str] = []
    if set(payload) != set(options):
        problems.append("options in payload do not match the question's options")
    values = [payload[o] for o in payload]
    if any((not math.isfinite(v)) or v < 0.001 or v > 0.999 for v in values):
        problems.append("every option probability must be within [0.001, 0.999]")
    if abs(sum(values) - 1.0) > 1e-6:
        problems.append(f"probabilities must sum to 1 (got {sum(values):.8f})")
    return problems


def validate_binary_payload(probability: float) -> list[str]:
    if probability is None or not math.isfinite(probability):
        return ["probability is not a finite number"]
    if probability < 0.001 or probability > 0.999:
        return ["probability must be within [0.001, 0.999]"]
    return []
