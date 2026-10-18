"""
Aggregation of several model runs into one forecast. Pure functions, no library imports.

Two methods ([ensemble] aggregate in config.toml):
- "per_run" (the original): binary = median of the runs; multiple choice = mean per option;
  numeric / discrete / date = equal-weight mixture of the runs' CDFs (bot/numeric.py).
- "per_model_mean": one vote per model. The runs of one model are combined first (mean, or
  equal-weight CDF mixture), then the models are combined with equal weight: binary = mean of the
  model means, multiple choice = mean of the model means per option, numeric = mixture in which
  every model has the same total weight.
Binary forecasts are then extremized (logit x [calibration] binary_extremize) and clipped away from
0 and 1; multiple choice is floored and renormalised.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Hashable, Mapping, Sequence, TypeVar

T = TypeVar("T")

AGGREGATE_METHODS = ("per_run", "per_model_mean")


def aggregate_binary(
    predictions: Sequence[float], *, low: float, high: float, extremize_factor: float = 1.0
) -> float:
    """Median of the runs ("per_run"), extremized, clipped."""
    usable = _check_binary(predictions)
    return clip_probability(extremize(statistics.median(usable), extremize_factor), low=low, high=high)


def _check_binary(predictions: Sequence[float]) -> list[float]:
    usable = [float(p) for p in predictions if p is not None and math.isfinite(p)]
    if not usable:
        raise ValueError("no binary predictions to aggregate")
    if any(p < 0 or p > 1 for p in usable):
        raise ValueError("binary predictions must be within [0, 1]")
    return usable


def extremize(probability: float, factor: float) -> float:
    """Pushes a probability away from 0.5 in log-odds: sigmoid(factor * logit(p)).
    factor 1 returns p unchanged; 0 and 1 stay where they are."""
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError("extremize factor must be a positive number")
    p = float(probability)
    if not (0.0 <= p <= 1.0):
        raise ValueError("probability must be within [0, 1]")
    if factor == 1.0 or p in (0.0, 1.0):
        return p
    z = factor * math.log(p / (1.0 - p))
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def group_by_model(predictions: Sequence[T], models: Sequence[Hashable]) -> dict[Hashable, list[T]]:
    """{model: [its runs]} in the order the models first appear."""
    if len(predictions) != len(models):
        raise ValueError(f"{len(predictions)} predictions but {len(models)} model labels")
    groups: dict[Hashable, list[T]] = {}
    for prediction, model in zip(predictions, models):
        groups.setdefault(model, []).append(prediction)
    return groups


def model_weights(models: Sequence[Hashable]) -> list[float]:
    """Weight of each run so that every model gets the same total weight (sums to 1)."""
    if not models:
        raise ValueError("no runs")
    counts: dict[Hashable, int] = {}
    for model in models:
        counts[model] = counts.get(model, 0) + 1
    return [1.0 / (len(counts) * counts[model]) for model in models]


@dataclass(frozen=True)
class BinaryAggregate:
    per_model: dict[str, list[float]]   # model -> its runs
    model_means: dict[str, float]       # model -> mean of its runs (its vote)
    mean: float                         # mean of the votes
    extremized: float                   # after logit x factor
    final: float                        # after the clip
    factor: float

    def describe(self) -> str:
        votes = ", ".join(
            f"{model}={self.model_means[model]:.3f}"
            + (f" (mean of {len(runs)} runs: {', '.join(f'{r:.3f}' for r in runs)})" if len(runs) > 1 else "")
            for model, runs in self.per_model.items()
        )
        return (
            f"per-model votes [{votes}]; mean={self.mean:.4f}; "
            f"extremized(logit x {self.factor:g})={self.extremized:.4f}; final={self.final:.4f}"
        )


def aggregate_binary_by_model(
    predictions: Sequence[float], models: Sequence[str], *, low: float, high: float, extremize_factor: float = 1.0
) -> BinaryAggregate:
    """"per_model_mean": mean within each model, mean across models, extremize, clip."""
    usable = _check_binary(predictions)
    if len(usable) != len(predictions):
        raise ValueError("binary predictions must be finite numbers")
    groups = group_by_model(usable, models)
    means = {model: sum(runs) / len(runs) for model, runs in groups.items()}
    mean = sum(means.values()) / len(means)
    pushed = extremize(mean, extremize_factor)
    return BinaryAggregate(
        per_model={str(m): list(r) for m, r in groups.items()},
        model_means={str(m): v for m, v in means.items()},
        mean=mean,
        extremized=pushed,
        final=clip_probability(pushed, low=low, high=high),
        factor=extremize_factor,
    )


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


def aggregate_options_by_model(
    predictions: Sequence[Mapping[str, float]], models: Sequence[str], options: Sequence[str], *, floor: float
) -> dict[str, float]:
    """"per_model_mean": mean per option within each model, then across models, then floored."""
    if not predictions:
        raise ValueError("no multiple-choice predictions to aggregate")
    groups = group_by_model(list(predictions), models)
    expected = set(options)
    votes = []
    for runs in groups.values():
        for prediction in runs:
            if set(prediction) != expected:
                raise ValueError(
                    f"prediction options {sorted(prediction)} do not match the question's {sorted(expected)}"
                )
        votes.append({o: sum(float(p[o]) for p in runs) / len(runs) for o in options})
    return aggregate_options(votes, options, floor=floor)


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
