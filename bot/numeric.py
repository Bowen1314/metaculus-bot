"""
Numeric / discrete helpers: percentile clean-up, the server's CDF rules, a repair step and the
mixture aggregation. Pure functions on plain floats (numpy only), so they can be tested without
the forecasting library or the network.

`validate_cdf` mirrors Metaculus's server-side check
(questions/serializers/common.py::continuous_validation, repo commit 7ac38781).
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

DEFAULT_INBOUND_OUTCOME_COUNT = 200
# Smallest distance between two stated percentile values, as a fraction of the question's range.
MIN_VALUE_GAP_FRACTION = 1e-6


def min_step(cdf_size: int) -> float:
    return float(np.round(0.01 / (cdf_size - 1), 9))


def max_step(cdf_size: int) -> float:
    return 0.2 * DEFAULT_INBOUND_OUTCOME_COUNT / (cdf_size - 1)


def validate_cdf(
    cdf: Sequence[float], *, cdf_size: int, open_lower_bound: bool, open_upper_bound: bool
) -> list[str]:
    """Returns the list of rule violations (empty list = the server would accept it)."""
    problems: list[str] = []
    if cdf is None or len(cdf) == 0:
        return ["cdf is empty"]
    if any(v is None or not math.isfinite(v) for v in cdf):
        return ["cdf contains non-finite values"]
    values = np.round(np.asarray(cdf, dtype=float), 10)
    if len(values) != cdf_size:
        problems.append(f"cdf must have {cdf_size} values, has {len(values)}")
    if np.any(values < 0) or np.any(values > 1):
        problems.append("cdf values must be within [0, 1]")
    steps = np.round(np.diff(values), 9)
    if len(steps) and not np.all(steps >= min_step(cdf_size)):
        problems.append(f"cdf must increase by at least {min_step(cdf_size)} at every step")
    if len(steps) and not np.all(steps <= max_step(cdf_size)):
        problems.append(f"cdf must increase by no more than {max_step(cdf_size)} at every step")
    if open_lower_bound:
        if not values[0] >= 0.001:
            problems.append("cdf[0] must be at least 0.001 (open lower bound)")
    elif values[0] != 0.0:
        problems.append("cdf[0] must be 0.0 (closed lower bound)")
    if open_upper_bound:
        if not values[-1] <= 0.999:
            problems.append("cdf[-1] must be at most 0.999 (open upper bound)")
    elif values[-1] != 1.0:
        problems.append("cdf[-1] must be 1.0 (closed upper bound)")
    return problems


def repair_cdf(
    cdf: Sequence[float], *, cdf_size: int, open_lower_bound: bool, open_upper_bound: bool
) -> list[float]:
    """
    Projects an almost-valid CDF onto the server's rules: fixes the end points, enforces the
    minimum step by blending in a uniform ramp, and spreads any step above the cap over the
    others. Raises ValueError if the input cannot be interpreted as a CDF of the right length.
    """
    if cdf is None or len(cdf) != cdf_size:
        raise ValueError(f"cannot repair: expected {cdf_size} values, got {0 if cdf is None else len(cdf)}")
    values = np.asarray(cdf, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("cannot repair: non-finite values")
    values = np.maximum.accumulate(np.clip(values, 0.0, 1.0))

    low = float(np.clip(values[0], 0.001, 0.5)) if open_lower_bound else 0.0
    high = float(np.clip(values[-1], 0.5, 0.999)) if open_upper_bound else 1.0
    n_steps = cdf_size - 1
    floor = min_step(cdf_size) * 1.02  # small margin over the server's rounding
    cap = max_step(cdf_size) * 0.98

    # The in-range part must be able to hold the minimum step everywhere; if (nearly) all the
    # mass sits outside open bounds, take the little that is needed from those tails.
    required = n_steps * floor * 1.05
    if high - low < required:
        deficit = required - (high - low)
        if open_lower_bound and open_upper_bound:
            low, high = low - deficit / 2, high + deficit / 2
        elif open_lower_bound:
            low -= deficit
        elif open_upper_bound:
            high += deficit
        low = max(low, 0.001 if open_lower_bound else 0.0)
        high = min(high, 0.999 if open_upper_bound else 1.0)
        if high - low < required and open_upper_bound:
            high = min(0.999, low + required)
        if high - low < required and open_lower_bound:
            low = max(0.001, high - required)
    inbound = high - low

    steps = np.diff(values)
    total = steps.sum()
    steps = steps / total if total > 0 else np.full(n_steps, 1.0 / n_steps)
    # every step gets the floor; the remaining mass keeps the original shape
    steps = floor + steps * (inbound - n_steps * floor)

    if cap * n_steps < inbound:
        raise ValueError("cannot repair: cap too small for this cdf size")
    for _ in range(200):
        over = steps > cap
        if not over.any():
            break
        excess = float((steps[over] - cap).sum())
        steps[over] = cap
        room = np.where(~over, cap - steps, 0.0)
        if room.sum() <= 0:
            break
        steps += room * (excess / room.sum())

    out = low + np.concatenate([[0.0], np.cumsum(steps)])
    out[0] = low
    out[-1] = high
    out = np.round(out, 10)
    return [float(v) for v in out]


def ensure_valid_cdf(
    cdf: Sequence[float], *, cdf_size: int, open_lower_bound: bool, open_upper_bound: bool
) -> list[float]:
    """Returns a CDF the server will accept (repairing if needed) or raises ValueError."""
    kwargs = dict(cdf_size=cdf_size, open_lower_bound=open_lower_bound, open_upper_bound=open_upper_bound)
    rounded = [float(v) for v in np.round(np.asarray(cdf, dtype=float), 10)] if cdf is not None else None
    if rounded is not None and not validate_cdf(rounded, **kwargs):
        return rounded
    repaired = repair_cdf(cdf, **kwargs)
    problems = validate_cdf(repaired, **kwargs)
    if problems:
        raise ValueError(f"cdf invalid after repair: {problems}")
    return repaired


def mixture_cdf(cdfs: Sequence[Sequence[float]], weights: Sequence[float] | None = None) -> list[float]:
    """
    Mixture of forecasts = pointwise (weighted) mean of their CDFs; equal weights by default.
    A convex combination of CDFs that each satisfy the server rules satisfies them too (all the
    rules are convex constraints).
    """
    if not cdfs:
        raise ValueError("no cdfs to aggregate")
    lengths = {len(c) for c in cdfs}
    if len(lengths) != 1:
        raise ValueError(f"cdfs have different lengths: {sorted(lengths)}")
    if weights is None:
        return [float(v) for v in np.mean(np.asarray(cdfs, dtype=float), axis=0)]
    w = np.asarray(weights, dtype=float)
    if w.shape != (len(cdfs),) or not np.all(np.isfinite(w)) or np.any(w < 0) or w.sum() <= 0:
        raise ValueError("mixture weights must be one non-negative number per cdf, not all zero")
    w = w / w.sum()
    return [float(v) for v in w @ np.asarray(cdfs, dtype=float)]


def sanitize_percentiles(
    pairs: Sequence[tuple[float, float]],
    *,
    lower_bound: float,
    upper_bound: float,
    zero_point: float | None = None,
) -> list[tuple[float, float]]:
    """
    Cleans model-stated percentiles before handing them to the library:
    sorts by percentile, drops duplicates and anything outside (0, 1), makes values strictly
    increasing (ties are separated by a millionth of the range), and pulls absurd values back to within 1.5 ranges of the bounds (the library
    rejects anything further than 2 ranges away, and anything below a log scale's zero point).
    If the whole distribution lies beyond one bound, its nearest percentile is moved to just
    outside that bound so the library accepts it; the submitted CDF still puts almost all mass
    out of range on that side. Raises ValueError if fewer than two percentiles remain.
    """
    span = upper_bound - lower_bound
    if not (span > 0):
        raise ValueError("upper_bound must be greater than lower_bound")
    cleaned: dict[float, float] = {}
    for percentile, value in pairs:
        if percentile is None or value is None:
            continue
        if not (math.isfinite(percentile) and math.isfinite(value)):
            continue
        if percentile > 1.0:  # given as 10, 20, ... instead of 0.1, 0.2, ...
            percentile = percentile / 100.0
        if not (0.0 < percentile < 1.0):
            continue
        cleaned[round(percentile, 6)] = float(value)
    ordered = sorted(cleaned.items())
    if len(ordered) < 2:
        raise ValueError("need at least two usable percentiles")
    low_limit, high_limit = lower_bound - 1.5 * span, upper_bound + 1.5 * span
    if zero_point is not None and zero_point < lower_bound:
        # log-scaled question: nothing at or below the zero point can be represented
        low_limit = max(low_limit, zero_point + 0.01 * (lower_bound - zero_point))
    percentiles = [p for p, _ in ordered]
    values = [min(max(v, low_limit), high_limit) for _, v in ordered]

    near_low, near_high = lower_bound - 0.2 * span, upper_bound + 0.2 * span
    if zero_point is not None and zero_point < lower_bound:
        near_low = max(near_low, low_limit)
    if all(v > near_high for v in values):
        values[values.index(min(values))] = near_high
    elif all(v < near_low for v in values):
        values[values.index(max(values))] = near_low

    # Strictly increasing, with a visible gap. Ties (or near-ties) must not reach the library: it
    # nudges tied values by 1e-10 and then extrapolates across that sliver, which yields a
    # non-monotone CDF and, at an open bound, an endless loop in its standardisation step.
    gap = span * MIN_VALUE_GAP_FRACTION
    result: list[tuple[float, float]] = []
    previous = -math.inf
    for percentile, value in zip(percentiles, values):
        value = max(value, previous + gap) if result else value
        result.append((percentile, value))
        previous = value
    return result


def percentile_values_from_cdf(
    cdf: Sequence[float], x_axis: Sequence[float], targets: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)
) -> list[tuple[float, float]]:
    """Reads off the x value at each target cumulative probability (for logs and comments)."""
    return [(t, float(np.interp(t, cdf, x_axis))) for t in targets]
