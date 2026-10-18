"""CDF construction: what we would send must satisfy the server's rules for every question shape."""

from __future__ import annotations

import numpy as np
import pytest
from forecasting_tools import NumericDistribution
from forecasting_tools.data_models.numeric_report import Percentile
from forecasting_tools.data_models.questions import DateQuestion

from bot import numeric
from bot.forecaster import cdf_points
from tests.conftest import log_scaled_post, parse_posts

FRACTIONS = (0.1, 0.2, 0.4, 0.6, 0.8, 0.9)


def rules(question) -> dict:
    return dict(
        cdf_size=question.cdf_size,
        open_lower_bound=question.open_lower_bound,
        open_upper_bound=question.open_upper_bound,
    )


def bounds(question) -> tuple[float, float]:
    if isinstance(question, DateQuestion):
        return question.lower_bound.timestamp(), question.upper_bound.timestamp()
    return float(question.lower_bound), float(question.upper_bound)


def cdf_for(question, positions) -> list[float]:
    """Library CDF for percentiles placed at the given positions (0 = lower bound, 1 = upper)."""
    low, high = bounds(question)
    pairs = [(f, low + p * (high - low)) for f, p in zip(FRACTIONS, positions)]
    clean = numeric.sanitize_percentiles(pairs, lower_bound=low, upper_bound=high, zero_point=question.zero_point)
    distribution = NumericDistribution.from_question([Percentile(percentile=p, value=v) for p, v in clean], question)
    return [point.percentile for point in cdf_points(distribution)]  # time-limited


SHAPES = {
    "central": (0.30, 0.38, 0.47, 0.53, 0.62, 0.70),
    "very narrow": (0.500, 0.501, 0.502, 0.503, 0.504, 0.505),
    "wide": (0.02, 0.10, 0.35, 0.65, 0.90, 0.98),
    "hugging the lower bound": (0.0, 0.0, 0.01, 0.02, 0.05, 0.10),
    "hugging the upper bound": (0.90, 0.95, 0.98, 0.99, 1.0, 1.0),
    "partly below the range": (-0.5, -0.2, 0.1, 0.3, 0.5, 0.7),
    "partly above the range": (0.4, 0.6, 0.9, 1.2, 1.6, 2.0),
    "entirely above the range": (3.0, 4.0, 5.0, 6.0, 7.0, 8.0),
    "entirely below the range": (-8.0, -7.0, -6.0, -5.0, -4.0, -3.0),
    "absurdly far out": (-1e9, 0.2, 0.4, 0.6, 0.8, 1e12),
}


@pytest.mark.parametrize("shape", list(SHAPES))
def test_every_numeric_shape_gives_a_server_valid_cdf(open_questions, posts, shape):
    targets = [q for q in open_questions if hasattr(q, "cdf_size")] + parse_posts([log_scaled_post(posts)])
    assert {q.cdf_size for q in targets} == {201, 72, 17}  # continuous, date, both discrete sizes, log scale
    for question in targets:
        cdf = cdf_for(question, SHAPES[shape])
        assert numeric.validate_cdf(cdf, **rules(question)) == [], (question.question_type, question.cdf_size, shape)
        assert len(cdf) == question.cdf_size


def test_validate_cdf_mirrors_each_server_rule():
    good = list(np.linspace(0.0, 1.0, 201))
    closed = dict(cdf_size=201, open_lower_bound=False, open_upper_bound=False)
    opened = dict(cdf_size=201, open_lower_bound=True, open_upper_bound=True)
    assert numeric.validate_cdf(good, **closed) == []
    assert numeric.validate_cdf(list(np.linspace(0.001, 0.999, 201)), **opened) == []

    assert any("201 values" in p for p in numeric.validate_cdf(good[:-1], **closed))
    assert any("closed lower" in p for p in numeric.validate_cdf([0.01] + good[1:], **closed))
    assert any("closed upper" in p for p in numeric.validate_cdf(good[:-1] + [0.99], **closed))
    assert any("at least 0.001" in p for p in numeric.validate_cdf(good, **opened))
    assert any("at most 0.999" in p for p in numeric.validate_cdf(list(np.linspace(0.001, 1.0, 201)), **opened))

    flat = list(good)
    flat[100] = flat[99]  # a zero step
    assert any("at least" in p for p in numeric.validate_cdf(flat, **closed))

    spike = [0.0] + list(np.linspace(0.3, 1.0, 200))  # first step 0.3 > cap 0.2
    assert any("no more than" in p for p in numeric.validate_cdf(spike, **closed))

    assert numeric.validate_cdf([], **closed)
    assert numeric.validate_cdf([float("nan")] * 201, **closed)
    assert any("within [0, 1]" in p for p in numeric.validate_cdf([-0.1] + good[1:], **closed))


def test_step_limits_scale_with_discrete_size():
    assert numeric.min_step(201) == pytest.approx(0.00005)
    assert numeric.max_step(201) == pytest.approx(0.2)
    assert numeric.min_step(17) == pytest.approx(0.000625)
    assert numeric.max_step(17) == pytest.approx(2.5)  # no effective cap for small discrete questions


@pytest.mark.parametrize("open_lower, open_upper", [(False, False), (True, True), (True, False), (False, True)])
@pytest.mark.parametrize("size", [201, 72, 17])
def test_repair_makes_broken_cdfs_valid(size, open_lower, open_upper):
    kwargs = dict(cdf_size=size, open_lower_bound=open_lower, open_upper_bound=open_upper)
    broken = {
        "flat": [0.5] * size,
        "step function": [0.0] * (size // 2) + [1.0] * (size - size // 2),
        "not monotone": list(np.clip(np.linspace(0, 1, size) + np.sin(np.arange(size)) * 0.05, 0, 1)),
        "wrong end points": list(np.linspace(0.2, 0.8, size)),
    }
    for name, cdf in broken.items():
        repaired = numeric.ensure_valid_cdf(cdf, **kwargs)
        assert numeric.validate_cdf(repaired, **kwargs) == [], name


def test_repair_leaves_a_valid_cdf_untouched_and_rejects_nonsense():
    kwargs = dict(cdf_size=201, open_lower_bound=False, open_upper_bound=False)
    good = [round(float(v), 10) for v in np.linspace(0.0, 1.0, 201)]
    assert numeric.ensure_valid_cdf(good, **kwargs) == good
    with pytest.raises(ValueError):
        numeric.ensure_valid_cdf(good[:50], **kwargs)
    with pytest.raises(ValueError):
        numeric.ensure_valid_cdf([float("nan")] * 201, **kwargs)


def test_mixture_of_valid_cdfs_is_valid_and_keeps_both_modes(by_type):
    question = by_type["numeric"]
    low_run = cdf_for(question, (0.10, 0.12, 0.15, 0.17, 0.20, 0.22))
    high_run = cdf_for(question, (0.78, 0.80, 0.83, 0.85, 0.88, 0.90))
    mixture = numeric.mixture_cdf([low_run, high_run])
    assert numeric.validate_cdf(mixture, **rules(question)) == []
    assert mixture == pytest.approx(list((np.array(low_run) + np.array(high_run)) / 2))
    mass = np.diff(mixture)
    middle = mass[90:110].sum()
    assert mass[20:45].sum() > 5 * middle and mass[155:180].sum() > 5 * middle  # two humps, not one in the middle


def test_mixture_rejects_mismatched_lengths():
    with pytest.raises(ValueError):
        numeric.mixture_cdf([[0.0, 1.0], [0.0, 0.5, 1.0]])
    with pytest.raises(ValueError):
        numeric.mixture_cdf([])


def test_sanitize_percentiles():
    clean = numeric.sanitize_percentiles(
        [(90, 80.0), (10, 20.0), (0.5, 50.0), (0.5, 55.0), (0.0, 1.0), (1.0, 99.0), (0.2, float("nan")), (0.6, 40.0)],
        lower_bound=0.0,
        upper_bound=100.0,
    )
    assert [p for p, _ in clean] == [0.1, 0.5, 0.6, 0.9]  # sorted, 10/90 rescaled, 0 and 1 and NaN dropped
    # later duplicate wins; a tie is separated by a millionth of the range
    assert [v for _, v in clean] == pytest.approx([20.0, 55.0, 55.0001, 80.0])
    assert all(b > a for a, b in zip([v for _, v in clean], [v for _, v in clean][1:]))

    far = numeric.sanitize_percentiles([(0.1, -1e9), (0.9, 1e9)], lower_bound=0.0, upper_bound=100.0)
    assert far == [(0.1, -150.0), (0.9, 250.0)]

    above = numeric.sanitize_percentiles([(0.1, 500.0), (0.5, 600.0), (0.9, 700.0)], lower_bound=0.0, upper_bound=100.0)
    assert above[0][1] == pytest.approx(120.0) and above[-1][1] == pytest.approx(250.0, abs=0.001)
    below = numeric.sanitize_percentiles([(0.1, -700.0), (0.5, -600.0), (0.9, -500.0)], lower_bound=0.0, upper_bound=100.0)
    assert below[-1][1] == pytest.approx(-20.0, abs=0.001) and below[0][1] < below[1][1] < below[2][1]

    ties = numeric.sanitize_percentiles([(p, 0.0) for p in (0.1, 0.2, 0.4, 0.6, 0.8, 0.9)], lower_bound=0.0, upper_bound=100.0)
    gaps = [b - a for a, b in zip([v for _, v in ties], [v for _, v in ties][1:])]
    assert all(gap >= 100.0 * numeric.MIN_VALUE_GAP_FRACTION * 0.999 for gap in gaps)

    log_scale = numeric.sanitize_percentiles([(0.1, -5.0), (0.9, 500.0)], lower_bound=1.0, upper_bound=1000.0, zero_point=0.0)
    assert log_scale[0][1] > 0.0

    with pytest.raises(ValueError):
        numeric.sanitize_percentiles([(0.5, 1.0)], lower_bound=0.0, upper_bound=100.0)
    with pytest.raises(ValueError):
        numeric.sanitize_percentiles([(0.1, 1.0), (0.9, 2.0)], lower_bound=5.0, upper_bound=5.0)


def test_weighted_mixture_gives_each_model_its_share(by_type):
    question = by_type["numeric"]
    low_run = cdf_for(question, (0.10, 0.12, 0.15, 0.17, 0.20, 0.22))
    mid_run = cdf_for(question, (0.40, 0.45, 0.48, 0.52, 0.55, 0.60))
    high_run = cdf_for(question, (0.78, 0.80, 0.83, 0.85, 0.88, 0.90))
    # two runs of model "gpt" (low, low), one of "claude" (mid), one of "gemini" (high)
    from bot.aggregate import model_weights

    weights = model_weights(["gpt", "gpt", "claude", "gemini"])
    mixture = numeric.mixture_cdf([low_run, low_run, mid_run, high_run], weights)
    expected = (np.array(low_run) + np.array(mid_run) + np.array(high_run)) / 3
    assert mixture == pytest.approx(list(expected))
    assert numeric.validate_cdf(mixture, **rules(question)) == []
    # equal weights reproduce the plain mixture; weights are normalised
    plain = numeric.mixture_cdf([low_run, high_run])
    assert numeric.mixture_cdf([low_run, high_run], [2.0, 2.0]) == pytest.approx(plain)


@pytest.mark.parametrize("weights", [[1.0], [1.0, -0.5], [0.0, 0.0], [1.0, float("nan")]])
def test_weighted_mixture_rejects_bad_weights(weights):
    with pytest.raises(ValueError):
        numeric.mixture_cdf([[0.0, 0.5, 1.0], [0.0, 0.4, 1.0]], weights)
