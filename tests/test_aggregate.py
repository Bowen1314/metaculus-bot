"""Aggregation of runs, and the payload checks that mirror the server."""

from __future__ import annotations

import pytest

from bot.aggregate import (
    aggregate_binary,
    aggregate_options,
    floor_and_normalise,
    validate_binary_payload,
    validate_option_payload,
)


def test_binary_is_median_and_ignores_one_outlier():
    assert aggregate_binary([0.30, 0.35, 0.99, 0.32, 0.28], low=0.03, high=0.97) == pytest.approx(0.32)


def test_binary_even_count_uses_midpoint():
    assert aggregate_binary([0.2, 0.4], low=0.03, high=0.97) == pytest.approx(0.3)


@pytest.mark.parametrize("runs, expected", [([0.001, 0.002, 0.0], 0.03), ([0.999, 1.0, 0.99], 0.97)])
def test_binary_is_clipped_away_from_certainty(runs, expected):
    assert aggregate_binary(runs, low=0.03, high=0.97) == pytest.approx(expected)


@pytest.mark.parametrize("runs", [[], [float("nan")], [1.2], [-0.1]])
def test_binary_rejects_bad_input(runs):
    with pytest.raises(ValueError):
        aggregate_binary(runs, low=0.03, high=0.97)


def test_options_mean_then_floor():
    options = ["A", "B", "C"]
    runs = [{"A": 0.9, "B": 0.1, "C": 0.0}, {"A": 0.7, "B": 0.3, "C": 0.0}]
    result = aggregate_options(runs, options, floor=0.02)
    assert list(result) == options
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-12)
    assert result["C"] == pytest.approx(0.02)
    assert result["A"] / result["B"] == pytest.approx(0.8 / 0.2)  # the rest keeps its proportions
    assert not validate_option_payload(result, options)


def test_floor_never_breaks_the_sum_with_many_options():
    options = [f"o{i}" for i in range(40)]
    values = {o: 0.0 for o in options}
    values["o0"] = 1.0
    result = floor_and_normalise(values, floor=0.02)
    assert sum(result.values()) == pytest.approx(1.0, abs=1e-12)
    assert min(result.values()) > 0.001
    assert not validate_option_payload(result, options)


def test_floor_leaves_values_above_the_floor_alone():
    values = {"A": 0.5, "B": 0.3, "C": 0.2}
    assert floor_and_normalise(values, floor=0.02) == pytest.approx(values)


def test_options_must_match_the_question():
    with pytest.raises(ValueError):
        aggregate_options([{"A": 0.5, "B": 0.5}], ["A", "B", "C"], floor=0.02)
    with pytest.raises(ValueError):
        aggregate_options([], ["A", "B"], floor=0.02)
    with pytest.raises(ValueError):
        floor_and_normalise({"A": 0.0, "B": 0.0}, floor=0.02)


def test_payload_validators_mirror_server_limits():
    assert validate_binary_payload(0.5) == []
    assert validate_binary_payload(0.0005)
    assert validate_binary_payload(0.9995)
    assert validate_binary_payload(float("nan"))
    assert validate_option_payload({"A": 0.5, "B": 0.5}, ["A", "B"]) == []
    assert validate_option_payload({"A": 0.9995, "B": 0.0005}, ["A", "B"])
    assert validate_option_payload({"A": 0.6, "B": 0.5}, ["A", "B"])
    assert validate_option_payload({"A": 1.0}, ["A", "B"])


# ------------------------------------------------------------------ extremizing (2026-10-18)

import math  # noqa: E402

from bot.aggregate import (  # noqa: E402
    aggregate_binary_by_model,
    aggregate_options_by_model,
    extremize,
    group_by_model,
    model_weights,
)


def _expected_extremized(p, k):
    odds = (p / (1 - p)) ** k
    return odds / (1 + odds)


@pytest.mark.parametrize("p", [0.01, 0.1, 0.3, 0.45, 0.5, 0.62, 0.9, 0.99])
def test_extremize_is_logit_times_factor(p):
    got = extremize(p, 1.2)
    assert got == pytest.approx(_expected_extremized(p, 1.2), abs=1e-12)
    assert math.log(got / (1 - got)) == pytest.approx(1.2 * math.log(p / (1 - p)), abs=1e-9)


def test_extremize_known_values():
    assert extremize(0.5, 1.2) == 0.5
    assert extremize(0.3, 1.2) == pytest.approx(0.2657, abs=1e-4)
    assert extremize(0.7, 1.2) == pytest.approx(0.7343, abs=1e-4)
    assert extremize(0.8, 1.2) == pytest.approx(0.8407, abs=1e-4)


def test_extremize_is_symmetric_monotone_and_moves_away_from_half():
    grid = [i / 100 for i in range(1, 100)]
    pushed = [extremize(p, 1.2) for p in grid]
    assert all(a < b for a, b in zip(pushed, pushed[1:]))
    for p, q in zip(grid, pushed):
        assert extremize(1 - p, 1.2) == pytest.approx(1 - q, abs=1e-12)
        assert abs(q - 0.5) >= abs(p - 0.5) - 1e-15


def test_extremize_factor_one_is_the_identity_and_bad_factors_are_refused():
    for p in (0.0, 0.03, 0.4, 0.97, 1.0):
        assert extremize(p, 1.0) == p
    assert extremize(0.0, 1.2) == 0.0 and extremize(1.0, 1.2) == 1.0
    for factor in (0, -1, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            extremize(0.4, factor)
    with pytest.raises(ValueError):
        extremize(1.2, 1.2)


@pytest.mark.parametrize("runs, expected", [([0.95, 0.96], 0.97), ([0.04, 0.05], 0.03)])
def test_clip_comes_after_extremizing(runs, expected):
    result = aggregate_binary_by_model(runs, ["a", "b"], low=0.03, high=0.97, extremize_factor=1.2)
    assert abs(result.extremized - 0.5) > abs(expected - 0.5)  # extremizing alone would cross the clip
    assert result.final == pytest.approx(expected)


def test_extremized_value_inside_the_clip_is_kept():
    result = aggregate_binary_by_model([0.85, 0.95], ["a", "b"], low=0.03, high=0.97, extremize_factor=1.2)
    assert result.mean == pytest.approx(0.90)
    assert result.final == result.extremized == pytest.approx(_expected_extremized(0.90, 1.2))
    assert 0.90 < result.final < 0.97


def test_per_run_binary_is_extremized_after_the_median():
    assert aggregate_binary([0.2, 0.3, 0.9], low=0.03, high=0.97, extremize_factor=1.2) == pytest.approx(
        _expected_extremized(0.3, 1.2)
    )


# ------------------------------------------------------------------ one vote per model (2026-10-18)

def test_runs_of_one_model_are_combined_before_the_models():
    # two GPT runs, one Claude, one Gemini: per run the mean would be 0.45, the median 0.45
    result = aggregate_binary_by_model(
        [0.60, 0.60, 0.20, 0.40], ["gpt", "gpt", "claude", "gemini"],
        low=0.03, high=0.97, extremize_factor=1.0,
    )
    assert result.per_model == {"gpt": [0.60, 0.60], "claude": [0.20], "gemini": [0.40]}
    assert result.model_means == pytest.approx({"gpt": 0.60, "claude": 0.20, "gemini": 0.40})
    assert result.mean == pytest.approx(0.40) and result.final == pytest.approx(0.40)


def test_binary_votes_are_averaged_not_medianed():
    result = aggregate_binary_by_model([0.20, 0.24, 0.99], ["gpt", "claude", "gemini"], low=0.03, high=0.97)
    assert result.mean == pytest.approx((0.20 + 0.24 + 0.99) / 3)


def test_binary_breakdown_text_shows_votes_mean_and_extremized_value():
    result = aggregate_binary_by_model(
        [0.30, 0.34, 0.10], ["openai/gpt", "openai/gpt", "anthropic/claude"], low=0.03, high=0.97, extremize_factor=1.2
    )
    text = result.describe()
    assert "per-model votes [openai/gpt=0.320 (mean of 2 runs: 0.300, 0.340), anthropic/claude=0.100]" in text
    assert "mean=0.2100" in text
    assert f"extremized(logit x 1.2)={_expected_extremized(0.21, 1.2):.4f}" in text
    assert text.endswith(f"final={result.final:.4f}")


def test_binary_by_model_rejects_bad_input():
    with pytest.raises(ValueError):
        aggregate_binary_by_model([], [], low=0.03, high=0.97)
    with pytest.raises(ValueError):
        aggregate_binary_by_model([0.2, 0.3], ["a"], low=0.03, high=0.97)
    with pytest.raises(ValueError):
        aggregate_binary_by_model([0.2, float("nan")], ["a", "b"], low=0.03, high=0.97)


def test_group_by_model_keeps_first_seen_order():
    assert group_by_model([1, 2, 3, 4], ["b", "a", "b", "c"]) == {"b": [1, 3], "a": [2], "c": [4]}
    assert list(group_by_model([1, 2, 3, 4], ["b", "a", "b", "c"])) == ["b", "a", "c"]


def test_model_weights_give_every_model_the_same_total():
    weights = model_weights(["gpt", "opus", "gemini", "gpt", "opus"])
    assert weights == pytest.approx([1 / 6, 1 / 6, 1 / 3, 1 / 6, 1 / 6])
    assert sum(weights) == pytest.approx(1.0)
    assert model_weights(["a", "b", "c"]) == pytest.approx([1 / 3] * 3)
    with pytest.raises(ValueError):
        model_weights([])


def test_options_by_model_average_within_then_across_models():
    options = ["A", "B"]
    runs = [{"A": 0.8, "B": 0.2}, {"A": 0.2, "B": 0.8}, {"A": 0.6, "B": 0.4}]
    result = aggregate_options_by_model(runs, ["gpt", "claude", "gpt"], options, floor=0.02)
    assert result == pytest.approx({"A": (0.7 + 0.2) / 2, "B": (0.3 + 0.8) / 2})
    assert aggregate_options(runs, options, floor=0.02) == pytest.approx({"A": 1.6 / 3, "B": 1.4 / 3})  # per run
    assert not validate_option_payload(result, options)
    with pytest.raises(ValueError):
        aggregate_options_by_model([{"A": 1.0}], ["gpt"], options, floor=0.02)
