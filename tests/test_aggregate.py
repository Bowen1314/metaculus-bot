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
