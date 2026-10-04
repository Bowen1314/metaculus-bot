"""Deterministic answer parsers: accept the formats the prompts ask for, refuse anything ambiguous."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from bot.parsing import (
    parse_binary_probability,
    parse_date_percentiles,
    parse_option_probabilities,
    parse_percentiles,
)


# ------------------------------------------------------------------ binary

@pytest.mark.parametrize(
    "text, expected",
    [
        ("Reasoning...\nProbability: 37%", 0.37),
        ("**Probability: 4.5%**", 0.045),
        ("Probability:** 12 %", 0.12),
        ("probability = 80%", 0.80),
        ("First guess Probability: 20%\nAfter more thought...\nProbability: 35%", 0.35),  # last one wins
        ("Probability: 0%", 0.0),
        ("Probability: 100%", 1.0),
    ],
)
def test_binary_accepts(text, expected):
    assert parse_binary_probability(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    ["", "I think it is likely.", "Probability: high", "Probability: 0.37", "Probability: 250%", "The odds are 37%"],
)
def test_binary_refuses(text):
    assert parse_binary_probability(text) is None


# ------------------------------------------------------------------ multiple choice

OPTIONS = ["Option A", "Option B", "Option C"]


def test_options_percent_format():
    text = "Rationale...\nOption A: 50%\nOption B: 30%\nOption C: 20%"
    assert parse_option_probabilities(text, OPTIONS) == pytest.approx({"Option A": 0.5, "Option B": 0.3, "Option C": 0.2})


def test_options_decimal_format_and_markdown():
    text = "- **Option A**: 0.5\n- **Option B**: 0.3\n- **Option C**: 0.2"
    assert parse_option_probabilities(text, OPTIONS) == pytest.approx({"Option A": 0.5, "Option B": 0.3, "Option C": 0.2})


def test_options_are_normalised_when_slightly_off():
    text = "Option A: 50%\nOption B: 31%\nOption C: 20%"
    result = parse_option_probabilities(text, OPTIONS)
    assert sum(result.values()) == pytest.approx(1.0)
    assert list(result) == OPTIONS


def test_options_last_block_wins():
    text = "Draft:\nOption A: 10%\nOption B: 10%\nOption C: 80%\n\nFinal:\nOption A: 60%\nOption B: 30%\nOption C: 10%"
    assert parse_option_probabilities(text, OPTIONS)["Option A"] == pytest.approx(0.6)


def test_option_names_that_contain_each_other():
    options = ["Other", "Other party", "Labour"]
    text = "Other party: 20%\nOther: 10%\nLabour: 70%"
    assert parse_option_probabilities(text, options) == pytest.approx({"Other": 0.1, "Other party": 0.2, "Labour": 0.7})


def test_option_names_with_regex_characters():
    options = ["0-5 (low)", "6+ (high)"]
    text = "0-5 (low): 35%\n6+ (high): 65%"
    assert parse_option_probabilities(text, options) == pytest.approx({"0-5 (low)": 0.35, "6+ (high)": 0.65})


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Option A: 50%\nOption B: 50%",  # one option missing
        "Option A: 80%\nOption B: 80%\nOption C: 80%",  # does not sum to ~1
        "Option A: likely\nOption B: unlikely\nOption C: no",
    ],
)
def test_options_refuses(text):
    assert parse_option_probabilities(text, OPTIONS) is None


# ------------------------------------------------------------------ numeric

TEMPLATE_ANSWER = """Some reasoning.

Percentile 10: 12
Percentile 20: 15.5
Percentile 40: 20
Percentile 60: 24
Percentile 80: 30
Percentile 90: 1,250
"""


def test_percentiles_template_format():
    assert parse_percentiles(TEMPLATE_ANSWER) == [(0.1, 12.0), (0.2, 15.5), (0.4, 20.0), (0.6, 24.0), (0.8, 30.0), (0.9, 1250.0)]


def test_percentiles_negative_currency_and_markdown():
    text = "**Percentile 10:** -30.5\n**Percentile 20:** -25\n**Percentile 40:** $-20\n**Percentile 60:** -15 (still negative)\n**Percentile 80:** -10\n**Percentile 90:** -5"
    result = parse_percentiles(text)
    assert [v for _, v in result] == [-30.5, -25.0, -20.0, -15.0, -10.0, -5.0]


def test_percentiles_last_complete_block_wins():
    text = (
        "Scenario A:\nPercentile 10: 1\nPercentile 20: 2\nPercentile 40: 3\nPercentile 60: 4\nPercentile 80: 5\nPercentile 90: 6\n"
        "Final answer:\nPercentile 10: 10\nPercentile 20: 20\nPercentile 40: 30\nPercentile 60: 40\nPercentile 80: 50\nPercentile 90: 60\n"
    )
    assert parse_percentiles(text)[0] == (0.1, 10.0)
    assert len(parse_percentiles(text)) == 6


@pytest.mark.parametrize(
    "text",
    [
        "",
        "The answer is around 40.",
        "Percentile 10: 5\nPercentile 90: 50",  # too few
        "Percentile 10: 1.2 million\nPercentile 20: 1.5 million\nPercentile 40: 2 million\nPercentile 60: 3 million",
        "Percentile 10: 12k\nPercentile 20: 15k\nPercentile 40: 20k\nPercentile 60: 30k",
        "Percentile 10: 1e6\nPercentile 20: 2e6\nPercentile 40: 3e6\nPercentile 60: 4e6",
        "Percentile 10: 1 500\nPercentile 20: 2 000\nPercentile 40: 2 500\nPercentile 60: 3 000",
        "Percentile 10: 1.234.567\nPercentile 20: 2.000.000\nPercentile 40: 3.000.000\nPercentile 60: 4.000.000",
        "Percentile 10: 2027-01-01\nPercentile 20: 2027-02-01\nPercentile 40: 2027-03-01\nPercentile 60: 2027-04-01",
        "Percentile 10: 50\nPercentile 20: 40\nPercentile 40: 30\nPercentile 60: 20",  # decreasing
    ],
)
def test_percentiles_refuses_ambiguous(text):
    assert parse_percentiles(text) is None


# ------------------------------------------------------------------ date

def test_date_percentiles():
    text = "Percentile 10: 2026-11-01\nPercentile 20: 2026-12-15\nPercentile 40: 2027-02-01\nPercentile 60: 2027-03-01T12:00:00Z\nPercentile 80: 2027-06-01\nPercentile 90: 2027-09-30"
    result = parse_date_percentiles(text)
    assert [p for p, _ in result] == [0.1, 0.2, 0.4, 0.6, 0.8, 0.9]
    assert result[0][1] == datetime(2026, 11, 1, tzinfo=timezone.utc).timestamp()
    assert result[3][1] == datetime(2027, 3, 1, 12, tzinfo=timezone.utc).timestamp()


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Percentile 10: next spring\nPercentile 20: summer",
        "Percentile 10: 2027-13-45\nPercentile 20: 2027-02-01\nPercentile 40: 2027-03-01\nPercentile 60: 2027-04-01",
        "Percentile 10: 2027-05-01\nPercentile 20: 2027-04-01\nPercentile 40: 2027-03-01\nPercentile 60: 2027-02-01",
    ],
)
def test_date_percentiles_refuses(text):
    assert parse_date_percentiles(text) is None
