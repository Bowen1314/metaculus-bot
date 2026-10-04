"""
Deterministic parsers for the answer formats the prompts ask for.

They return None when the text is not unambiguous; the caller then falls back to the library's
LLM-based `structure_output`. Pure functions, no network, no library imports.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, timezone

_NUMBER = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|[-+]?\.\d+"

_PROBABILITY = re.compile(
    r"probability\s*\**\s*[:=]\s*\**\s*(?P<num>\d{1,3}(?:\.\d+)?)\s*%",
    re.IGNORECASE,
)

_PERCENTILE_LINE = re.compile(
    r"percentile\s*\**\s*(?P<pct>\d{1,2}(?:\.\d+)?)\s*(?:th|st|nd|rd)?\s*\**\s*[:=]\s*\**\s*"
    r"(?P<prefix>[$€£]?)\s*(?P<num>" + _NUMBER + r")(?P<rest>[^\n]*)",
    re.IGNORECASE,
)

# A magnitude word right after the number means the value is not in plain units.
_SCALE_WORD = re.compile(
    r"^\s*(?:k|m|mm|b|bn|t|tn|thousand|million|billion|trillion|hundred|e[-+]?\d+|x\s*10)\b",
    re.IGNORECASE,
)
_DATE_LIKE = re.compile(r"^\s*-\d{1,2}-\d{1,2}")  # "2027" followed by "-03-01"
_MORE_DIGITS = re.compile(r"^(?:\s+\d|[.,]\d)")  # "1 500" or "1.234.567": grouping we cannot trust


_DATE_PERCENTILE_LINE = re.compile(
    r"percentile\s*\**\s*(?P<pct>\d{1,2}(?:\.\d+)?)\s*(?:th|st|nd|rd)?\s*\**\s*[:=]\s*\**\s*"
    r"(?P<date>\d{4}-\d{2}-\d{2})(?:[T ](?P<time>\d{2}:\d{2}(?::\d{2})?)\s*(?:Z|UTC)?)?",
    re.IGNORECASE,
)


def _to_float(text: str) -> float:
    return float(text.replace(",", ""))


def _last_complete_block(entries: list[tuple[float, float]]) -> list[tuple[float, float]] | None:
    """Within one stated distribution the percentiles ascend; a percentile that does not exceed
    the previous one starts a new block (a draft, or another scenario). Returns the last block
    with at least four entries."""
    blocks: list[list[tuple[float, float]]] = []
    previous = None
    for pct, value in entries:
        if previous is None or pct <= previous:
            blocks.append([])
        blocks[-1].append((pct, value))
        previous = pct
    complete = [block for block in blocks if len(block) >= 4]
    return complete[-1] if complete else None


def parse_binary_probability(text: str) -> float | None:
    """Last 'Probability: ZZ%' in the text, as a decimal in [0, 1]."""
    if not text:
        return None
    matches = list(_PROBABILITY.finditer(text))
    if not matches:
        return None
    value = float(matches[-1].group("num"))
    if not (0.0 <= value <= 100.0) or math.isnan(value):
        return None
    return value / 100.0


def parse_option_probabilities(text: str, options: list[str]) -> dict[str, float] | None:
    """
    Reads the final 'Option: probability' block. Every option must be found; values may be
    percentages or decimals. Returns probabilities normalised to sum to 1, in `options` order.
    """
    if not text or not options:
        return None
    found: dict[str, tuple[float, bool]] = {}
    lines = text.splitlines()
    # Longest names first so "Other" cannot steal a line that belongs to "Other party".
    for option in sorted(options, key=len, reverse=True):
        pattern = re.compile(
            r"^[\s>*\-•\d.)]*(?:option[\s_]*[A-Za-z0-9]*\s*[:.)\-]\s*)?[\"'`*_]*"
            + re.escape(option)
            + r"[\"'`*_]*\s*[:=\-–—]+\s*\**\s*(?P<num>\d{1,3}(?:\.\d+)?|\.\d+)\s*(?P<pct>%?)",
            re.IGNORECASE,
        )
        for line in reversed(lines):
            match = pattern.match(line)
            if match:
                found[option] = (float(match.group("num")), bool(match.group("pct")))
                break
    if len(found) != len(options):
        return None
    values = [found[o][0] for o in options]
    any_percent_sign = any(found[o][1] for o in options)
    total = sum(values)
    if any_percent_sign or total > 1.5:
        values = [v / 100.0 for v in values]
        total = sum(values)
    if not (0.9 <= total <= 1.1) or any(v < 0 or v > 1 for v in values):
        return None
    return {o: v / total for o, v in zip(options, values)}


def parse_percentiles(text: str) -> list[tuple[float, float]] | None:
    """
    Reads 'Percentile N: value' lines and returns [(fraction, value), ...] in ascending order.
    If the text holds several distributions (drafts, scenarios) the last complete one is used
    (the final answer comes last). Returns None when no block has at least four percentiles,
    when any value carries a magnitude word (million, k, ...), looks like a date or has
    ambiguous digit grouping, or when values decrease as percentiles increase.
    """
    if not text:
        return None
    entries: list[tuple[float, float]] = []
    for match in _PERCENTILE_LINE.finditer(text):
        rest = match.group("rest") or ""
        if _SCALE_WORD.match(rest) or _DATE_LIKE.match(rest) or _MORE_DIGITS.match(rest):
            return None
        pct = float(match.group("pct"))
        if not (0.0 < pct < 100.0):
            continue
        try:
            value = _to_float(match.group("num"))
        except ValueError:
            return None
        entries.append((pct, value))
    block = _last_complete_block(entries)
    if block is None:
        return None
    pairs = [(pct / 100.0, value) for pct, value in block]
    values = [value for _, value in pairs]
    if any(b < a for a, b in zip(values, values[1:])):
        return None
    if any(math.isnan(v) or math.isinf(v) for v in values):
        return None
    return pairs


def parse_date_percentiles(text: str) -> list[tuple[float, float]] | None:
    """
    Reads 'Percentile N: YYYY-MM-DD' lines (optionally with 'THH:MM:SSZ') and returns
    [(fraction, unix timestamp in UTC), ...]. Same block and ordering rules as parse_percentiles.
    """
    if not text:
        return None
    entries: list[tuple[float, float]] = []
    for match in _DATE_PERCENTILE_LINE.finditer(text):
        pct = float(match.group("pct"))
        if not (0.0 < pct < 100.0):
            continue
        clock = match.group("time") or "00:00:00"
        if len(clock) == 5:
            clock += ":00"
        try:
            moment = datetime.strptime(f"{match.group('date')} {clock}", "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return None
        entries.append((pct, moment.replace(tzinfo=timezone.utc).timestamp()))
    block = _last_complete_block(entries)
    if block is None:
        return None
    values = [value for _, value in block]
    if any(b < a for a, b in zip(values, values[1:])):
        return None
    return [(pct / 100.0, value) for pct, value in block]
