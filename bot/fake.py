"""
Offline stand-in for the LLMs (`--fake-llm`). It answers each prompt in the format the prompt
asks for, with arbitrary numbers derived from a hash, so the whole pipeline (fetch, prompts,
parsing, aggregation, payload validation, printing) can be exercised without a provider key and
without spending anything. Its output is NOT a forecast and can never be submitted (the runner
refuses to go live with a fake factory).
"""

from __future__ import annotations

import ast
import hashlib
import math
import re
from datetime import datetime, timedelta, timezone

MARK = "FAKE ANSWER (offline test mode, not a forecast)"

_LOWER = re.compile(r"not (?:be )?lower than\s+(\S+)")
_UPPER = re.compile(r"not (?:be )?higher than\s+(\S+)")
_OPTIONS = re.compile(r"The options are:\s*(\[.*?\])\s*$", re.MULTILINE)
_Z = (-1.0, -0.66, -0.2, 0.2, 0.66, 1.0)
_PERCENTILES = (10, 20, 40, 60, 80, 90)


def _unit(seed: str, salt: str) -> float:
    digest = hashlib.sha256(f"{salt}|{seed}".encode()).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _plain(value: float, span: float) -> str:
    decimals = 0 if span <= 0 else min(10, max(0, 4 - int(math.floor(math.log10(span)))))
    return f"{value:.{decimals}f}"


def fake_responder(model: str, prompt: str) -> str:
    seed = f"{model}|{prompt[:400]}"
    if "You are an assistant to a superforecaster" in prompt:
        return f"{MARK}\nNo search was run. This text stands in for a research report."

    if "Percentile 10: YYYY-MM-DD" in prompt:
        low = datetime.strptime(_LOWER.search(prompt).group(1)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        high = datetime.strptime(_UPPER.search(prompt).group(1)[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)
        span = (high - low).total_seconds()
        centre = 0.3 + 0.4 * _unit(seed, "centre")
        width = 0.08 + 0.15 * _unit(seed, "width")
        lines = []
        for pct, z in zip(_PERCENTILES, _Z):
            moment = low + timedelta(seconds=span * min(0.98, max(0.02, centre + z * width)))
            lines.append(f"Percentile {pct}: {moment:%Y-%m-%d}")
        return f"{MARK}\n" + "\n".join(lines)

    if "Percentile 10: XX" in prompt:
        low = float(_LOWER.search(prompt).group(1).rstrip("."))
        high = float(_UPPER.search(prompt).group(1).rstrip("."))
        span = high - low
        centre = 0.3 + 0.4 * _unit(seed, "centre")
        width = 0.08 + 0.15 * _unit(seed, "width")
        lines = [
            f"Percentile {pct}: {_plain(low + span * min(0.98, max(0.02, centre + z * width)), span)}"
            for pct, z in zip(_PERCENTILES, _Z)
        ]
        return f"{MARK}\n" + "\n".join(lines)

    if "Option_A: Probability_A" in prompt:
        options = ast.literal_eval(_OPTIONS.search(prompt).group(1))
        weights = [0.2 + _unit(seed, f"option{i}") for i in range(len(options))]
        total = sum(weights)
        lines = [f"{option}: {100 * weight / total:.1f}%" for option, weight in zip(options, weights)]
        return f"{MARK}\n" + "\n".join(lines)

    if "Probability: ZZ%" in prompt:
        return f"{MARK}\nProbability: {5 + round(90 * _unit(seed, 'binary'))}%"

    return f"{MARK}\n(no known answer format in the prompt)"
