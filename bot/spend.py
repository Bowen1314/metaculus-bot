"""
Spend guard. Every LLM call goes through `SpendMeter.before_call` / `after_call`.

Costs come from the provider response when present; otherwise they are estimated from token
counts and the price table in config.toml (unknown models are charged at a deliberately high
default so a typo in a model name cannot produce "free" calls). Totals are persisted per UTC day
in the ledger, so the caps hold across runs and restarts.
"""

from __future__ import annotations

import contextvars
import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

from bot.config import Limits, Prices, is_search_model
from bot.ledger import Ledger

logger = logging.getLogger(__name__)


class BudgetExceeded(RuntimeError):
    """Raised before a call that would break a cap. Never retried."""


@dataclass
class QuestionScope:
    label: str
    usd: float = 0.0
    calls: int = 0


_scope: contextvars.ContextVar[QuestionScope | None] = contextvars.ContextVar("spend_scope", default=None)


class SpendMeter:
    def __init__(self, limits: Limits, prices: Prices, ledger: Ledger, season_cap_override: float | None = None) -> None:
        self.limits = limits
        self.prices = prices
        self.ledger = ledger
        self.run_usd = 0.0
        self.run_calls = 0
        self.season_cap = season_cap_override if season_cap_override is not None else limits.max_season_usd

    # ------------------------------------------------------------ scopes

    @contextmanager
    def question(self, label: str) -> Iterator[QuestionScope]:
        scope = QuestionScope(label=label)
        token = _scope.set(scope)
        try:
            yield scope
        finally:
            _scope.reset(token)

    # ------------------------------------------------------------ checks

    def remaining(self) -> dict[str, float]:
        return {
            "run": self.limits.max_run_usd - self.run_usd,
            "day": self.limits.max_day_usd - self.ledger.spend_on(),
            "season": self.season_cap - self.ledger.spend_total(),
        }

    def exhausted_reason(self) -> str | None:
        for name, left in self.remaining().items():
            if left <= 0:
                return f"{name} budget exhausted"
        return None

    def check(self) -> None:
        """Raises BudgetExceeded if no further call may be started. Does not count a call."""
        reason = self.exhausted_reason()
        if reason:
            raise BudgetExceeded(reason)
        scope = _scope.get()
        if scope is not None:
            if scope.usd >= self.limits.max_question_usd:
                raise BudgetExceeded(
                    f"question budget exhausted for {scope.label} (${scope.usd:.2f} >= ${self.limits.max_question_usd:.2f})"
                )
            if scope.calls >= self.limits.max_llm_calls_per_question:
                raise BudgetExceeded(
                    f"LLM call cap reached for {scope.label} ({scope.calls} calls)"
                )

    def before_call(self, model: str) -> None:
        self.check()
        scope = _scope.get()
        if scope is not None:
            scope.calls += 1  # counted when started, so parallel calls cannot overshoot the cap

    def estimate(self, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        price_in, price_out = self.prices.for_model(model)
        usd = (prompt_tokens * price_in + completion_tokens * price_out) / 1_000_000
        if is_search_model(model):
            usd += self.prices.search_call_fee_usd
        return usd

    def after_call(self, model: str, prompt_tokens: int, completion_tokens: int, reported_cost: float | None) -> float:
        estimated = self.estimate(model, prompt_tokens, completion_tokens)
        # Trust the larger figure: a provider may omit search fees, the table may be stale.
        usd = max(float(reported_cost or 0.0), estimated)
        self.run_usd += usd
        self.run_calls += 1
        scope = _scope.get()
        if scope is not None:
            scope.usd += usd
        self.ledger.add_spend(usd, llm_calls=1)
        return usd
