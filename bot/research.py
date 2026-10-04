"""
Research step: several independent sources in parallel, each allowed to fail.

Sources are (a) frontier models with the provider's own web search (":online" on OpenRouter) and
(b) AskNews, if credentials are in .env and the monthly allowance is not used up. The question
is only forecast if at least one source answered (unless the config says otherwise).
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import Awaitable, Callable

from forecasting_tools import MetaculusQuestion
from forecasting_tools.util.misc import clean_indents

from bot.config import Config, Profile
from bot.envguard import redact
from bot.ledger import Ledger
from bot.llm import LlmFactory
from bot.spend import BudgetExceeded

logger = logging.getLogger(__name__)

ASKNEWS_CALLS_PER_QUESTION_TIMEOUT_S = 120

AskNewsFn = Callable[[str], Awaitable[str]]


class ResearchFailed(RuntimeError):
    """No research source produced anything."""


def research_prompt(question: MetaculusQuestion, today: datetime | None = None) -> str:
    """The official template's research prompt, plus today's date and the question's dates."""
    today = today or datetime.now(timezone.utc)
    dates = [f"Today is {today.strftime('%Y-%m-%d')}."]
    if question.scheduled_resolution_time:
        dates.append(f"The question is scheduled to resolve on {question.scheduled_resolution_time.strftime('%Y-%m-%d')}.")
    return clean_indents(
        f"""
        You are an assistant to a superforecaster.
        The superforecaster will give you a question they intend to forecast on.
        To be a great assistant, you generate a concise but detailed rundown of the most relevant news, including if the question would resolve Yes or No based on current information.
        Search the web. Report the latest relevant facts with their dates and sources, the current value or status of whatever the question measures, relevant base rates or historical frequencies, and any prediction-market or expert estimates you find.
        You do not produce forecasts yourself.

        {' '.join(dates)}

        Question:
        {question.question_text}

        This question's outcome will be determined by the specific criteria below:
        {question.resolution_criteria}

        {question.fine_print}
        """
    )


def _error_excerpt(error: BaseException, limit: int = 800) -> str:
    """The error text on one line, long enough to keep the provider's own reason (quota name etc.)."""
    return re.sub(r"(?:\\n|\s)+", " ", redact(str(error)))[:limit]


def asknews_configured() -> bool:
    return bool(
        (os.environ.get("ASKNEWS_CLIENT_ID") and os.environ.get("ASKNEWS_SECRET"))
        or os.environ.get("ASKNEWS_API_KEY")
    )


async def _library_asknews(query: str) -> str:
    from forecasting_tools.helpers.asknews_searcher import AskNewsSearcher

    return await AskNewsSearcher().call_preconfigured_version("asknews/news-summaries", query)


def _asknews_counter_name(now: datetime | None = None) -> str:
    return "asknews:" + (now or datetime.now(timezone.utc)).strftime("%Y-%m")


class Researcher:
    def __init__(
        self,
        config: Config,
        factory: LlmFactory,
        ledger: Ledger | None,
        asknews: AskNewsFn | None = None,
    ) -> None:
        self.config = config
        self.factory = factory
        self.ledger = ledger
        # Never call AskNews in fake mode; in normal mode only if credentials are in .env.
        if asknews is not None:
            self._asknews: AskNewsFn | None = asknews
        elif not factory.is_fake and asknews_configured():
            self._asknews = _library_asknews
        else:
            self._asknews = None

    def _asknews_allowed(self, profile: Profile) -> bool:
        if not profile.use_asknews or self._asknews is None:
            return False
        if self.ledger is None:
            return True
        limits = self.config.limits
        used = self.ledger.counter(_asknews_counter_name())
        return used + limits.asknews_units_per_question <= limits.asknews_monthly_call_units

    async def _search(self, model: str, prompt: str) -> str:
        text = await self.factory.searcher(model).invoke(prompt)
        if not text or not text.strip():
            raise RuntimeError("empty research answer")
        return text.strip()

    async def _news(self, question: MetaculusQuestion) -> str:
        assert self._asknews is not None
        if self.ledger is not None:
            # counted before the call: a failed call may still have used the allowance
            self.ledger.add_counter(_asknews_counter_name(), self.config.limits.asknews_units_per_question)
        text = await asyncio.wait_for(
            self._asknews(question.question_text), timeout=ASKNEWS_CALLS_PER_QUESTION_TIMEOUT_S
        )
        if not text or not text.strip():
            raise RuntimeError("empty AskNews answer")
        return text.strip()

    async def run(self, question: MetaculusQuestion, profile: Profile) -> str:
        prompt = research_prompt(question)
        labels: list[str] = []
        jobs: list[Awaitable[str]] = []
        for model in profile.search_models:
            labels.append(f"Web search ({model.removeprefix('openrouter/')})")
            jobs.append(self._search(model, prompt))
        if self._asknews_allowed(profile):
            labels.append("News (AskNews)")
            jobs.append(self._news(question))

        results = await asyncio.gather(*jobs, return_exceptions=True) if jobs else []
        limit = self.config.limits.research_chars_per_source
        sections: list[str] = []
        budget_error: BudgetExceeded | None = None
        for label, result in zip(labels, results):
            if isinstance(result, BudgetExceeded):
                budget_error = result
                continue
            if isinstance(result, BaseException):
                logger.warning(
                    "research source failed: %s: %s: %s",
                    label,
                    result.__class__.__name__,
                    _error_excerpt(result),
                )
                continue
            text = result if len(result) <= limit else result[:limit] + "\n[truncated]"
            sections.append(f"## {label}\n{text}")

        if not sections:
            if budget_error is not None:
                raise budget_error
            if self.config.ensemble.allow_forecast_without_research:
                logger.warning("no research source answered; forecasting without research")
                return "No research was available for this question."
            raise ResearchFailed(f"all {len(jobs)} research sources failed")
        logger.info("research: %d of %d sources answered", len(sections), len(jobs))
        return "\n\n".join(sections)
