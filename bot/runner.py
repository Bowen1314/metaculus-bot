"""
One unattended pass: fetch open questions, forecast the ones not yet done, submit (live) or
print (dry run). Everything that touches Metaculus or an LLM is injected, so this module is
fully testable offline.

Live submission order for one question (see bot/ledger.py):
    ledger `submitting` (payload stored) -> POST forecast -> ledger `submitted`
    -> POST private comment -> ledger `done`
A crash anywhere is repaired by the next run: `submitting` rows are checked against the server
before anything is posted again, and the stored payload (not a new forecast) is what is re-sent.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable

from forecasting_tools import (
    ApiFilter,
    BinaryQuestion,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.data_models.questions import ConditionalQuestion, DateQuestion

from bot import aggregate, numeric
from bot.config import Config, Profile, Tournament
from bot.envguard import redact
from bot.forecaster import EnsembleBot, UnsupportedQuestion, cdf_points
from bot.ledger import FINAL_STATES, Ledger
from bot.llm import LlmFactory
from bot.research import Researcher
from bot.spend import BudgetExceeded, SpendMeter

logger = logging.getLogger(__name__)

MAX_COMMENT_ATTEMPTS = 6
PAGE_SIZE = 100
MAX_PAGES = 6


def fetch_open_questions(client: Any, tournament_id: str | int) -> list[MetaculusQuestion]:
    """All open questions of a tournament, newest first.

    The library's own `get_all_open_questions_from_tournament` reads only the first page
    (100 posts); this walks further pages when the first one is full. Falls back to the
    library call if its paging helper is not there."""
    grab = getattr(client, "_grab_filtered_questions_with_offset", None)
    if grab is None:
        return list(client.get_all_open_questions_from_tournament(tournament_id))
    api_filter = ApiFilter(
        allowed_tournaments=[tournament_id],
        allowed_statuses=["open"],
        group_question_mode="unpack_subquestions",
    )
    questions: list[MetaculusQuestion] = []
    seen: set[Any] = set()
    for page in range(MAX_PAGES):
        try:
            found, more = grab(api_filter, page * PAGE_SIZE)
        except Exception:
            if page == 0:
                raise
            logger.warning("page %d of %s could not be read; using what was fetched", page + 1, tournament_id)
            break
        for question in found:
            key = (question.id_of_post, question.id_of_question)
            if key not in seen:
                seen.add(key)
                questions.append(question)
        # A full page holds 100 posts (each at least one question, a few may be notebooks).
        if not more or len(found) < PAGE_SIZE - 10:
            break
    return questions


class NotArmed(RuntimeError):
    """Live submission was requested without the second gate (armed = true in config.toml)."""


@dataclass
class Submission:
    question_id: int
    post_id: int
    qtype: str
    payload: dict[str, Any]
    comment: str
    readable: str


@dataclass
class RunOptions:
    live: bool = False
    tournaments: tuple[Tournament, ...] = ()
    limit: int | None = None
    verbose: bool = False
    # `submit-test`: real posts, but only to the unscored testing tournament; needs no `armed`.
    testing_submit: bool = False


@dataclass
class Summary:
    mode: str
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds"))
    stopped_reason: str | None = None
    tournaments: dict[str, dict[str, int]] = field(default_factory=dict)
    recovered: int = 0
    comments_posted: int = 0
    spend_usd: float = 0.0
    llm_calls: int = 0

    def bucket(self, name: str) -> dict[str, int]:
        return self.tournaments.setdefault(name, {})

    def count(self, tournament: str, key: str, amount: int = 1) -> None:
        bucket = self.bucket(tournament)
        bucket[key] = bucket.get(key, 0) + amount

    def as_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "started_at": self.started_at,
            "stopped_reason": self.stopped_reason,
            "tournaments": self.tournaments,
            "recovered": self.recovered,
            "comments_posted": self.comments_posted,
            "spend_usd": round(self.spend_usd, 4),
            "llm_calls": self.llm_calls,
        }


# ---------------------------------------------------------------------- payloads


def question_type(question: MetaculusQuestion) -> str:
    return getattr(question, "question_type", type(question).__name__)


def build_submission(question: MetaculusQuestion, report: Any, bot: EnsembleBot) -> Submission:
    """Turns a report into the exact JSON the forecast endpoint takes, and validates it against
    the server's rules. Raises ValueError rather than return something the server would reject."""
    if question.id_of_question is None or question.id_of_post is None:
        raise ValueError("question has no question id / post id")
    qid, post_id = int(question.id_of_question), int(question.id_of_post)
    prediction = report.prediction

    if isinstance(question, BinaryQuestion):
        probability = float(prediction)
        problems = aggregate.validate_binary_payload(probability)
        payload: dict[str, Any] = {"probability_yes": round(probability, 6)}
        readable = f"{probability:.1%}"
    elif isinstance(question, MultipleChoiceQuestion):
        options = list(question.options)
        values = {o.option_name: float(o.probability) for o in prediction.predicted_options}
        problems = aggregate.validate_option_payload(values, options)
        payload = {"probability_yes_per_category": {o: values[o] for o in options} if not problems else values}
        readable = ", ".join(f"{o}: {values.get(o, float('nan')):.1%}" for o in options)
    elif isinstance(question, (NumericQuestion, DateQuestion)):
        rules = dict(
            cdf_size=question.cdf_size,
            open_lower_bound=question.open_lower_bound,
            open_upper_bound=question.open_upper_bound,
        )
        cdf = bot.final_cdfs.get(qid)
        if cdf is None:
            cdf = [point.percentile for point in cdf_points(prediction)]
        cdf = numeric.ensure_valid_cdf(cdf, **rules)
        problems = numeric.validate_cdf(cdf, **rules)
        payload = {"continuous_cdf": cdf}
        x_axis = [point.value for point in prediction.declared_percentiles]
        if len(x_axis) != len(cdf):
            x_axis = [point.value for point in cdf_points(prediction)]
        marks = numeric.percentile_values_from_cdf(cdf, x_axis)
        is_date = isinstance(question, DateQuestion)
        readable = ", ".join(f"p{round(t * 100)}={_fmt_value(v, is_date)}" for t, v in marks)
        readable += f"  [cdf {len(cdf)} points, {cdf[0]:.4f} .. {cdf[-1]:.4f}]"
    else:
        raise UnsupportedQuestion(f"unsupported question type {type(question).__name__}")

    if problems:
        raise ValueError(f"forecast would be rejected by Metaculus: {problems}")
    comment = (report.explanation or "").strip()
    if not comment:
        raise ValueError("empty comment; a reasoning comment is required for prize eligibility")
    return Submission(qid, post_id, question_type(question), payload, comment, readable)


def _fmt_value(value: float, is_date: bool) -> str:
    if is_date:
        return datetime.fromtimestamp(value, tz=timezone.utc).strftime("%Y-%m-%d")
    return f"{value:.6g}"


def post_forecast(client: Any, question_id: int, payload: dict[str, Any]) -> None:
    if "probability_yes" in payload:
        client.post_binary_question_prediction(question_id, payload["probability_yes"])
    elif "probability_yes_per_category" in payload:
        client.post_multiple_choice_question_prediction(question_id, payload["probability_yes_per_category"])
    elif "continuous_cdf" in payload:
        client.post_numeric_question_prediction(question_id, payload["continuous_cdf"])
    else:
        raise ValueError("unknown payload shape")


# ---------------------------------------------------------------------- credentials / credit


def missing_credentials(models: list[str]) -> list[str]:
    """Names of the .env variables the configured models need and that are not loaded."""
    needed: set[str] = set()
    for model in models:
        if model.startswith("openrouter/"):
            needed.add("OPENROUTER_API_KEY")
        elif model.startswith("metaculus/"):
            needed.add("METACULUS_TOKEN")
        elif model.startswith(("anthropic/", "claude")):
            needed.add("ANTHROPIC_API_KEY")
        elif model.startswith(("gemini/", "google/")):
            needed.add("GEMINI_API_KEY")
        elif model.startswith("perplexity/"):
            needed.add("PERPLEXITY_API_KEY")
        else:
            needed.add("OPENAI_API_KEY")
    return sorted(name for name in needed if not os.environ.get(name))


def openrouter_remaining_usd(timeout: float = 15.0) -> float | None:
    """Credit left on the OpenRouter key, or None if unknown (no key, no limit, or the call failed)."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None
    try:
        import requests

        response = requests.get(
            "https://openrouter.ai/api/v1/key",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
        response.raise_for_status()
        remaining = response.json().get("data", {}).get("limit_remaining")
        return float(remaining) if remaining is not None else None
    except Exception as error:
        logger.warning("could not read OpenRouter credit: %s", error.__class__.__name__)
        return None


# ---------------------------------------------------------------------- runner


class Runner:
    def __init__(
        self,
        *,
        config: Config,
        ledger: Ledger,
        client: Any,
        factory: LlmFactory,
        meter: SpendMeter | None,
        options: RunOptions,
        credit_check: Callable[[], float | None] | None = None,
        asknews: Any = None,
        out: Callable[[str], None] = print,
    ) -> None:
        if options.live:
            if factory.is_fake:
                raise NotArmed("fake LLM answers can never be submitted")
            if options.testing_submit:
                wrong = [t.name for t in options.tournaments if str(t.id) != str(config.testing.id)]
                if wrong:
                    raise NotArmed("submit-test may only target the testing tournament")
            elif not config.armed:
                raise NotArmed("live submission needs armed = true in config.toml as well as --live")
        self.config = config
        self.ledger = ledger
        self.client = client
        self.factory = factory
        self.meter = meter
        self.options = options
        self.credit_check = credit_check
        self.asknews = asknews
        self.out = out
        self.summary = Summary(mode=self._mode_name())
        self._started = time.monotonic()
        self._confirmed_by_listing = False

    def _mode_name(self) -> str:
        if not self.options.live:
            return "dry-run-fake-llm" if self.factory.is_fake else "dry-run"
        return "submit-test" if self.options.testing_submit else "live"

    # ------------------------------------------------------------------ entry point

    def run(self) -> Summary:
        run_id = self.ledger.start_run(self.summary.mode)
        try:
            fetched = self._fetch_all()  # sync: the library fetch runs its own event loop
            asyncio.run(self._process_all(fetched))
        finally:
            if self.meter is not None:
                self.summary.spend_usd = self.meter.run_usd
                self.summary.llm_calls = self.meter.run_calls
            self.ledger.finish_run(run_id, self.summary.as_dict())
        return self.summary

    def _fetch_all(self) -> list[tuple[Tournament, list[MetaculusQuestion]]]:
        fetched = []
        for tournament in self.options.tournaments:
            try:
                questions = fetch_open_questions(self.client, tournament.id)
            except Exception as error:
                logger.error(
                    "could not fetch questions for %s: %s: %s",
                    tournament.name,
                    error.__class__.__name__,
                    redact(str(error))[:300],
                )
                self.summary.count(tournament.name, "fetch_failed")
                continue
            self.summary.count(tournament.name, "open", len(questions))
            fetched.append((tournament, questions))
        return fetched

    async def _process_all(self, fetched: list[tuple[Tournament, list[MetaculusQuestion]]]) -> None:
        if self.options.live:
            await self._recover_unconfirmed()
            await self._post_pending_comments()
        preflight_done = False
        for tournament, questions in fetched:
            todo = self._select(tournament, questions)
            if self._confirmed_by_listing:
                self._confirmed_by_listing = False
                await self._post_pending_comments()  # the comment for what the listing just confirmed
            if not todo:
                continue
            if not preflight_done:
                preflight_done = True
                blocked = self._preflight()
                if blocked:
                    self.summary.stopped_reason = blocked
                    logger.error("not forecasting: %s", blocked)
                    return
            if self.meter is not None and tournament.min_season_budget_left_usd > 0:
                if self.meter.remaining()["season"] < tournament.min_season_budget_left_usd:
                    logger.warning("skipping %s: season budget is being kept for earlier tournaments", tournament.name)
                    self.summary.count(tournament.name, "skipped_budget", len(todo))
                    continue
            await self._forecast_tournament(tournament, todo)
            if self.summary.stopped_reason:
                return

    # ------------------------------------------------------------------ selection

    def _select(self, tournament: Tournament, questions: list[MetaculusQuestion]) -> list[MetaculusQuestion]:
        limits = self.config.limits
        now = datetime.now(timezone.utc)
        todo: list[MetaculusQuestion] = []
        for question in questions:
            name = tournament.name
            if isinstance(question, ConditionalQuestion) or question.id_of_question is None or question.id_of_post is None:
                self.summary.count(name, "unsupported")
                logger.warning("skipping unsupported question (post %s, %s)", question.id_of_post, question_type(question))
                continue
            qid = int(question.id_of_question)
            if question.close_time is not None:
                seconds_left = (question.close_time - now).total_seconds()
                if seconds_left < limits.min_seconds_before_close:
                    self.summary.count(name, "closing_too_soon")
                    continue
            if self.options.live:
                status = self.ledger.status(qid)
                if status in FINAL_STATES:
                    self.summary.count(name, "already_done")
                    continue
                if status == "submitting":
                    if question.already_forecasted:
                        # The listing itself confirms that the earlier POST landed.
                        self.ledger.mark_submitted(qid)
                        self._confirmed_by_listing = True
                        self.summary.recovered += 1
                        self.summary.count(name, "already_done")
                    else:
                        self.summary.count(name, "awaiting_confirmation")
                    continue
                if question.already_forecasted:
                    # A forecast exists on Metaculus that this ledger did not make (state lost,
                    # or made by hand). Never forecast twice.
                    self.ledger.record_seen(qid, self._meta(tournament, question))
                    self.ledger.mark_server_already(qid)
                    self.summary.count(name, "already_on_server")
                    continue
                if self.ledger.attempts(qid) >= limits.max_attempts_per_question:
                    self.ledger.mark_gave_up(qid, f"gave up after {self.ledger.attempts(qid)} attempts")
                    self.summary.count(name, "gave_up")
                    continue
            todo.append(question)
        todo.sort(key=lambda q: q.close_time or datetime.max.replace(tzinfo=timezone.utc))
        if self.options.limit is not None:
            todo = todo[: self.options.limit]
        return todo

    def _meta(self, tournament: Tournament, question: MetaculusQuestion) -> dict[str, Any]:
        return {
            "post_id": question.id_of_post,
            "tournament": tournament.name,
            "qtype": question_type(question),
            "title": (question.question_text or "")[:300],
            "url": question.page_url,
            "close_time": question.close_time.isoformat() if question.close_time else None,
        }

    def _preflight(self) -> str | None:
        """Checks done once, only when there is something to forecast. Returns a reason to stop."""
        if self.factory.is_fake:
            return None
        models: list[str] = [self.config.models.parser]
        for tournament in self.options.tournaments:
            profile = self.config.profile(tournament.profile)
            models += [*profile.forecasters, *profile.search_models]
        missing = missing_credentials(models)
        if missing:
            return f"missing in .env: {', '.join(missing)}"
        if self.meter is not None:
            reason = self.meter.exhausted_reason()
            if reason:
                return reason
        if self.credit_check is not None:
            remaining = self.credit_check()
            if remaining is not None and remaining < self.config.limits.openrouter_min_remaining_usd:
                return f"OpenRouter credit is down to ${remaining:.2f}"
        return None

    # ------------------------------------------------------------------ forecasting

    def _make_bot(self, profile: Profile) -> EnsembleBot:
        researcher = Researcher(self.config, self.factory, self.ledger if not self.factory.is_fake else None, asknews=self.asknews)
        return EnsembleBot(
            config=self.config,
            profile=profile,
            factory=self.factory,
            researcher=researcher,
            metaculus_client=self.client,
        )

    async def _forecast_tournament(self, tournament: Tournament, todo: list[MetaculusQuestion]) -> None:
        limits = self.config.limits
        bot = self._make_bot(self.config.profile(tournament.profile))
        gate = asyncio.Semaphore(max(1, limits.max_concurrent_questions))

        async def one(question: MetaculusQuestion) -> None:
            async with gate:
                if self.summary.stopped_reason:
                    self.summary.count(tournament.name, "deferred")
                    return
                if time.monotonic() - self._started > limits.run_time_budget_s:
                    self.summary.count(tournament.name, "deferred")
                    return
                await self._forecast_one(tournament, bot, question)

        await asyncio.gather(*(one(question) for question in todo))

    async def _forecast_one(self, tournament: Tournament, bot: EnsembleBot, question: MetaculusQuestion) -> None:
        name = tournament.name
        qid = int(question.id_of_question)  # type: ignore[arg-type]
        label = f"{name}/q{qid}"
        live = self.options.live
        if live:
            self.ledger.record_seen(qid, self._meta(tournament, question))
        started = time.monotonic()
        cost = 0.0
        try:
            if self.meter is not None:
                with self.meter.question(label) as scope:
                    try:
                        report = await asyncio.wait_for(
                            bot.forecast_question(question), timeout=self.config.limits.question_timeout_s
                        )
                    finally:
                        cost = scope.usd
            else:
                report = await asyncio.wait_for(
                    bot.forecast_question(question), timeout=self.config.limits.question_timeout_s
                )
            submission = build_submission(question, report, bot)
        except BaseException as error:  # includes ExceptionGroup from the library and TimeoutError
            if isinstance(error, (KeyboardInterrupt, SystemExit, asyncio.CancelledError)):
                raise
            message = f"{error.__class__.__name__}: {redact(str(error))}"[:1500]
            # The library turns the runs' exceptions into text, so the budget is asked directly
            # rather than looked for in the exception chain.
            if self.meter is not None and self.meter.exhausted_reason():
                # run/day/season cap: stop the whole run; not the question's fault, no attempt used
                self.summary.stopped_reason = self.meter.exhausted_reason()
                self.summary.count(name, "deferred")
                logger.warning("%s stopped: %s", label, self.summary.stopped_reason)
                return
            self.summary.count(name, "failed")
            logger.error("%s failed after %.0fs: %s", label, time.monotonic() - started, message)
            if live:
                self.ledger.record_attempt_failed(qid, message, cost)
            return

        elapsed = time.monotonic() - started
        self._print(tournament, question, bot, submission, cost, elapsed)
        if not live:
            self.ledger.record_dry_run(qid, self._meta(tournament, question), submission.payload, cost)
            self.summary.count(name, "forecast_not_submitted")
            return
        await self._submit(name, submission, cost)

    async def _submit(self, name: str, submission: Submission, cost: float) -> None:
        qid = submission.question_id
        self.ledger.mark_submitting(qid, submission.payload, submission.comment, cost)
        try:
            await asyncio.to_thread(post_forecast, self.client, qid, submission.payload)
        except Exception as error:
            # The POST may have landed. Leave `submitting`; the next run asks the server.
            self.ledger.note_submit_error(qid, f"{error.__class__.__name__}: {redact(str(error))}")
            self.summary.count(name, "submit_unconfirmed")
            logger.error("q%s: forecast POST raised %s; will confirm on the next run", qid, error.__class__.__name__)
            return
        self.ledger.mark_submitted(qid)
        self.summary.count(name, "submitted")
        logger.info("q%s: forecast submitted", qid)
        await self._comment(qid, submission.post_id, submission.comment)

    async def _comment(self, question_id: int, post_id: int, text: str) -> bool:
        try:
            await asyncio.to_thread(self.client.post_question_comment, post_id, text)
        except Exception as error:
            self.ledger.mark_comment_failed(question_id, f"comment: {error.__class__.__name__}: {redact(str(error))}")
            logger.error("q%s: comment POST raised %s; will retry on the next run", question_id, error.__class__.__name__)
            return False
        self.ledger.mark_done(question_id)
        self.summary.comments_posted += 1
        return True

    # ------------------------------------------------------------------ recovery

    def _server_state(self, row: dict[str, Any]) -> tuple[bool, bool] | None:
        """(has our forecast, is open) for a ledger row, or None if the server could not be asked."""
        try:
            result = self.client.get_question_by_post_id(row["post_id"], group_question_mode="unpack_subquestions")
        except Exception as error:
            logger.warning("q%s: could not confirm server state: %s", row["question_id"], error.__class__.__name__)
            return None
        for question in result if isinstance(result, list) else [result]:
            if question.id_of_question == row["question_id"]:
                is_open = getattr(question.state, "value", question.state) == "open"
                return bool(question.already_forecasted), is_open
        return None

    async def _recover_unconfirmed(self) -> None:
        for row in self.ledger.in_state("submitting"):
            qid = row["question_id"]
            state = await asyncio.to_thread(self._server_state, row)
            if state is None:
                continue
            has_forecast, is_open = state
            if has_forecast:
                self.ledger.mark_submitted(qid)
                self.summary.recovered += 1
                logger.info("q%s: earlier submission confirmed on the server", qid)
                continue
            if not is_open:
                self.ledger.mark_gave_up(qid, "closed before the submission could be confirmed")
                continue
            if self.ledger.attempts(qid) >= self.config.limits.max_attempts_per_question:
                self.ledger.mark_gave_up(qid, f"submission failed {self.ledger.attempts(qid)} times: {row.get('last_error')}")
                continue
            # Re-send the stored forecast; do not ask the models again.
            self.ledger.bump_attempts(qid)
            try:
                await asyncio.to_thread(post_forecast, self.client, qid, json.loads(row["payload_json"]))
            except Exception as error:
                self.ledger.note_submit_error(qid, f"{error.__class__.__name__}: {redact(str(error))}")
                logger.error("q%s: re-submission raised %s", qid, error.__class__.__name__)
                continue
            self.ledger.mark_submitted(qid)
            self.summary.recovered += 1
            logger.info("q%s: stored forecast re-submitted", qid)

    async def _post_pending_comments(self) -> None:
        for row in self.ledger.pending_comments(MAX_COMMENT_ATTEMPTS):
            await self._comment(row["question_id"], row["post_id"], row["comment_text"])

    # ------------------------------------------------------------------ output

    def _print(self, tournament: Tournament, question: MetaculusQuestion, bot: EnsembleBot, submission: Submission, cost: float, elapsed: float) -> None:
        if self.options.live and not self.options.verbose:
            logger.info("q%s forecast ready: %s", submission.question_id, submission.readable)
            return
        lines = [
            "",
            f"[{submission.qtype}] question {submission.question_id} (post {submission.post_id}) in {tournament.name}",
            f"  title:   {(question.question_text or '').strip()[:160]}",
        ]
        if question.group_question_option:
            lines.append(f"  option:  {question.group_question_option}")
        if question.close_time:
            lines.append(f"  closes:  {question.close_time.isoformat(timespec='minutes')}")
        for run in bot.run_details.get(submission.question_id, []):
            lines.append(f"  run:     {run['model']} -> {run['prediction']}  (parsed by {run['parsed_by']})")
        lines.append(f"  FINAL:   {submission.readable}")
        payload_text = json.dumps(submission.payload)
        if len(payload_text) > 240 and not self.options.verbose:
            payload_text = payload_text[:240] + f"... ({len(payload_text)} chars)"
        lines.append(f"  payload: {payload_text}")
        lines.append(f"  comment: {len(submission.comment)} chars; cost ${cost:.3f}; {elapsed:.0f}s")
        if self.options.verbose:
            lines.append("  ----- comment -----")
            lines.append(submission.comment)
        self.out("\n".join(lines))


def _find(error: BaseException, kind: type) -> BaseException | None:
    """Looks for an exception of `kind` in an exception, its causes and (nested) groups."""
    seen: set[int] = set()
    stack = [error]
    while stack:
        current = stack.pop()
        if current is None or id(current) in seen:
            continue
        seen.add(id(current))
        if isinstance(current, kind):
            return current
        stack.extend(getattr(current, "exceptions", ()) or ())
        stack.append(current.__cause__)
        stack.append(current.__context__)
    return None
