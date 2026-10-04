"""
Shared test helpers. Nothing here (or in any test) touches the network: LLM answers come from
local functions, Metaculus is a stub, and credentials are scrubbed before any library import.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from bot import envguard  # noqa: E402

# Before any library import: no ambient credentials, and never the project's real .env.
envguard.apply(PROJECT_DIR / "tests" / "no-such.env")

import pytest  # noqa: E402
from forecasting_tools import MetaculusClient  # noqa: E402

from bot.config import load_config  # noqa: E402
from bot.fake import fake_responder  # noqa: E402
from bot.ledger import Ledger  # noqa: E402
from bot.llm import LlmFactory  # noqa: E402
from bot.spend import SpendMeter  # noqa: E402

FIXTURES = PROJECT_DIR / "tests" / "fixtures"


@pytest.fixture(autouse=True)
def _no_network(monkeypatch):
    """Any attempt to open a socket fails the test."""
    import socket

    def refuse(*args, **kwargs):
        raise AssertionError("network access attempted in a test")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(autouse=True)
def _clean_env():
    """Each test starts and ends without credentials in the process environment."""
    envguard.scrub_ambient()
    yield
    envguard.scrub_ambient()


def load_posts() -> list[dict]:
    return json.loads((FIXTURES / "posts.json").read_text())


def parse_posts(posts: list[dict]) -> list:
    client = MetaculusClient(token="test-token-not-real")
    questions = []
    for post in posts:
        questions += client._post_json_to_questions_while_handling_groups(copy.deepcopy(post), "unpack_subquestions")
    return questions


@pytest.fixture(scope="session")
def posts() -> list[dict]:
    return load_posts()


@pytest.fixture()
def questions(posts) -> list:
    """Every question in the fixtures, including closed group sub-questions."""
    return parse_posts(posts)


@pytest.fixture()
def open_questions(questions) -> list:
    return [q for q in questions if q.state.value == "open"]


@pytest.fixture()
def by_type(open_questions) -> dict:
    """First open question of each type: binary, multiple_choice, numeric, discrete, date."""
    found: dict = {}
    for question in open_questions:
        found.setdefault(question.question_type, question)
    return found


def log_scaled_post(posts: list[dict]) -> dict:
    """A numeric post turned into a log-scaled one (no such question is in the testing area)."""
    post = copy.deepcopy(next(p for p in posts if p.get("question", {}).get("type") == "numeric"))
    question = post["question"]
    post["id"] = 990001
    question["id"] = 990002
    question["post_id"] = 990001
    question["open_lower_bound"] = False
    question["open_upper_bound"] = True
    question["scaling"].update(
        {"range_min": 1.0, "range_max": 1000.0, "zero_point": 0.0, "nominal_min": None, "nominal_max": None,
         "open_lower_bound": False, "open_upper_bound": True}
    )
    question["scaling"].pop("continuous_range", None)
    return post


@pytest.fixture()
def config(tmp_path):
    base = load_config(PROJECT_DIR / "config.toml")
    return dataclasses.replace(
        base,
        armed=False,
        state_dir=tmp_path / "state",
        log_dir=tmp_path / "logs",
        models=dataclasses.replace(base.models, allowed_tries=1, allowed_tries_by_model={}),
    )


@pytest.fixture()
def ledger(tmp_path):
    book = Ledger(tmp_path / "state" / "ledger.sqlite3")
    yield book
    book.close()


class StubClient:
    """Stands in for MetaculusClient. Records every post; can be told to fail."""

    def __init__(self, questions_by_tournament: dict | None = None) -> None:
        self.questions_by_tournament = questions_by_tournament or {}
        self.forecasts: list[tuple[int, dict]] = []
        self.comments: list[tuple[int, str]] = []
        self.fetches: list = []
        self.fail_fetch: Exception | None = None
        self.fail_forecast_times = 0  # raise before recording (the forecast does not land)
        self.fail_forecast_after_landing_times = 0  # record, then raise (it lands, caller sees an error)
        self.fail_comment_times = 0
        self.fail_lookup = False

    # --- reads
    def _all(self):
        return [q for qs in self.questions_by_tournament.values() for q in qs]

    def _landed(self) -> set[int]:
        return {qid for qid, _ in self.forecasts}

    def get_all_open_questions_from_tournament(self, tournament_id):
        self.fetches.append(tournament_id)
        if self.fail_fetch is not None:
            raise self.fail_fetch
        result = []
        for question in self.questions_by_tournament.get(tournament_id, []):
            if question.state.value != "open":
                continue
            fresh = question.model_copy(deep=True)
            fresh.already_forecasted = bool(question.already_forecasted) or question.id_of_question in self._landed()
            result.append(fresh)
        return result

    def get_question_by_post_id(self, post_id, group_question_mode="exclude"):
        if self.fail_lookup:
            raise ConnectionError("stub: lookup failed")
        found = []
        for question in self._all():
            if question.id_of_post == post_id:
                fresh = question.model_copy(deep=True)
                fresh.already_forecasted = bool(question.already_forecasted) or question.id_of_question in self._landed()
                found.append(fresh)
        return found if len(found) != 1 else found[0]

    # --- writes
    def _forecast(self, question_id: int, payload: dict) -> None:
        if self.fail_forecast_times > 0:
            self.fail_forecast_times -= 1
            raise ConnectionError("stub: forecast POST failed")
        self.forecasts.append((question_id, payload))
        if self.fail_forecast_after_landing_times > 0:
            self.fail_forecast_after_landing_times -= 1
            raise TimeoutError("stub: forecast POST timed out after the server accepted it")

    def post_binary_question_prediction(self, question_id, prediction_in_decimal):
        self._forecast(question_id, {"probability_yes": prediction_in_decimal})

    def post_multiple_choice_question_prediction(self, question_id, options_with_probabilities):
        self._forecast(question_id, {"probability_yes_per_category": dict(options_with_probabilities)})

    def post_numeric_question_prediction(self, question_id, cdf_values):
        self._forecast(question_id, {"continuous_cdf": list(cdf_values)})

    def post_question_comment(self, post_id, comment_text, is_private=True, included_forecast=True):
        if self.fail_comment_times > 0:
            self.fail_comment_times -= 1
            raise ConnectionError("stub: comment POST failed")
        self.comments.append((post_id, comment_text))


class CountingResponder:
    """Wraps a responder, counting calls per model and optionally misbehaving for some models."""

    def __init__(self, inner=fake_responder) -> None:
        self.inner = inner
        self.calls: list[tuple[str, str]] = []
        self.raise_for: dict[str, Exception] = {}  # substring of model name -> exception
        self.text_for: dict[str, str] = {}  # substring of model name -> fixed answer

    def count(self, substring: str = "") -> int:
        return sum(1 for model, _ in self.calls if substring in model)

    def forecast_calls(self) -> int:
        return sum(1 for model, prompt in self.calls if "professional forecaster" in prompt)

    def __call__(self, model: str, prompt: str) -> str:
        self.calls.append((model, prompt))
        for key, error in self.raise_for.items():
            if key in model:
                raise error
        for key, text in self.text_for.items():
            if key in model and "professional forecaster" in prompt:
                return text
        return self.inner(model, prompt)


@pytest.fixture()
def responder() -> CountingResponder:
    return CountingResponder()


@pytest.fixture()
def make_factory(config, ledger):
    def build(responder, *, metered: bool = True, cfg=None):
        cfg = cfg or config
        meter = SpendMeter(cfg.limits, cfg.prices, ledger) if metered else None
        return LlmFactory(cfg, meter, responder=responder), meter

    return build


def with_future_close(questions: list, hours: float = 3.0) -> list:
    """Copies of the questions that close `hours` from now (tests must not depend on the calendar)."""
    from datetime import datetime, timedelta, timezone

    result = []
    for question in questions:
        fresh = question.model_copy(deep=True)
        fresh.close_time = datetime.now(timezone.utc) + timedelta(hours=hours)
        result.append(fresh)
    return result


@pytest.fixture()
def live_questions(open_questions) -> list:
    return with_future_close(open_questions)


@pytest.fixture()
def fast_sleep(monkeypatch):
    """Library retry pauses (asyncio.sleep between parser retries) take no time."""
    import asyncio

    real_sleep = asyncio.sleep

    async def instant(delay, *args, **kwargs):
        await real_sleep(0)

    monkeypatch.setattr(asyncio, "sleep", instant)
