"""The forecasting bot end to end with local stand-in models: every type, and every failure path."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import time

import pytest
from forecasting_tools import NumericDistribution

from bot import aggregate, forecaster, numeric
from bot.forecaster import EnsembleBot
from bot.research import Researcher, ResearchFailed
from bot.runner import _find, build_submission
from bot.spend import BudgetExceeded
from tests.conftest import CountingResponder, StubClient, log_scaled_post, parse_posts

PARSER_PROMPT = "data analyst helping to convert text"


def make_bot(config, factory, *, profile="full", client=None, ledger=None, asknews=None) -> EnsembleBot:
    researcher = Researcher(config, factory, ledger, asknews=asknews)
    return EnsembleBot(
        config=config,
        profile=config.profile(profile),
        factory=factory,
        researcher=researcher,
        metaculus_client=client or StubClient(),
    )


def forecast(bot, question):
    return asyncio.run(bot.forecast_question(question))


def runs_of(bot, question) -> list[dict]:
    return bot.run_details[question.id_of_question]


# ------------------------------------------------------------------ happy path, every type

@pytest.mark.parametrize("qtype", ["binary", "multiple_choice", "numeric", "discrete", "date"])
def test_every_question_type_yields_a_server_valid_forecast(config, make_factory, responder, by_type, qtype):
    factory, meter = make_factory(responder)
    client = StubClient()
    bot = make_bot(config, factory, client=client)
    question = by_type[qtype]

    report = forecast(bot, question)
    submission = build_submission(question, report, bot)

    assert submission.qtype == qtype
    assert submission.question_id == question.id_of_question and submission.post_id == question.id_of_post
    if qtype == "binary":
        assert aggregate.validate_binary_payload(submission.payload["probability_yes"]) == []
        assert 0.03 <= submission.payload["probability_yes"] <= 0.97
    elif qtype == "multiple_choice":
        values = submission.payload["probability_yes_per_category"]
        assert list(values) == question.options
        assert aggregate.validate_option_payload(values, question.options) == []
        assert min(values.values()) >= 0.02 - 1e-12
    else:
        cdf = submission.payload["continuous_cdf"]
        assert len(cdf) == question.cdf_size
        assert numeric.validate_cdf(cdf, cdf_size=question.cdf_size, open_lower_bound=question.open_lower_bound,
                                    open_upper_bound=question.open_upper_bound) == []
    json.dumps(submission.payload)  # what goes on the wire is plain JSON
    assert submission.comment.startswith("#") and "Forecaster 5" in submission.comment and "NOTES" in submission.comment

    assert len(runs_of(bot, question)) == 5 and all(run["parsed_by"] == "regex" for run in runs_of(bot, question))
    assert responder.forecast_calls() == 5  # five runs ...
    assert responder.count(":online") == 2  # ... on one shared research step with two sources
    assert responder.count("gpt-6-luna") == 0  # the LLM parser was never needed
    assert client.forecasts == [] and client.comments == []  # the bot itself never posts
    assert meter.run_calls == 7


def test_runs_are_spread_over_three_labs_and_see_the_research(config, make_factory, responder, by_type):
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    forecast(bot, by_type["binary"])
    models = sorted(run["model"] for run in runs_of(bot, by_type["binary"]))
    assert models == sorted(m.removeprefix("openrouter/") for m in config.profile("full").forecasters)
    assert {m.split("/")[0] for m in models} == {"openai", "anthropic", "google"}
    forecast_prompts = [prompt for _, prompt in responder.calls if "professional forecaster" in prompt]
    assert all("## Web search (" in prompt and "Synthetic resolution criteria" in prompt for prompt in forecast_prompts)


def test_log_scaled_question(config, make_factory, responder, posts):
    (question,) = parse_posts([log_scaled_post(posts)])
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    submission = build_submission(question, forecast(bot, question), bot)
    cdf = submission.payload["continuous_cdf"]
    assert cdf[0] == 0.0 and cdf[-1] <= 0.999 and len(cdf) == 201


def test_the_bot_cannot_be_switched_to_publishing(config, make_factory, responder):
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    assert bot.publish_reports_to_metaculus is False
    with pytest.raises(AttributeError):
        bot.publish_reports_to_metaculus = True
    assert EnsembleBot._llm_config_defaults() == {"default": None, "summarizer": None, "researcher": None, "parser": None}


# ------------------------------------------------------------------ calibration guards

@pytest.mark.parametrize("answer, expected", [("Probability: 100%", 0.97), ("Probability: 0%", 0.03)])
def test_unanimous_certainty_is_clipped(config, make_factory, by_type, answer, expected):
    responder = CountingResponder()
    responder.text_for = {"": answer}
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    submission = build_submission(by_type["binary"], forecast(bot, by_type["binary"]), bot)
    assert submission.payload["probability_yes"] == pytest.approx(expected)


def test_one_wild_run_does_not_move_a_binary_forecast(config, make_factory, by_type):
    responder = CountingResponder()
    responder.text_for = {"gemini": "Probability: 99%", "gpt": "Probability: 20%", "claude": "Probability: 24%"}
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    submission = build_submission(by_type["binary"], forecast(bot, by_type["binary"]), bot)
    assert 0.20 <= submission.payload["probability_yes"] <= 0.24


def test_zero_probability_options_are_floored(config, make_factory, by_type):
    responder = CountingResponder()
    responder.text_for = {"": "Option A: 100%\nOption B: 0%\nOption C: 0%"}
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    question = by_type["multiple_choice"]
    values = build_submission(question, forecast(bot, question), bot).payload["probability_yes_per_category"]
    assert values["Option B"] == pytest.approx(0.02) and values["Option C"] == pytest.approx(0.02)
    assert values["Option A"] == pytest.approx(0.96)


def test_percentiles_tied_at_an_open_bound_do_not_hang(config, make_factory, by_type):
    """The input on which the unpatched library loops forever (see bot/guard.py)."""
    question = by_type["numeric"]
    low = question.lower_bound
    responder = CountingResponder()
    responder.text_for = {"": "\n".join(f"Percentile {p}: {low}" for p in (10, 20, 40, 60, 80, 90))}
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    started = time.monotonic()
    submission = build_submission(question, forecast(bot, question), bot)
    assert time.monotonic() - started < 20
    assert len(submission.payload["continuous_cdf"]) == 201


# ------------------------------------------------------------------ failure: LLM timeout / provider down

def test_llm_timeout_on_one_lab_fails_over_to_another(config, make_factory, by_type):
    responder = CountingResponder()
    responder.raise_for = {"claude-opus-5.5": TimeoutError("litellm.Timeout: request timed out")}
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    report = forecast(bot, by_type["binary"])
    runs = runs_of(bot, by_type["binary"])
    assert len(runs) == 5  # no run was lost
    assert not any("anthropic" in run["model"] for run in runs)
    assert responder.count("claude-opus-5.5") == 2  # each Claude slot tried once (allowed_tries=1), then moved on
    assert 0.03 <= report.prediction <= 0.97


def test_two_labs_down_still_forecasts_with_the_third(config, make_factory, by_type):
    responder = CountingResponder()
    responder.raise_for = {"anthropic": TimeoutError("timed out"), "gemini-3.8-flash": ConnectionError("503")}
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    forecast(bot, by_type["binary"])  # research: the gpt search source still answers
    runs = runs_of(bot, by_type["binary"])
    assert 2 <= len(runs) < 5 and all("openai" in run["model"] for run in runs)


def test_too_few_successful_runs_fails_the_question(config, make_factory, by_type):
    cfg = dataclasses.replace(config, ensemble=dataclasses.replace(config.ensemble, model_failover=0))
    responder = CountingResponder()
    # gpt-6.1-sol also covers its search source; the Claude search source still answers
    responder.raise_for = {"gpt-6.1-sol": TimeoutError("timed out"), "claude-opus-5.5": TimeoutError("timed out")}
    factory, _ = make_factory(responder, cfg=cfg)
    bot = make_bot(cfg, factory)
    with pytest.raises(Exception, match="successful predictions"):
        forecast(bot, by_type["binary"])


def test_all_forecasters_down_fails_the_question_without_hanging(config, make_factory, by_type):
    class OnlyResearch(CountingResponder):
        def __call__(self, model, prompt):
            if "professional forecaster" in prompt:
                self.calls.append((model, prompt))
                raise TimeoutError("timed out")
            return super().__call__(model, prompt)

    responder = OnlyResearch()
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    with pytest.raises(BaseException) as caught:
        forecast(bot, by_type["numeric"])
    assert _find(caught.value, TimeoutError) is not None
    assert responder.forecast_calls() == 10  # 5 runs x (primary + one fail-over), no endless retrying


def test_question_level_hang_is_cut_by_the_caller(config, make_factory, by_type):
    async def slow(model, prompt):
        await asyncio.sleep(30)
        return "Probability: 50%"

    factory, _ = make_factory(lambda model, prompt: slow(model, prompt))
    bot = make_bot(config, factory)

    async def go():
        await asyncio.wait_for(bot.forecast_question(by_type["binary"]), timeout=0.3)

    started = time.monotonic()
    with pytest.raises(asyncio.TimeoutError):
        asyncio.run(go())
    assert time.monotonic() - started < 5


def test_synchronous_hang_inside_the_library_is_cut_off(config, make_factory, responder, by_type, monkeypatch):
    def spin(self):
        while True:
            pass

    monkeypatch.setattr(NumericDistribution, "get_cdf", spin)
    monkeypatch.setattr(forecaster, "CDF_TIME_LIMIT_S", 0.2)
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    started = time.monotonic()
    with pytest.raises(BaseException) as caught:
        forecast(bot, by_type["numeric"])
    assert _find(caught.value, TimeoutError) is not None
    assert time.monotonic() - started < 10


# ------------------------------------------------------------------ failure: malformed model output

def test_unreadable_answer_is_rescued_by_the_llm_parser(config, make_factory, by_type):
    def inner(model, prompt):
        if PARSER_PROMPT in prompt:
            return '{"prediction_in_decimal": 0.4}'
        from bot.fake import fake_responder

        return fake_responder(model, prompt)

    responder = CountingResponder(inner)
    responder.text_for = {"gemini": "On balance I would put this at roughly forty percent."}
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    forecast(bot, by_type["binary"])
    runs = runs_of(bot, by_type["binary"])
    gemini = [run for run in runs if "gemini" in run["model"]]
    assert len(runs) == 5 and gemini == [{"model": "google/gemini-3.8-flash", "prediction": "40.0%", "parsed_by": "llm"}]
    assert responder.count("gpt-6-luna") == 2  # two agreeing parser samples, as in the template


def test_answer_nobody_can_parse_drops_that_run_instead_of_guessing(config, make_factory, by_type, fast_sleep):
    def inner(model, prompt):
        if PARSER_PROMPT in prompt:
            return "<<REQUESTED TYPE WAS NOT FOUND IN TEXT>>"
        from bot.fake import fake_responder

        return fake_responder(model, prompt)

    responder = CountingResponder(inner)
    responder.text_for = {"gemini": "I cannot give a number for this."}
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    for qtype in ("binary", "multiple_choice", "numeric", "date"):
        question = by_type[qtype]
        submission = build_submission(question, forecast(bot, question), bot)
        runs = runs_of(bot, question)
        assert len(runs) == 4 and not any("gemini" in run["model"] for run in runs), qtype
        assert submission.payload


@pytest.mark.parametrize(
    "garbage",
    [
        "Probability: 250%",
        "Percentile 10: 9\nPercentile 20: 8\nPercentile 40: 7\nPercentile 60: 6\nPercentile 80: 5\nPercentile 90: 4",
        "Option A: 80%\nOption B: 80%\nOption C: 80%",
        "{}",
        "\x00\x00",
    ],
)
def test_malformed_output_from_every_model_fails_cleanly(config, make_factory, by_type, fast_sleep, garbage):
    def inner(model, prompt):
        if PARSER_PROMPT in prompt:
            return "not json at all"
        from bot.fake import fake_responder

        return fake_responder(model, prompt)

    responder = CountingResponder(inner)
    responder.text_for = {"": garbage}
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    for qtype in ("binary", "multiple_choice", "numeric"):
        with pytest.raises(BaseException):
            forecast(bot, by_type[qtype])


def test_numbers_in_the_wrong_units_are_not_trusted_from_the_regex(config, make_factory, by_type):
    question = by_type["numeric"]  # range -35 .. 0
    wrong_units = "\n".join(f"Percentile {p}: {v}" for p, v in zip((10, 20, 40, 60, 80, 90), (-25e6, -22e6, -20e6, -18e6, -15e6, -12e6)))

    def inner(model, prompt):
        if PARSER_PROMPT in prompt:
            return json.dumps([{"percentile": p / 100, "value": v} for p, v in zip((10, 20, 40, 60, 80, 90), (-25, -22, -20, -18, -15, -12))])
        from bot.fake import fake_responder

        return fake_responder(model, prompt)

    responder = CountingResponder(inner)
    responder.text_for = {"gemini": wrong_units}
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    forecast(bot, question)
    (gemini,) = [run for run in runs_of(bot, question) if "gemini" in run["model"]]
    assert gemini["parsed_by"] == "llm" and gemini["prediction"].startswith("p10=-25,")


# ------------------------------------------------------------------ research

def test_one_research_source_down_is_tolerated(config, make_factory, by_type):
    responder = CountingResponder()
    responder.raise_for = {"claude-sonnet-5.5:online": ConnectionError("search backend down")}
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    report = forecast(bot, by_type["binary"])
    assert "Web search (openai/gpt-6.1-sol:online)" in report.explanation
    assert "claude-sonnet-5.5:online)" not in report.explanation


def test_no_research_means_no_forecast(config, make_factory, by_type):
    responder = CountingResponder()
    responder.raise_for = {":online": ConnectionError("search backend down")}
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    with pytest.raises(BaseException) as caught:
        forecast(bot, by_type["binary"])
    assert _find(caught.value, ResearchFailed) is not None
    assert responder.forecast_calls() == 0  # nothing was spent on forecasting blind


def test_forecasting_without_research_is_an_explicit_opt_in(config, make_factory, by_type):
    cfg = dataclasses.replace(config, ensemble=dataclasses.replace(config.ensemble, allow_forecast_without_research=True))
    responder = CountingResponder()
    responder.raise_for = {":online": ConnectionError("search backend down")}
    factory, _ = make_factory(responder, cfg=cfg)
    bot = make_bot(cfg, factory)
    assert 0.03 <= forecast(bot, by_type["binary"]).prediction <= 0.97


def test_asknews_is_used_within_its_monthly_allowance_only(config, make_factory, responder, by_type, ledger):
    calls = []

    async def asknews(query):
        calls.append(query)
        return "Here are the relevant news articles: (synthetic)"

    factory, _ = make_factory(responder)
    bot = make_bot(config, factory, ledger=ledger, asknews=asknews)
    report = forecast(bot, by_type["binary"])
    assert len(calls) == 1 and "## News (AskNews)" in report.explanation
    (counter,) = [row for row in ("asknews",) if True]
    from bot.research import _asknews_counter_name

    assert ledger.counter(_asknews_counter_name()) == config.limits.asknews_units_per_question

    ledger.add_counter(_asknews_counter_name(), config.limits.asknews_monthly_call_units)  # allowance used up
    forecast(bot, by_type["multiple_choice"])
    assert len(calls) == 1  # not called again

    lean = make_bot(config, factory, profile="lean", ledger=ledger, asknews=asknews)
    forecast(lean, by_type["date"])
    assert len(calls) == 1  # the lean (MiniBench) profile never uses it


def test_asknews_failure_does_not_block_the_forecast(config, make_factory, responder, by_type, ledger):
    async def asknews(query):
        raise ConnectionError("asknews 429")

    factory, _ = make_factory(responder)
    bot = make_bot(config, factory, ledger=ledger, asknews=asknews)
    assert "AskNews" not in forecast(bot, by_type["binary"]).explanation


# ------------------------------------------------------------------ spend guard inside a question

def test_call_cap_stops_a_question_without_retry_storms(config, make_factory, responder, by_type):
    cfg = dataclasses.replace(config, limits=dataclasses.replace(config.limits, max_llm_calls_per_question=3))
    factory, meter = make_factory(responder, cfg=cfg)
    bot = make_bot(cfg, factory)

    async def go():
        with meter.question("q"):
            return await bot.forecast_question(by_type["binary"])

    with pytest.raises(BaseException) as caught:
        asyncio.run(go())
    assert _find(caught.value, BudgetExceeded) is not None or "BudgetExceeded" in str(caught.value)
    assert len(responder.calls) == 3  # two research calls, one forecast, then nothing more was sent
