"""One vote per model, no fail-over between models, binary extremizing (2026-10-18).

The basis is analysis/2026-10-18-accuracy.md (57 resolved MiniBench questions): every Gemini slot
had fallen back to GPT, so GPT held two of three votes, and binary forecasts were under-confident.
"""

from __future__ import annotations

import asyncio
import dataclasses
import logging

import pytest

from bot import aggregate, numeric
from bot.config import load_config
from bot.runner import build_submission
from tests.conftest import PROJECT_DIR, CountingResponder, StubClient
from tests.test_bot import forecast, make_bot, runs_of
from tests.test_runner import MAIN, MINI, key, make_runner  # noqa: F401  (key is a fixture)

GPT, OPUS, SONNET, GEMINI = (
    "openai/gpt-6.1-sol",
    "anthropic/claude-opus-5.5",
    "anthropic/claude-sonnet-5.5",
    "google/gemini-3.8-flash",
)


class DownForForecasts(CountingResponder):
    """Forecast calls to the models in `down` fail; their search sources keep working."""

    def __init__(self, down=(), error=ConnectionError("503 high demand")) -> None:
        super().__init__()
        self.down = set(down)
        self.error = error

    def __call__(self, model, prompt):
        if "professional forecaster" in prompt and any(key in model for key in self.down):
            self.calls.append((model, prompt))
            raise self.error
        return super().__call__(model, prompt)


def binary_answers(responder, **by_model):
    responder.text_for = {key: f"Probability: {value}%" for key, value in by_model.items()}
    return responder


def payload(bot, question):
    return build_submission(question, forecast(bot, question), bot).payload


def test_the_project_config_turns_the_new_ensemble_on():
    cfg = load_config(PROJECT_DIR / "config.toml")
    assert cfg.ensemble.aggregate == "per_model_mean" and cfg.ensemble.model_failover == 0
    assert cfg.calibration.binary_extremize == 1.2
    assert (cfg.calibration.binary_min, cfg.calibration.binary_max) == (0.03, 0.97)
    assert cfg.profile("lean").search_models == cfg.profile("full").search_models == (
        "openrouter/openai/gpt-6.1-sol:online",
        "openrouter/anthropic/claude-sonnet-5.5:online",
    )
    assert cfg.profile("lean").forecasters == (
        "openrouter/openai/gpt-6.1-sol",
        "openrouter/anthropic/claude-sonnet-5.5",
        "openrouter/google/gemini-3.8-flash",
    )


# ------------------------------------------------------------------ a failed model is skipped

def test_lean_failed_model_is_dropped_not_replaced(config, make_factory, by_type):
    responder = binary_answers(DownForForecasts({"gemini"}), gpt=20, claude=40)
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory, profile="lean")
    question = by_type["binary"]

    value = payload(bot, question)["probability_yes"]

    assert [run["model"] for run in sorted(runs_of(bot, question), key=lambda r: r["slot"])] == [GPT, SONNET]
    assert responder.forecast_calls() == 3  # one call per slot; Gemini's slot was not handed to GPT
    assert responder.count("gemini") == 1
    assert value == pytest.approx(aggregate.extremize(0.30, 1.2), abs=1e-6)
    assert bot.aggregation_notes[question.id_of_question].startswith(f"per-model votes [{GPT}=0.200, {SONNET}=0.400]")


def test_full_failed_model_is_dropped_not_replaced(config, make_factory, by_type):
    responder = binary_answers(DownForForecasts({"gemini"}), gpt=70, opus=20)
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    question = by_type["binary"]

    value = payload(bot, question)["probability_yes"]

    assert sorted(run["model"] for run in runs_of(bot, question)) == sorted([GPT, GPT, OPUS, OPUS])
    assert responder.forecast_calls() == 5
    assert value == pytest.approx(aggregate.extremize(0.45, 1.2), abs=1e-6)  # (0.70 + 0.20) / 2


def test_lean_with_one_model_left_fails_the_question(config, make_factory, by_type):
    responder = DownForForecasts({"gemini", "claude-sonnet-5.5"})
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory, profile="lean")
    with pytest.raises(Exception, match="successful predictions"):
        forecast(bot, by_type["binary"])
    assert responder.forecast_calls() == 3


def test_full_two_runs_of_one_model_are_not_two_models(config, make_factory, by_type):
    # The library's run count passes (2 of 5 runs), but both are GPT: one model of three.
    responder = DownForForecasts({"claude-opus", "gemini"})
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    with pytest.raises(Exception, match=r"at least 1\.2 of 3 models, but only got 1"):
        forecast(bot, by_type["binary"])


def test_lean_one_model_left_is_retried_on_the_next_run(config, ledger, live_questions, key):
    armed = dataclasses.replace(config, armed=True)
    question = next(q for q in live_questions if q.question_type == "binary")
    client = StubClient({MAIN: [], MINI: [question]})

    down = DownForForecasts({"gemini", "claude-sonnet-5.5"})
    summary = make_runner(armed, ledger, client, down, live=True)[0].run()
    assert summary.tournaments["minibench"] == {"open": 1, "failed": 1}
    row = ledger.get(question.id_of_question)
    assert row["status"] == "failed" and row["attempts"] == 1 and "successful predictions" in row["last_error"]
    assert client.forecasts == []

    back = binary_answers(DownForForecasts({"gemini"}), gpt=20, claude=40)  # two of three answer again
    summary = make_runner(armed, ledger, client, back, live=True)[0].run()
    assert summary.tournaments["minibench"] == {"open": 1, "submitted": 1}
    assert client.forecasts == [
        (question.id_of_question, {"probability_yes": pytest.approx(aggregate.extremize(0.30, 1.2), abs=1e-6)})
    ]
    assert "one vote per model" in client.comments[0][1] and "Aggregation: per-model votes" in client.comments[0][1]


# ------------------------------------------------------------------ one vote per model

def test_full_runs_of_a_model_are_combined_first(config, make_factory, by_type, caplog):
    # per run: mean 0.42, median 0.30; per model: (0.70 + 0.20 + 0.30) / 3 = 0.40
    responder = binary_answers(CountingResponder(), gpt=70, opus=20, gemini=30)
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    question = by_type["binary"]

    with caplog.at_level(logging.INFO, logger="bot.forecaster"):
        report = forecast(bot, question)
    submission = build_submission(question, report, bot)

    pushed = aggregate.extremize(0.40, 1.2)
    assert submission.payload["probability_yes"] == pytest.approx(pushed, abs=1e-6)
    note = bot.aggregation_notes[question.id_of_question]
    assert note == (
        f"per-model votes [{GPT}=0.700 (mean of 2 runs: 0.700, 0.700), {OPUS}=0.200 (mean of 2 runs: 0.200, 0.200), "
        f"{GEMINI}=0.300]; mean=0.4000; extremized(logit x 1.2)={pushed:.4f}; final={pushed:.4f}"
    )
    assert f"q{question.id_of_question} aggregate (per_model_mean): {note}" in caplog.text
    assert "one vote per model" in submission.comment and "log-odds x 1.2" in submission.comment
    assert submission.comment.rstrip().endswith("Aggregation: " + note)


def test_votes_follow_the_slot_not_the_order_runs_finish(config, make_factory, by_type):
    """GPT answers last, so the run records are in a different order from the predictions the
    library hands over; the slot number keeps the votes with the right model."""
    inner = binary_answers(CountingResponder(), gpt=70, opus=20, gemini=30)

    async def answer(model, prompt):
        if "professional forecaster" in prompt and "gpt-6.1-sol" in model:
            await asyncio.sleep(0.05)
        return inner(model, prompt)

    factory, _ = make_factory(lambda model, prompt: answer(model, prompt))
    bot = make_bot(config, factory)
    question = by_type["binary"]

    value = payload(bot, question)["probability_yes"]

    finished = [run["model"] for run in runs_of(bot, question)]
    assert finished[-2:] == [GPT, GPT] and finished != [GPT, OPUS, GEMINI, GPT, OPUS]  # the case is exercised
    assert value == pytest.approx(aggregate.extremize(0.40, 1.2), abs=1e-6)
    assert bot.aggregation_notes[question.id_of_question].startswith(f"per-model votes [{GPT}=0.700 (mean of 2 runs")


def test_multiple_choice_is_one_vote_per_model(config, make_factory, by_type):
    question = by_type["multiple_choice"]
    responder = CountingResponder()
    responder.text_for = {
        "gpt": "Option A: 70%\nOption B: 20%\nOption C: 10%",
        "opus": "Option A: 10%\nOption B: 60%\nOption C: 30%",
        "gemini": "Option A: 40%\nOption B: 40%\nOption C: 20%",
    }
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)

    values = payload(bot, question)["probability_yes_per_category"]

    expected = {"Option A": 0.40, "Option B": 0.40, "Option C": 0.20}  # per run it would be A 0.40, B 0.38
    assert values == pytest.approx(expected, abs=1e-9)
    assert bot.aggregation_notes[question.id_of_question].startswith(
        f"per-model votes [{GPT} (2 runs): Option A=0.700, Option B=0.200, Option C=0.100; "
    )


@pytest.mark.parametrize("qtype", ["numeric", "discrete", "date"])
def test_continuous_mixture_gives_every_model_the_same_weight(config, make_factory, responder, by_type, monkeypatch, qtype):
    seen = []
    real = numeric.mixture_cdf

    def spy(cdfs, weights=None):
        seen.append(weights)
        return real(cdfs, weights)

    monkeypatch.setattr(numeric, "mixture_cdf", spy)
    factory, _ = make_factory(responder)
    bot = make_bot(config, factory)
    question = by_type[qtype]

    cdf = payload(bot, question)["continuous_cdf"]

    # slots: gpt, opus, gemini, gpt, opus
    assert seen == [pytest.approx([1 / 6, 1 / 6, 1 / 3, 1 / 6, 1 / 6])]
    assert numeric.validate_cdf(
        cdf, cdf_size=question.cdf_size, open_lower_bound=question.open_lower_bound,
        open_upper_bound=question.open_upper_bound,
    ) == []
    assert bot.aggregation_notes[question.id_of_question] == (
        f"mixture, equal weight per model [{GPT}=0.333 (2 runs), {OPUS}=0.333 (2 runs), {GEMINI}=0.333]"
    )


def test_per_run_setting_keeps_the_original_numeric_mixture(config, make_factory, responder, by_type, monkeypatch):
    cfg = dataclasses.replace(config, ensemble=dataclasses.replace(config.ensemble, aggregate="per_run"))
    seen = []
    real = numeric.mixture_cdf
    monkeypatch.setattr(numeric, "mixture_cdf", lambda cdfs, weights=None: seen.append(weights) or real(cdfs, weights))
    factory, _ = make_factory(responder, cfg=cfg)
    bot = make_bot(cfg, factory)
    payload(bot, by_type["numeric"])
    assert seen == [None]


# ------------------------------------------------------------------ config

def _config_with(tmp_path, old: str, new: str):
    text = (PROJECT_DIR / "config.toml").read_text()
    assert old in text
    path = tmp_path / "config.toml"
    path.write_text(text.replace(old, new))
    return load_config(path)


def test_without_the_new_keys_the_original_behaviour_is_kept(tmp_path):
    """The deployed image may meet a config.toml that has not been edited yet."""
    text = (PROJECT_DIR / "config.toml").read_text()
    for line in ('aggregate = "per_model_mean"', "binary_extremize = 1.2", "model_failover = 0"):
        text = text.replace(line, "")
    path = tmp_path / "old.toml"
    path.write_text(text)
    old = load_config(path)
    assert old.ensemble.aggregate == "per_run" and old.ensemble.model_failover == 1
    assert old.calibration.binary_extremize == 1.0


@pytest.mark.parametrize(
    "old, new",
    [
        ('aggregate = "per_model_mean"', 'aggregate = "median"'),
        ("binary_extremize = 1.2", "binary_extremize = 0.2"),
        ("binary_extremize = 1.2", "binary_extremize = 12"),
        ("model_failover = 0", "model_failover = -1"),
    ],
)
def test_bad_ensemble_settings_are_refused(tmp_path, old, new):
    with pytest.raises(ValueError):
        _config_with(tmp_path, old, new)
