"""The unattended run: what gets forecast, what gets posted, and what happens when things break."""

from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from datetime import datetime, timedelta, timezone

import pytest
from forecasting_tools import ConditionalQuestion

from bot.ledger import Ledger
from bot.llm import LlmFactory
from bot.runner import NotArmed, RunOptions, Runner
from bot.spend import SpendMeter
from tests.conftest import CountingResponder, StubClient

MAIN, MINI, TESTING = "fall-futureeval-2026", "minibench", "bot-testing-area"
FAKE_KEY = "sk-or-test-not-a-real-key-000000000000"


class LocalModels(LlmFactory):
    """Counts as real models for the runner (so the live paths can run) but answers locally."""

    @property
    def is_fake(self) -> bool:  # type: ignore[override]
        return False


def make_runner(config, ledger, client, responder, *, live=False, tournaments=None, limit=None,
                testing_submit=False, credit=None, fake=False, verbose=False):
    if tournaments is None:
        tournaments = config.tournaments if live and not testing_submit else (config.testing,)
    meter = None if fake else SpendMeter(config.limits, config.prices, ledger)
    factory = (LlmFactory if fake else LocalModels)(config, meter, responder=responder)
    lines: list[str] = []
    runner = Runner(
        config=config,
        ledger=ledger,
        client=client,
        factory=factory,
        meter=meter,
        options=RunOptions(live=live, tournaments=tuple(tournaments), limit=limit, verbose=verbose, testing_submit=testing_submit),
        credit_check=lambda: credit,
        out=lines.append,
    )
    return runner, lines


@pytest.fixture()
def armed(config):
    return dataclasses.replace(config, armed=True)


@pytest.fixture()
def key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", FAKE_KEY)


def statuses(ledger) -> dict[str, int]:
    return ledger.counts_by_status()


# ------------------------------------------------------------------ dry run

def test_dry_run_prints_every_type_and_submits_nothing(config, ledger, live_questions, responder, key):
    client = StubClient({TESTING: live_questions})
    runner, lines = make_runner(config, ledger, client, responder)
    summary = runner.run()

    assert client.forecasts == [] and client.comments == []
    assert summary.mode == "dry-run"
    assert summary.tournaments["testing"] == {"open": 9, "forecast_not_submitted": 9}
    text = "\n".join(lines)
    for qtype in ("binary", "multiple_choice", "numeric", "discrete", "date"):
        assert f"[{qtype}] question" in text
    assert text.count("FINAL:") == 9 and text.count("payload:") == 9
    assert statuses(ledger) == {}  # a dry run leaves no "already forecast" record ...
    assert ledger._db.execute("SELECT COUNT(*) FROM dry_runs").fetchone()[0] == 9  # ... only a log of what it printed


def test_dry_run_with_offline_stand_in_needs_no_key(config, ledger, live_questions, responder):
    client = StubClient({TESTING: live_questions})
    runner, lines = make_runner(config, ledger, client, responder, fake=True)
    summary = runner.run()
    assert summary.mode == "dry-run-fake-llm" and summary.stopped_reason is None
    assert summary.tournaments["testing"]["forecast_not_submitted"] == 9
    assert client.forecasts == [] and client.comments == []


def test_dry_run_is_repeatable_and_ignores_the_ledger(config, ledger, live_questions, responder, key):
    client = StubClient({TESTING: live_questions})
    make_runner(config, ledger, client, responder, limit=2)[0].run()
    first = len(responder.calls)
    summary = make_runner(config, ledger, client, responder, limit=2)[0].run()
    assert len(responder.calls) == 2 * first and summary.tournaments["testing"]["forecast_not_submitted"] == 2


def test_verbose_dry_run_shows_the_comment(config, ledger, live_questions, responder, key):
    client = StubClient({TESTING: live_questions})
    runner, lines = make_runner(config, ledger, client, responder, limit=1, verbose=True)
    runner.run()
    assert "----- comment -----" in lines[0] and "# NOTES" in lines[0]


# ------------------------------------------------------------------ the gates in front of live

def test_live_is_refused_unless_armed(config, ledger, responder):
    with pytest.raises(NotArmed, match="armed = true"):
        make_runner(config, ledger, StubClient(), responder, live=True)


def test_fake_answers_can_never_be_submitted(armed, ledger, responder):
    with pytest.raises(NotArmed, match="fake"):
        make_runner(armed, ledger, StubClient(), responder, live=True, fake=True)
    with pytest.raises(NotArmed, match="fake"):
        make_runner(armed, ledger, StubClient(), responder, live=True, fake=True, testing_submit=True, tournaments=(armed.testing,))


def test_submit_test_only_targets_the_testing_area(config, ledger, live_questions, responder, key):
    with pytest.raises(NotArmed, match="testing tournament"):
        make_runner(config, ledger, StubClient(), responder, live=True, testing_submit=True, tournaments=config.tournaments)

    client = StubClient({TESTING: live_questions, MAIN: live_questions})
    runner, _ = make_runner(config, ledger, client, responder, live=True, testing_submit=True, tournaments=(config.testing,), limit=1)
    summary = runner.run()  # works with armed = false: the testing area is unscored
    assert summary.mode == "submit-test"
    assert client.fetches == [TESTING] and len(client.forecasts) == 1 and len(client.comments) == 1


# ------------------------------------------------------------------ live: exactly once

def test_live_run_posts_each_forecast_and_comment_once(armed, ledger, live_questions, responder, key):
    client = StubClient({MAIN: live_questions, MINI: []})
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()

    assert summary.mode == "live" and summary.stopped_reason is None
    assert summary.tournaments["main"] == {"open": 9, "submitted": 9} and summary.comments_posted == 9
    assert sorted(qid for qid, _ in client.forecasts) == sorted(q.id_of_question for q in live_questions)
    assert sorted(post for post, _ in client.comments) == sorted(q.id_of_post for q in live_questions)
    assert statuses(ledger) == {"done": 9}
    shapes = {next(iter(payload)) for _, payload in client.forecasts}
    assert shapes == {"probability_yes", "probability_yes_per_category", "continuous_cdf"}
    assert summary.llm_calls == len(responder.calls) and summary.spend_usd > 0
    assert ledger.spend_total() == pytest.approx(summary.spend_usd)


def test_second_run_does_nothing(armed, ledger, live_questions, responder, key):
    client = StubClient({MAIN: live_questions, MINI: []})
    make_runner(armed, ledger, client, responder, live=True)[0].run()
    calls, forecasts, comments = len(responder.calls), len(client.forecasts), len(client.comments)

    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert summary.tournaments["main"] == {"open": 9, "already_done": 9}
    assert (len(responder.calls), len(client.forecasts), len(client.comments)) == (calls, forecasts, comments)
    assert summary.llm_calls == 0 and summary.spend_usd == 0


def test_ledger_survives_a_restart(armed, tmp_path, live_questions, responder, key):
    client = StubClient({MAIN: live_questions, MINI: []})
    path = tmp_path / "state" / "ledger.sqlite3"
    first = Ledger(path)
    make_runner(armed, first, client, responder, live=True, limit=3)[0].run()
    first.close()

    second = Ledger(path)  # a new process
    try:
        summary = make_runner(armed, second, client, responder, live=True)[0].run()
        assert summary.tournaments["main"] == {"open": 9, "already_done": 3, "submitted": 6}
        assert len(client.forecasts) == 9 and len({qid for qid, _ in client.forecasts}) == 9
    finally:
        second.close()


def test_lost_ledger_does_not_cause_a_second_forecast(armed, ledger, live_questions, responder, key):
    for question in live_questions[:4]:
        question.already_forecasted = True  # Metaculus says this bot already answered
    client = StubClient({MAIN: live_questions, MINI: []})
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert summary.tournaments["main"] == {"open": 9, "already_on_server": 4, "submitted": 5}
    done = {qid for qid, _ in client.forecasts}
    assert done.isdisjoint({q.id_of_question for q in live_questions[:4]}) and len(done) == 5
    assert statuses(ledger) == {"server_already": 4, "done": 5}


def test_most_urgent_questions_go_first_and_limit_applies(armed, ledger, live_questions, responder, key):
    soon = live_questions[5]
    soon.close_time = datetime.now(timezone.utc) + timedelta(minutes=20)
    client = StubClient({MAIN: live_questions, MINI: []})
    make_runner(armed, ledger, client, responder, live=True, limit=1)[0].run()
    assert [qid for qid, _ in client.forecasts] == [soon.id_of_question]


def test_minibench_runs_on_the_lean_profile(armed, ledger, live_questions, responder, key):
    client = StubClient({MAIN: [], MINI: live_questions[:2]})
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert summary.tournaments["minibench"]["submitted"] == 2
    assert responder.forecast_calls() == 2 * len(armed.profile("lean").forecasters)


# ------------------------------------------------------------------ Metaculus API errors

def test_fetch_error_on_one_tournament_does_not_stop_the_other(armed, ledger, live_questions, responder, key, caplog):
    class Flaky(StubClient):
        def get_all_open_questions_from_tournament(self, tournament_id):
            if tournament_id == MAIN:
                self.fetches.append(tournament_id)
                raise ConnectionError("HTTP 502 from Metaculus")
            return super().get_all_open_questions_from_tournament(tournament_id)

    client = Flaky({MINI: live_questions[:1]})
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert summary.tournaments["main"] == {"fetch_failed": 1}
    assert summary.tournaments["minibench"]["submitted"] == 1


def test_fetch_error_everywhere_ends_quietly(armed, ledger, responder, key):
    client = StubClient()
    client.fail_fetch = TimeoutError("read timed out")
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert summary.tournaments == {"main": {"fetch_failed": 1}, "minibench": {"fetch_failed": 1}}
    assert responder.calls == [] and client.forecasts == []


def test_forecast_post_that_did_not_land_is_resent_from_the_ledger(armed, ledger, live_questions, responder, key):
    question = live_questions[0]
    client = StubClient({MAIN: [question], MINI: []})
    client.fail_forecast_times = 1
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    qid = question.id_of_question
    assert summary.tournaments["main"] == {"open": 1, "submit_unconfirmed": 1}
    assert client.forecasts == [] and client.comments == []
    row = ledger.get(qid)
    assert row["status"] == "submitting" and "ConnectionError" in row["last_error"]
    stored = json.loads(row["payload_json"])
    calls = len(responder.calls)

    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert len(responder.calls) == calls  # the models were not asked again
    assert summary.recovered == 1 and summary.comments_posted == 1
    assert [qid_ for qid_, _ in client.forecasts] == [qid]
    assert list(client.forecasts[0][1].values())[0] == list(stored.values())[0]  # the stored forecast, unchanged
    assert ledger.status(qid) == "done" and len(client.comments) == 1


def test_forecast_post_that_landed_but_errored_is_not_sent_again(armed, ledger, live_questions, responder, key):
    question = live_questions[1]
    client = StubClient({MAIN: [question], MINI: []})
    client.fail_forecast_after_landing_times = 1
    make_runner(armed, ledger, client, responder, live=True)[0].run()
    qid = question.id_of_question
    assert len(client.forecasts) == 1 and ledger.status(qid) == "submitting"

    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert len(client.forecasts) == 1  # confirmed with the server instead of posting twice
    assert summary.recovered == 1 and ledger.status(qid) == "done" and len(client.comments) == 1


def test_unconfirmed_submission_waits_while_the_server_cannot_be_asked(armed, ledger, live_questions, responder, key):
    question = live_questions[0]
    client = StubClient({MAIN: [question], MINI: []})
    client.fail_forecast_times = 1
    make_runner(armed, ledger, client, responder, live=True)[0].run()
    calls = len(responder.calls)

    client.fail_lookup = True
    client.fail_forecast_times = 0
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert summary.tournaments["main"] == {"open": 1, "awaiting_confirmation": 1}
    assert client.forecasts == [] and len(responder.calls) == calls  # neither re-sent blind nor re-forecast
    assert ledger.status(question.id_of_question) == "submitting"


def test_question_listing_alone_can_confirm_a_submission(armed, ledger, live_questions, responder, key):
    question = live_questions[1]
    client = StubClient({MAIN: [question], MINI: []})
    client.fail_forecast_after_landing_times = 1
    make_runner(armed, ledger, client, responder, live=True)[0].run()
    calls = len(responder.calls)

    client.fail_lookup = True  # the per-question lookup is down, the listing still works
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert summary.recovered == 1 and summary.comments_posted == 1
    assert len(client.forecasts) == 1 and len(responder.calls) == calls
    assert ledger.status(question.id_of_question) == "done"


def test_repeated_submit_failures_give_up_instead_of_looping(armed, ledger, live_questions, responder, key):
    cfg = dataclasses.replace(armed, limits=dataclasses.replace(armed.limits, max_attempts_per_question=2))
    question = live_questions[0]
    client = StubClient({MAIN: [question], MINI: []})
    client.fail_forecast_times = 99
    for _ in range(5):
        make_runner(cfg, ledger, client, responder, live=True)[0].run()
    row = ledger.get(question.id_of_question)
    assert row["status"] == "gave_up" and "submission failed" in row["last_error"]
    assert responder.forecast_calls() == 5  # forecast once; only the POST was retried
    assert client.forecasts == []


def test_submission_that_closes_before_confirmation_is_dropped(armed, ledger, live_questions, responder, key):
    from forecasting_tools.data_models.questions import QuestionState

    question = live_questions[0]
    client = StubClient({MAIN: [question], MINI: []})
    client.fail_forecast_times = 1
    make_runner(armed, ledger, client, responder, live=True)[0].run()
    question.state = QuestionState.CLOSED
    make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert ledger.status(question.id_of_question) == "gave_up" and client.forecasts == []


def test_failed_comment_is_retried_without_a_new_forecast(armed, ledger, live_questions, responder, key):
    question = live_questions[2]
    client = StubClient({MAIN: [question], MINI: []})
    client.fail_comment_times = 2
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    qid = question.id_of_question
    assert summary.tournaments["main"]["submitted"] == 1 and summary.comments_posted == 0
    assert ledger.status(qid) == "submitted" and len(client.forecasts) == 1
    calls = len(responder.calls)

    make_runner(armed, ledger, client, responder, live=True)[0].run()  # second failure
    assert ledger.status(qid) == "submitted" and ledger.get(qid)["comment_attempts"] == 2
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert summary.comments_posted == 1 and ledger.status(qid) == "done"
    assert len(client.forecasts) == 1 and len(client.comments) == 1 and len(responder.calls) == calls
    assert client.comments[0][1] == ledger.get(qid)["comment_text"]


# ------------------------------------------------------------------ LLM failures in a live run

def test_one_failing_question_does_not_block_the_others(armed, ledger, live_questions, key):
    victim = live_questions[3]

    class Sabotage(CountingResponder):
        def __call__(self, model, prompt):
            if victim.question_text in prompt and "professional forecaster" in prompt:
                self.calls.append((model, prompt))
                raise TimeoutError("litellm.Timeout")
            return super().__call__(model, prompt)

    siblings = [q for q in live_questions if q.question_text == victim.question_text]
    responder = Sabotage()
    client = StubClient({MAIN: live_questions, MINI: []})
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert summary.tournaments["main"] == {"open": 9, "failed": len(siblings), "submitted": 9 - len(siblings)}
    row = ledger.get(victim.id_of_question)
    assert row["status"] == "failed" and row["attempts"] == 1 and row["last_error"]
    assert victim.id_of_question not in {qid for qid, _ in client.forecasts}


def test_failed_question_is_retried_next_run_then_given_up(armed, ledger, live_questions, key, fast_sleep):
    cfg = dataclasses.replace(armed, limits=dataclasses.replace(armed.limits, max_attempts_per_question=2))
    question = live_questions[1]
    responder = CountingResponder()
    responder.text_for = {"": "no number here"}
    responder.inner = lambda model, prompt: "still nothing" if "data analyst" in prompt else CountingResponder().inner(model, prompt)
    client = StubClient({MAIN: [question], MINI: []})

    for expected in (1, 2):
        make_runner(cfg, ledger, client, responder, live=True)[0].run()
        assert ledger.get(question.id_of_question)["attempts"] == expected
    calls = len(responder.calls)
    summary = make_runner(cfg, ledger, client, responder, live=True)[0].run()
    assert summary.tournaments["main"] == {"open": 1, "gave_up": 1}
    assert len(responder.calls) == calls and ledger.status(question.id_of_question) == "gave_up"
    assert client.forecasts == []

    make_runner(cfg, ledger, client, responder, live=True)[0].run()  # and it stays given up
    assert len(responder.calls) == calls


def test_hung_question_is_timed_out_and_the_run_goes_on(armed, ledger, live_questions, key):
    cfg = dataclasses.replace(armed, limits=dataclasses.replace(armed.limits, question_timeout_s=1, max_concurrent_questions=1))
    stuck, fine = live_questions[1], live_questions[2]
    inner = CountingResponder()

    async def answer(model, prompt):
        if stuck.question_text in prompt:
            await asyncio.sleep(60)
        return inner(model, prompt)

    client = StubClient({MAIN: [stuck, fine], MINI: []})
    stuck.close_time = datetime.now(timezone.utc) + timedelta(hours=1)  # goes first
    summary = make_runner(cfg, ledger, client, lambda model, prompt: answer(model, prompt), live=True)[0].run()
    assert summary.tournaments["main"] == {"open": 2, "failed": 1, "submitted": 1}
    assert "TimeoutError" in ledger.get(stuck.id_of_question)["last_error"]
    assert [qid for qid, _ in client.forecasts] == [fine.id_of_question]


def test_dry_run_failures_are_not_counted_as_attempts(config, ledger, live_questions, key):
    responder = CountingResponder()
    responder.raise_for = {"": TimeoutError("all models down")}
    client = StubClient({TESTING: live_questions[:2]})
    summary = make_runner(config, ledger, client, responder)[0].run()
    assert summary.tournaments["testing"] == {"open": 2, "failed": 2} and statuses(ledger) == {}


# ------------------------------------------------------------------ what is skipped

def test_questions_about_to_close_are_left_alone(armed, ledger, live_questions, responder, key):
    late, fine = live_questions[0], live_questions[1]
    late.close_time = datetime.now(timezone.utc) + timedelta(seconds=armed.limits.min_seconds_before_close - 60)
    client = StubClient({MAIN: [late, fine], MINI: []})
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert summary.tournaments["main"] == {"open": 2, "closing_too_soon": 1, "submitted": 1}
    assert [qid for qid, _ in client.forecasts] == [fine.id_of_question]


def test_conditional_questions_are_skipped_not_crashed_on(armed, ledger, live_questions, responder, key):
    conditional = ConditionalQuestion.model_construct(
        question_text="If A, then B?", id_of_post=777001, id_of_question=777002, close_time=None, already_forecasted=False
    )

    class WithConditional(StubClient):
        def get_all_open_questions_from_tournament(self, tournament_id):
            found = super().get_all_open_questions_from_tournament(tournament_id)
            return [conditional, *found] if tournament_id == MAIN else found

    client = WithConditional({MAIN: live_questions[:1], MINI: []})
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert summary.tournaments["main"] == {"open": 2, "unsupported": 1, "submitted": 1}
    assert ledger.get(777002) is None


def test_run_time_budget_defers_the_rest_to_the_next_timer_tick(armed, ledger, live_questions, responder, key):
    cfg = dataclasses.replace(armed, limits=dataclasses.replace(armed.limits, run_time_budget_s=0, max_concurrent_questions=1))
    client = StubClient({MAIN: live_questions, MINI: []})
    runner, _ = make_runner(cfg, ledger, client, responder, live=True)
    runner._started -= 5
    summary = runner.run()
    assert summary.tournaments["main"] == {"open": 9, "deferred": 9} and client.forecasts == []
    assert statuses(ledger) == {}  # nothing counted against the questions


# ------------------------------------------------------------------ spend guard and preflight

def test_missing_provider_key_stops_before_any_call(armed, ledger, live_questions, responder):
    client = StubClient({MAIN: live_questions, MINI: []})
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert summary.stopped_reason == "missing in .env: OPENROUTER_API_KEY"
    assert responder.calls == [] and client.forecasts == [] and statuses(ledger) == {}


def test_nothing_to_do_needs_no_key_and_no_credit_check(armed, ledger, responder):
    checks = []
    runner, _ = make_runner(armed, ledger, StubClient({MAIN: [], MINI: []}), responder, live=True)
    runner.credit_check = lambda: checks.append(1)
    summary = runner.run()
    assert summary.stopped_reason is None and checks == []


def test_low_openrouter_credit_stops_the_run(armed, ledger, live_questions, responder, key):
    client = StubClient({MAIN: live_questions, MINI: []})
    summary = make_runner(armed, ledger, client, responder, live=True, credit=0.75)[0].run()
    assert summary.stopped_reason == "OpenRouter credit is down to $0.75"
    assert responder.calls == [] and client.forecasts == []


def test_unknown_openrouter_credit_does_not_block(armed, ledger, live_questions, responder, key):
    client = StubClient({MAIN: live_questions[:1], MINI: []})
    summary = make_runner(armed, ledger, client, responder, live=True, credit=None)[0].run()
    assert summary.tournaments["main"]["submitted"] == 1


def test_day_budget_already_spent_stops_before_any_call(armed, ledger, live_questions, responder, key):
    ledger.add_spend(armed.limits.max_day_usd + 0.01)
    client = StubClient({MAIN: live_questions, MINI: []})
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert summary.stopped_reason == "day budget exhausted" and responder.calls == []


def test_run_budget_stops_mid_run_without_burning_attempts(armed, ledger, live_questions, responder, key):
    cfg = dataclasses.replace(armed, limits=dataclasses.replace(armed.limits, max_run_usd=0.35, max_concurrent_questions=1))
    client = StubClient({MAIN: live_questions, MINI: []})
    summary = make_runner(cfg, ledger, client, responder, live=True)[0].run()
    counts = summary.tournaments["main"]
    assert summary.stopped_reason == "run budget exhausted"
    assert 1 <= counts.get("submitted", 0) < 9 and counts["deferred"] == 9 - counts["submitted"]
    assert "failed" not in counts
    unfinished = ledger._db.execute("SELECT attempts FROM questions WHERE status != 'done'").fetchall()
    assert all(row["attempts"] == 0 for row in unfinished) and "failed" not in statuses(ledger)
    assert summary.spend_usd < 0.35 + armed.limits.max_question_usd

    again = make_runner(armed, ledger, client, responder, live=True)[0].run()  # next tick, fresh run budget
    assert again.tournaments["main"]["submitted"] == 9 - counts["submitted"]
    assert len({qid for qid, _ in client.forecasts}) == len(client.forecasts) == 9


def test_per_question_cap_fails_that_question_only(armed, ledger, live_questions, responder, key):
    cfg = dataclasses.replace(armed, limits=dataclasses.replace(armed.limits, max_llm_calls_per_question=3))
    client = StubClient({MAIN: live_questions[:2], MINI: []})
    summary = make_runner(cfg, ledger, client, responder, live=True)[0].run()
    assert summary.stopped_reason is None and summary.tournaments["main"] == {"open": 2, "failed": 2}
    assert len(responder.calls) == 6


def test_minibench_is_skipped_when_the_season_budget_runs_low(armed, ledger, live_questions, responder, key):
    reserve = next(t for t in armed.tournaments if t.name == "minibench").min_season_budget_left_usd
    ledger.add_spend(armed.limits.max_season_usd - reserve + 1, day="2026-09-01")  # an earlier day
    client = StubClient({MAIN: live_questions[:1], MINI: live_questions[1:3]})
    summary = make_runner(armed, ledger, client, responder, live=True)[0].run()
    assert summary.tournaments["main"]["submitted"] == 1
    assert summary.tournaments["minibench"] == {"open": 2, "skipped_budget": 2}


def test_every_run_is_recorded(armed, ledger, live_questions, responder, key):
    client = StubClient({MAIN: live_questions[:1], MINI: []})
    make_runner(armed, ledger, client, responder, live=True)[0].run()
    row = ledger._db.execute("SELECT mode, finished_at, summary_json FROM runs").fetchone()
    assert row["mode"] == "live" and row["finished_at"] and json.loads(row["summary_json"])["tournaments"]["main"]["submitted"] == 1


def test_secrets_never_reach_output_or_logs(armed, ledger, live_questions, responder, tmp_path, caplog):
    from bot import envguard

    token = "metaculus-token-value-that-must-not-appear"
    env_file = tmp_path / ".env"
    env_file.write_text(f"METACULUS_TOKEN={token}\nOPENROUTER_API_KEY={FAKE_KEY}\n")
    env_file.chmod(0o600)
    envguard.apply(env_file)  # the only way secrets get in
    client = StubClient({MAIN: live_questions[:2], MINI: []})
    client.fail_forecast_times = 1

    def leaky(question_id, payload):
        raise ConnectionError(f"401 for Authorization: Token {token} / Bearer {FAKE_KEY}")

    client._forecast = leaky  # type: ignore[method-assign]
    with caplog.at_level(logging.DEBUG):
        runner, lines = make_runner(armed, ledger, client, responder, live=True, verbose=True)
        runner.run()
    everything = "\n".join(lines) + caplog.text + json.dumps([dict(r) for r in ledger._db.execute("SELECT * FROM questions").fetchall()])
    assert token not in everything and FAKE_KEY not in everything
    assert "ConnectionError" in everything


# ------------------------------------------------------------------ fetching more than one page

def test_fetch_reads_past_the_first_hundred_posts(live_questions):
    from bot.runner import fetch_open_questions

    base = live_questions[0]

    def clone(number):
        fresh = base.model_copy(deep=True)
        fresh.id_of_question, fresh.id_of_post = 500000 + number, 600000 + number
        return fresh

    everything = [clone(n) for n in range(230)]

    class Paged:
        offsets: list[int] = []

        def _grab_filtered_questions_with_offset(self, api_filter, offset):
            assert api_filter.allowed_tournaments == [MAIN] and api_filter.allowed_statuses == ["open"]
            self.offsets.append(offset)
            page = everything[offset:offset + 100]
            return page, len(page) > 0

    client = Paged()
    found = fetch_open_questions(client, MAIN)
    assert len(found) == 230 and client.offsets == [0, 100, 200]


def test_fetch_stops_after_a_short_first_page_and_survives_a_bad_later_page(live_questions):
    from bot.runner import fetch_open_questions

    class Short:
        calls = 0

        def _grab_filtered_questions_with_offset(self, api_filter, offset):
            self.calls += 1
            return live_questions, True

    short = Short()
    assert len(fetch_open_questions(short, MAIN)) == 9 and short.calls == 1

    full = [live_questions[0].model_copy(update={"id_of_question": 700000 + n, "id_of_post": 800000 + n}) for n in range(100)]

    class SecondPageDown:
        def _grab_filtered_questions_with_offset(self, api_filter, offset):
            if offset:
                raise ConnectionError("HTTP 502")
            return full, True

    assert len(fetch_open_questions(SecondPageDown(), MAIN)) == 100

    class FirstPageDown:
        def _grab_filtered_questions_with_offset(self, api_filter, offset):
            raise ConnectionError("HTTP 502")

    with pytest.raises(ConnectionError):
        fetch_open_questions(FirstPageDown(), MAIN)
