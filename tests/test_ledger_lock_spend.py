"""The local record (nothing is submitted twice), the run lock, and the spend guard."""

from __future__ import annotations

import dataclasses
import subprocess
import sys
import textwrap

import pytest

from bot.config import Limits, Prices
from bot.ledger import FINAL_STATES, Ledger
from bot.lock import AlreadyRunning, RunLock
from bot.spend import BudgetExceeded, SpendMeter
from tests.conftest import PROJECT_DIR

PAYLOAD = {"probability_yes": 0.4}


# ------------------------------------------------------------------ ledger

def test_lifecycle_failed_submitting_submitted_done(ledger):
    assert ledger.status(1) is None and not ledger.is_final(1)
    ledger.record_seen(1, {"post_id": 10, "tournament": "main", "qtype": "binary", "title": "t"})
    assert ledger.status(1) == "seen" and not ledger.is_final(1)

    assert ledger.record_attempt_failed(1, "LLM timeout", cost_usd=0.2) == 1
    assert ledger.status(1) == "failed" and not ledger.is_final(1)

    ledger.mark_submitting(1, PAYLOAD, "comment text", cost_usd=0.3)
    row = ledger.get(1)
    assert row["status"] == "submitting" and row["attempts"] == 2
    assert row["payload_json"] == '{"probability_yes": 0.4}' and row["comment_text"] == "comment text"
    assert row["cost_usd"] == pytest.approx(0.5) and row["post_id"] == 10
    assert not ledger.is_final(1)  # not confirmed yet

    ledger.mark_submitted(1)
    assert ledger.is_final(1) and ledger.get(1)["submitted_at"]
    assert [r["question_id"] for r in ledger.pending_comments()] == [1]

    ledger.mark_done(1)
    assert ledger.status(1) == "done" and ledger.pending_comments() == []


@pytest.mark.parametrize("final", sorted(FINAL_STATES))
def test_a_final_question_can_never_be_submitted_or_failed_again(ledger, final):
    ledger.mark_submitting(7, PAYLOAD, "c", 0.0)
    {"submitted": ledger.mark_submitted, "done": ledger.mark_done, "server_already": ledger.mark_server_already,
     "gave_up": lambda q: ledger.mark_gave_up(q, "why")}[final](7)
    assert ledger.is_final(7)
    with pytest.raises(RuntimeError):
        ledger.mark_submitting(7, PAYLOAD, "c", 0.0)
    with pytest.raises(RuntimeError):
        ledger.record_attempt_failed(7, "late failure")
    assert ledger.status(7) == final


def test_state_survives_reopening_the_file(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    first = Ledger(path)
    first.mark_submitting(3, PAYLOAD, "c", 0.1)
    first.mark_submitted(3)
    first.add_spend(1.25, day="2026-10-02")
    first.add_counter("asknews:2026-10", 2)
    first.close()

    second = Ledger(path)
    assert second.is_final(3)
    assert second.spend_on("2026-10-02") == pytest.approx(1.25)
    assert second.counter("asknews:2026-10") == 2
    with pytest.raises(RuntimeError):
        second.mark_submitting(3, PAYLOAD, "c", 0.0)
    second.close()


def test_comment_retry_bookkeeping(ledger):
    ledger.mark_submitting(5, PAYLOAD, "the comment", 0.0)
    ledger.mark_submitted(5)
    for expected in (1, 2, 3):
        assert ledger.mark_comment_failed(5, "HTTP 500") == expected
    assert ledger.status(5) == "submitted"  # the forecast is still on record
    assert len(ledger.pending_comments(max_attempts=6)) == 1
    assert ledger.pending_comments(max_attempts=3) == []  # stops retrying eventually


def test_dry_runs_are_recorded_separately_and_do_not_block_live(ledger):
    ledger.record_dry_run(9, {"tournament": "testing", "qtype": "binary", "title": "t"}, PAYLOAD, 0.0)
    assert ledger.status(9) is None and not ledger.is_final(9)
    ledger.mark_submitting(9, PAYLOAD, "c", 0.0)


def test_spend_counters_and_run_log(ledger):
    ledger.add_spend(0.5, day="2026-10-01")
    ledger.add_spend(0.25, day="2026-10-02")
    ledger.add_spend(0.25, day="2026-10-02")
    assert ledger.spend_on("2026-10-02") == pytest.approx(0.5)
    assert ledger.spend_total() == pytest.approx(1.0)
    assert ledger.add_counter("x", 2) == 2 and ledger.add_counter("x", 3) == 5
    run_id = ledger.start_run("dry-run")
    ledger.finish_run(run_id, {"ok": 1})
    assert ledger.counts_by_status() == {}


# ------------------------------------------------------------------ lock

def test_lock_blocks_a_second_holder_and_frees_on_exit(tmp_path):
    path = tmp_path / "run.lock"
    with RunLock(path):
        with pytest.raises(AlreadyRunning):
            RunLock(path).acquire()
    with RunLock(path):  # free again
        pass


def test_lock_is_held_across_processes_and_released_when_the_holder_dies(tmp_path):
    path = tmp_path / "run.lock"
    script = textwrap.dedent(
        f"""
        import sys, time
        sys.path.insert(0, {str(PROJECT_DIR)!r})
        from bot.lock import RunLock
        lock = RunLock({str(path)!r}).acquire()
        print("held", flush=True)
        time.sleep(30)
        """
    )
    holder = subprocess.Popen([sys.executable, "-c", script], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        with pytest.raises(AlreadyRunning):
            RunLock(path).acquire()
    finally:
        holder.kill()
        holder.wait()
    with RunLock(path):  # the OS released it with the dead process
        pass


# ------------------------------------------------------------------ spend guard

PRICES = Prices(per_million={"openai/known": (2.0, 10.0)}, default=(15.0, 75.0), search_call_fee_usd=0.04)


def meter_with(ledger, **limits) -> SpendMeter:
    return SpendMeter(dataclasses.replace(Limits(), **limits), PRICES, ledger)


def test_cost_estimate_uses_price_table_and_search_fee(ledger):
    meter = meter_with(ledger)
    assert meter.estimate("openrouter/openai/known", 1_000_000, 100_000) == pytest.approx(3.0)
    assert meter.estimate("openrouter/openai/known:online", 0, 0) == pytest.approx(0.04)
    # an unknown model is charged at the deliberately high default, never zero
    assert meter.estimate("openrouter/vendor/typo-model", 1_000_000, 0) == pytest.approx(15.0)


def test_larger_of_reported_and_estimated_cost_is_recorded(ledger):
    meter = meter_with(ledger)
    assert meter.after_call("openrouter/openai/known", 1000, 1000, reported_cost=0.5) == pytest.approx(0.5)
    assert meter.after_call("openrouter/openai/known", 1_000_000, 0, reported_cost=0.0) == pytest.approx(2.0)
    assert meter.after_call("openrouter/openai/known", 1_000_000, 0, reported_cost=None) == pytest.approx(2.0)
    assert ledger.spend_total() == pytest.approx(4.5) and meter.run_calls == 3


def test_question_cap_stops_further_calls_for_that_question_only(ledger):
    meter = meter_with(ledger, max_question_usd=1.0, max_run_usd=100.0, max_day_usd=100.0)
    with meter.question("q1"):
        meter.before_call("m")
        meter.after_call("openrouter/openai/known", 1_000_000, 0, None)  # $2
        with pytest.raises(BudgetExceeded, match="question budget"):
            meter.before_call("m")
    with meter.question("q2"):
        meter.before_call("m")  # a new question may proceed


def test_call_count_cap_per_question(ledger):
    meter = meter_with(ledger, max_llm_calls_per_question=3)
    with meter.question("q"):
        for _ in range(3):
            meter.before_call("m")
        with pytest.raises(BudgetExceeded, match="call cap"):
            meter.before_call("m")


@pytest.mark.parametrize("cap, match", [("max_run_usd", "run budget"), ("max_day_usd", "day budget")])
def test_run_and_day_caps(ledger, cap, match):
    meter = meter_with(ledger, **{"max_question_usd": 100.0, "max_run_usd": 100.0, "max_day_usd": 100.0, cap: 1.0})
    meter.before_call("m")
    meter.after_call("openrouter/openai/known", 1_000_000, 0, None)  # $2
    assert match in meter.exhausted_reason()
    with pytest.raises(BudgetExceeded, match=match):
        meter.before_call("m")


def test_day_and_season_caps_hold_across_runs(ledger):
    ledger.add_spend(5.0)  # an earlier run today
    assert "day budget" in meter_with(ledger, max_day_usd=4.0).exhausted_reason()
    ledger.add_spend(500.0, day="2026-09-30")  # earlier in the season
    fresh = meter_with(ledger, max_day_usd=1000.0, max_season_usd=400.0)
    assert "season budget" in fresh.exhausted_reason()
    with pytest.raises(BudgetExceeded):
        fresh.check()
