"""run.py: the command line the timer and the owner use."""

from __future__ import annotations

import logging
import os
import re
import sys
from pathlib import Path

import pytest

from tests.conftest import PROJECT_DIR, StubClient, with_future_close

sys.path.insert(0, str(PROJECT_DIR))
import run as cli  # noqa: E402

TOKEN = "cli-test-metaculus-token-not-real-0000"
KEY = "sk-or-cli-test-key-not-real-0000000000"


@pytest.fixture(autouse=True)
def _restore_logging_and_no_watchdog(monkeypatch):
    """run.py reconfigures the root logger and arms a process watchdog; undo both for the suite."""
    import bot.guard

    monkeypatch.setattr(bot.guard, "start_watchdog", lambda seconds: None)
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    yield
    for handler in list(root.handlers):
        if handler not in handlers:
            handler.close()
        root.removeHandler(handler)
    for handler in handlers:
        root.addHandler(handler)
    root.setLevel(level)


@pytest.fixture()
def project(tmp_path):
    """A scratch project: its own config.toml (state and logs inside tmp) and its own .env."""

    def build(*, armed=False, token=True, key=False):
        text = (PROJECT_DIR / "config.toml").read_text()
        text, n = re.subn(r"(?m)^armed\s*=\s*\w+", f"armed = {'true' if armed else 'false'}", text)
        assert n == 1
        (tmp_path / "config.toml").write_text(text)
        lines = []
        if token:
            lines.append(f"METACULUS_TOKEN={TOKEN}")
        if key:
            lines.append(f"OPENROUTER_API_KEY={KEY}")
        env = tmp_path / ".env"
        env.write_text("\n".join(lines) + "\n")
        env.chmod(0o600)
        return ["--config", str(tmp_path / "config.toml"), "--env-file", str(env)]

    return build


@pytest.fixture()
def stub(monkeypatch, open_questions):
    import forecasting_tools

    questions = with_future_close(open_questions)
    client = StubClient({name: questions for name in ("bot-testing-area", "fall-futureeval-2026", "minibench")})
    monkeypatch.setattr(forecasting_tools, "MetaculusClient", lambda *args, **kwargs: client)
    return client


def test_run_without_live_is_refused(project, stub, capsys):
    assert cli.main([*project(), "run"]) == cli.EXIT_USAGE
    assert "--live" in capsys.readouterr().err and stub.fetches == []


def test_submit_test_without_confirm_is_refused(project, stub, capsys):
    assert cli.main([*project(key=True), "submit-test"]) == cli.EXIT_USAGE
    assert "--confirm" in capsys.readouterr().err and stub.fetches == [] and stub.forecasts == []


def test_live_run_is_refused_while_not_armed(project, stub, capsys):
    assert cli.main([*project(armed=False, key=True), "run", "--live"]) == cli.EXIT_USAGE
    assert "armed = true" in capsys.readouterr().err
    assert stub.fetches == [] and stub.forecasts == []


def test_no_token_means_nothing_happens(project, stub, capsys):
    assert cli.main([*project(armed=True, token=False), "run", "--live"]) == cli.EXIT_BLOCKED
    assert "METACULUS_TOKEN is not set" in capsys.readouterr().err and stub.fetches == []


def test_armed_live_run_without_a_provider_key_posts_nothing(project, stub, tmp_path):
    assert cli.main([*project(armed=True), "run", "--live"]) == cli.EXIT_BLOCKED
    assert stub.forecasts == [] and stub.comments == []
    assert "missing in .env: OPENROUTER_API_KEY" in (tmp_path / "logs" / "bot.log").read_text()


def test_unknown_tournament_name_is_a_usage_error(project, stub, capsys):
    assert cli.main([*project(armed=True, key=True), "run", "--live", "--only", "nope"]) == cli.EXIT_USAGE
    assert "no tournament named" in capsys.readouterr().err


def test_fake_dry_run_prints_forecasts_and_submits_nothing(project, stub, capsys, tmp_path):
    assert cli.main([*project(), "dry-run", "--fake-llm", "--limit", "3"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "NOTE: --fake-llm" in out and out.count("FINAL:") == 3 and '"forecast_not_submitted": 3' in out
    assert stub.fetches == ["bot-testing-area"]  # a dry run never looks at the scored tournaments
    assert stub.forecasts == [] and stub.comments == []
    assert (tmp_path / "state" / "ledger.sqlite3").exists() and (tmp_path / "logs" / "bot.log").exists()


def test_dry_run_with_real_models_needs_the_key_in_env_file(project, stub, capsys):
    assert cli.main([*project(), "dry-run", "--limit", "1"]) == cli.EXIT_BLOCKED
    assert "missing in .env: OPENROUTER_API_KEY" in capsys.readouterr().out


def test_ambient_keys_are_not_used_by_the_cli(project, stub, capsys, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "ambient-llm-key-must-be-ignored")
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key-must-be-ignored")
    monkeypatch.setenv("OPENROUTER_API_KEY", "ambient-openrouter-key-must-be-ignored")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://example.invalid/v1")
    # Real models requested, the only keys are ambient ones: must stop for lack of a key.
    assert cli.main([*project(), "dry-run", "--limit", "1"]) == cli.EXIT_BLOCKED
    captured = capsys.readouterr()
    assert "missing in .env: OPENROUTER_API_KEY" in captured.out
    for name in ("LLM_API_KEY", "OPENAI_API_KEY", "OPENROUTER_API_KEY", "OPENAI_BASE_URL"):
        assert name not in os.environ
    assert "must-be-ignored" not in captured.out + captured.err


def test_second_process_exits_quietly_while_the_first_holds_the_lock(project, stub, tmp_path):
    from bot.lock import RunLock

    args = project()
    (tmp_path / "state").mkdir()
    held = RunLock(tmp_path / "state" / "run.lock").acquire()
    try:
        assert cli.main([*args, "dry-run", "--fake-llm"]) == cli.EXIT_OK
        assert stub.fetches == []
    finally:
        held.release()
    assert cli.main([*args, "dry-run", "--fake-llm", "--limit", "1"]) == cli.EXIT_OK
    assert stub.fetches == ["bot-testing-area"]


def test_status_before_and_after_a_run(project, stub, capsys):
    args = project()
    assert cli.main([*args, "status"]) == cli.EXIT_OK
    assert "no ledger yet" in capsys.readouterr().out
    cli.main([*args, "dry-run", "--fake-llm", "--limit", "1"])
    capsys.readouterr()
    assert cli.main([*args, "status"]) == cli.EXIT_OK
    out = capsys.readouterr().out
    assert "armed: False" in out and "spend total: $0.00" in out


def test_the_token_is_never_printed(project, stub, capsys, tmp_path):
    cli.main([*project(key=True), "dry-run", "--fake-llm", "--limit", "2", "--verbose"])
    captured = capsys.readouterr()
    log = (tmp_path / "logs" / "bot.log").read_text()
    for secret in (TOKEN, KEY):
        assert secret not in captured.out + captured.err + log
    assert "METACULUS_TOKEN" in log  # names are fine, values are not
