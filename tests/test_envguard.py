"""Secrets come only from the project's .env; ambient credentials are ignored."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap

from bot import envguard
from bot.logsetup import RedactingFormatter
from tests.conftest import PROJECT_DIR

AMBIENT = {
    "LLM_API_KEY": "ambient-llm-key-000000",
    "OPENAI_API_KEY": "ambient-openai-key-000000",
    "ANTHROPIC_API_KEY": "ambient-anthropic-key-000000",
    "OPENROUTER_API_KEY": "ambient-openrouter-key-000000",
    "METACULUS_TOKEN": "ambient-metaculus-token-000000",
    "OPENAI_BASE_URL": "http://ambient.invalid/v1",
    "ANTHROPIC_BASE_URL": "http://ambient.invalid",
    "METACULUS_API_BASE_URL": "http://ambient.invalid/api",
    "LITELLM_MODE": "DEV",
}


def _write_env(path, text: str):
    path.write_text(textwrap.dedent(text))
    path.chmod(0o600)
    return path


def test_ambient_llm_and_openai_keys_are_ignored(tmp_path, monkeypatch):
    for name, value in AMBIENT.items():
        monkeypatch.setenv(name, value)
    env = _write_env(tmp_path / ".env", "METACULUS_TOKEN=token-from-project-env-123\n")

    report = envguard.apply(env)

    assert os.environ.get("LLM_API_KEY") is None
    assert os.environ.get("OPENAI_API_KEY") is None
    assert os.environ.get("ANTHROPIC_API_KEY") is None
    assert os.environ.get("OPENROUTER_API_KEY") is None
    assert os.environ.get("OPENAI_BASE_URL") is None
    assert os.environ.get("ANTHROPIC_BASE_URL") is None
    assert os.environ.get("METACULUS_API_BASE_URL") is None
    assert os.environ["METACULUS_TOKEN"] == "token-from-project-env-123"  # the file wins over ambient
    assert os.environ["LITELLM_MODE"] == "PRODUCTION"
    assert report.loaded_names == ["METACULUS_TOKEN"]
    assert {"LLM_API_KEY", "OPENAI_API_KEY"} <= set(report.scrubbed_names)


def test_keys_are_used_only_when_in_the_project_env(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", AMBIENT["OPENAI_API_KEY"])
    monkeypatch.setenv("LLM_API_KEY", AMBIENT["LLM_API_KEY"])
    env = _write_env(
        tmp_path / ".env",
        """
        METACULUS_TOKEN=token-from-project-env-123
        OPENAI_API_KEY=openai-key-from-project-env-456
        LLM_API_KEY=llm-key-written-into-env-789
        """,
    )
    report = envguard.apply(env)
    assert os.environ["OPENAI_API_KEY"] == "openai-key-from-project-env-456"
    # LLM_API_KEY is not a name the bot uses: ignored even when it IS in .env
    assert os.environ.get("LLM_API_KEY") is None
    assert "LLM_API_KEY" in report.ignored_names


def test_no_env_file_means_no_credentials(tmp_path, monkeypatch):
    for name, value in AMBIENT.items():
        monkeypatch.setenv(name, value)
    report = envguard.apply(tmp_path / "missing.env")
    assert not report.env_file_found
    assert report.loaded_names == []
    assert not [name for name in os.environ if envguard.is_credential_name(name) and name not in ("LITELLM_MODE", "LITELLM_LOCAL_MODEL_COST_MAP")]


def test_placeholders_and_interpolation_are_not_loaded(tmp_path, monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", AMBIENT["LLM_API_KEY"])
    env = _write_env(
        tmp_path / ".env",
        """
        METACULUS_TOKEN=REPLACE_ME
        OPENROUTER_API_KEY=
        OPENAI_API_KEY=${LLM_API_KEY}
        """,
    )
    report = envguard.apply(env)
    assert os.environ.get("METACULUS_TOKEN") is None
    assert os.environ.get("OPENROUTER_API_KEY") is None
    assert set(report.placeholder_names) == {"METACULUS_TOKEN", "OPENROUTER_API_KEY"}
    # ${...} is not expanded from the ambient environment
    assert os.environ.get("OPENAI_API_KEY") != AMBIENT["LLM_API_KEY"]


def test_loose_file_mode_is_reported(tmp_path):
    env = _write_env(tmp_path / ".env", "METACULUS_TOKEN=token-from-project-env-123\n")
    env.chmod(0o644)
    report = envguard.apply(env)
    assert any("chmod 600" in warning for warning in report.warnings)


def test_describe_and_logs_never_contain_secret_values(tmp_path):
    env = _write_env(tmp_path / ".env", "METACULUS_TOKEN=token-from-project-env-123\n")
    report = envguard.apply(env)
    assert "token-from-project-env-123" not in report.describe()
    assert envguard.redact("Authorization: Token token-from-project-env-123") == "Authorization: Token ***REDACTED***"

    record = logging.LogRecord("x", logging.ERROR, __file__, 1, "failed with %s", ("token-from-project-env-123",), None)
    assert "token-from-project-env-123" not in RedactingFormatter("%(message)s").format(record)
    try:
        raise RuntimeError("header was Token token-from-project-env-123")
    except RuntimeError:
        record = logging.LogRecord("x", logging.ERROR, __file__, 1, "boom", (), sys.exc_info())
    assert "token-from-project-env-123" not in RedactingFormatter("%(message)s").format(record)


def test_fresh_process_importing_the_library_cannot_see_ambient_or_stray_dotenv_keys(tmp_path):
    """
    End to end, in a clean interpreter: ambient LLM_API_KEY / OPENAI_API_KEY are set, the working
    directory holds a decoy .env (litellm loads any .env it finds unless told not to), and the
    project's own .env holds only a Metaculus token. After the guard and the library imports,
    no LLM credential may be visible and the library's defaults must not have picked one up.
    """
    project_env = _write_env(tmp_path / "project.env", "METACULUS_TOKEN=token-from-project-env-123\n")
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    _write_env(workdir / ".env", "OPENAI_API_KEY=decoy-key-from-stray-dotenv\nLLM_API_KEY=decoy-llm-key\n")
    script = textwrap.dedent(
        f"""
        import json, os, sys
        sys.path.insert(0, {str(PROJECT_DIR)!r})
        from bot import envguard
        envguard.apply({str(project_env)!r})
        import litellm
        import forecasting_tools
        after_import = {{k: os.environ.get(k) for k in ("LLM_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_BASE_URL")}}
        envguard.apply({str(project_env)!r})
        from bot.config import load_config
        from bot.llm import LlmFactory
        llm = LlmFactory(load_config(), None).forecaster("openrouter/openai/gpt-6.1-sol")
        print(json.dumps({{
            "after_import": after_import,
            "token_is_project_one": os.environ.get("METACULUS_TOKEN") == "token-from-project-env-123",
            "llm_api_key_kwarg": llm.litellm_kwargs.get("api_key"),
            "llm_base_url_kwarg": llm.litellm_kwargs.get("base_url"),
            "litellm_api_key": getattr(litellm, "api_key", None),
            "litellm_openai_key": getattr(litellm, "openai_key", None),
        }}))
        """
    )
    env = {"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path), **AMBIENT}
    result = subprocess.run([sys.executable, "-c", script], cwd=workdir, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
    data = json.loads(result.stdout.strip().splitlines()[-1])
    assert data["after_import"] == {"LLM_API_KEY": None, "OPENAI_API_KEY": None, "ANTHROPIC_API_KEY": None, "OPENAI_BASE_URL": None}
    assert data["token_is_project_one"] is True
    assert data["llm_api_key_kwarg"] is None and data["llm_base_url_kwarg"] is None
    assert data["litellm_api_key"] is None and data["litellm_openai_key"] is None
    for value in AMBIENT.values():
        assert value not in result.stdout and value not in result.stderr
    assert "decoy-key-from-stray-dotenv" not in result.stdout + result.stderr
