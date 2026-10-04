"""
Credential hygiene. This module must be imported, and `apply()` called, BEFORE
forecasting_tools / litellm are imported anywhere in the process.

Rules it enforces:
  1. Secrets come only from the project's `.env` file (path given explicitly).
  2. Every credential-looking variable in the ambient process environment is
     removed first (e.g. an unrelated LLM_API_KEY / OPENAI_API_KEY exported by
     the desktop app or a shell profile), as are provider base-URL overrides
     that could redirect a key to another host.
  3. Only names on ALLOWED_KEYS are copied from `.env` into os.environ.
  4. litellm's own "load any .env I can find on import" behaviour is disabled.
  5. Secret values are never printed; `describe()` reports names only and
     `SecretRedactingFilter` masks them if they ever reach a log record.

It deliberately imports nothing heavier than python-dotenv.
"""

from __future__ import annotations

import logging
import os
import re
import stat
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

PROJECT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENV_PATH = PROJECT_DIR / ".env"

# The only names the bot will ever read from `.env`.
ALLOWED_KEYS: frozenset[str] = frozenset(
    {
        "METACULUS_TOKEN",  # required
        "OPENROUTER_API_KEY",  # sponsored credits arrive as an OpenRouter key
        "OPENAI_API_KEY",  # optional direct-provider alternatives
        "ANTHROPIC_API_KEY",
        "GEMINI_API_KEY",
        "ASKNEWS_CLIENT_ID",  # optional research provider (free for participants)
        "ASKNEWS_SECRET",
        "ASKNEWS_API_KEY",
        "PERPLEXITY_API_KEY",
        "EXA_API_KEY",
    }
)

# Values shipped in env.example; treated as "not set".
PLACEHOLDERS: frozenset[str] = frozenset(
    {"", "REPLACE_ME", "1234567890", "your-token-here", "your-api-key-here", "changeme"}
)

# Anything in the ambient environment matching this is removed from this process.
_CREDENTIAL_NAME = re.compile(
    r"""
    (API_?KEY|ACCESS_?KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIAL)   # generic secret words
    | (_BASE_URL|_API_BASE|_API_URL|_ENDPOINT)$                      # endpoint overrides
    | ^(LLM|OPENAI|ANTHROPIC|OPENROUTER|AZURE|GEMINI|GOOGLE|VERTEX|VERTEXAI|AWS|BEDROCK
        |COHERE|MISTRAL|GROQ|DEEPSEEK|TOGETHER|TOGETHERAI|XAI|REPLICATE|HUGGINGFACE|HF
        |PERPLEXITY|PERPLEXITYAI|EXA|ASKNEWS|METACULUS|LITELLM|FIREWORKS|NEBIUS|CEREBRAS)_
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Set by us; must survive the scrub.
_OWN_SETTINGS = {"LITELLM_MODE": "PRODUCTION", "LITELLM_LOCAL_MODEL_COST_MAP": "True"}

_loaded_secret_values: set[str] = set()


@dataclass
class EnvReport:
    env_path: Path
    env_file_found: bool
    scrubbed_names: list[str] = field(default_factory=list)
    loaded_names: list[str] = field(default_factory=list)
    ignored_names: list[str] = field(default_factory=list)  # in .env but not allowed
    placeholder_names: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def has(self, name: str) -> bool:
        return name in self.loaded_names

    def describe(self) -> str:
        """Names only. Never includes a value."""
        return (
            f"env file: {self.env_path} ({'found' if self.env_file_found else 'MISSING'}); "
            f"loaded: {sorted(self.loaded_names) or 'none'}; "
            f"ambient credentials removed: {len(self.scrubbed_names)}; "
            f"ignored .env names: {sorted(self.ignored_names) or 'none'}"
        )


def is_credential_name(name: str) -> bool:
    return bool(_CREDENTIAL_NAME.search(name))


def scrub_ambient(environ: dict[str, str] | None = None) -> list[str]:
    """Remove credential-looking variables. Returns the removed NAMES."""
    env = os.environ if environ is None else environ
    removed = [name for name in list(env) if is_credential_name(name)]
    for name in removed:
        del env[name]
    return sorted(removed)


def apply(env_path: Path | str | None = None) -> EnvReport:
    """
    Scrub the ambient environment, then load allowed keys from `env_path`.
    Safe to call more than once (e.g. again after importing third-party code).
    """
    path = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    report = EnvReport(env_path=path, env_file_found=path.is_file())

    report.scrubbed_names = scrub_ambient()
    for name, value in _OWN_SETTINGS.items():
        os.environ[name] = value

    if not report.env_file_found:
        report.warnings.append(f"{path} not found; no credentials loaded")
        return report

    try:
        mode = stat.S_IMODE(path.stat().st_mode)
        if mode & 0o077:
            report.warnings.append(
                f"{path.name} is readable by group/others (mode {oct(mode)}); run: chmod 600 {path.name}"
            )
    except OSError:
        pass

    # interpolate=False: `${VAR}` in .env must not pull values from the ambient environment.
    values = dotenv_values(path, interpolate=False)
    for name, value in values.items():
        clean = (value or "").strip()
        if name not in ALLOWED_KEYS:
            report.ignored_names.append(name)
            continue
        if clean in PLACEHOLDERS:
            report.placeholder_names.append(name)
            continue
        os.environ[name] = clean
        report.loaded_names.append(name)
        _loaded_secret_values.add(clean)
    if report.ignored_names:
        report.warnings.append(
            f"ignored names in .env that the bot does not use: {sorted(report.ignored_names)}"
        )
    return report


_AUTH_HEADER_VALUE = re.compile(r"(?i)\b(token|bearer|api[-_ ]?key)([\s:=]+)[A-Za-z0-9._\-]{16,}")


def redact(text: str) -> str:
    """Mask any loaded secret value that appears in `text`, and anything shaped like an
    authorization header value (in case a secret reached the process some other way)."""
    for secret in _loaded_secret_values:
        if secret and len(secret) >= 6 and secret in text:
            text = text.replace(secret, "***REDACTED***")
    return _AUTH_HEADER_VALUE.sub(lambda m: f"{m.group(1)}{m.group(2)}***REDACTED***", text)


class SecretRedactingFilter(logging.Filter):
    """Attach to handlers so a secret can never be written to a log, even by library code."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:
            return True
        redacted = redact(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        if record.exc_text:
            record.exc_text = redact(record.exc_text)
        return True
