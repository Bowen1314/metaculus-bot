"""Logging: stderr + rotating file, with secrets masked in everything that is written."""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

from bot.envguard import redact

_FORMAT = "%(asctime)s %(levelname)s %(name)s: %(message)s"


class RedactingFormatter(logging.Formatter):
    """Masks loaded secret values in the final text, including tracebacks."""

    def format(self, record: logging.LogRecord) -> str:
        return redact(super().format(record))


def setup_logging(log_dir: Path | str | None, level: int = logging.INFO) -> None:
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    root.setLevel(level)
    formatter = RedactingFormatter(_FORMAT)

    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(formatter)
    root.addHandler(stream)

    if log_dir is not None:
        path = Path(log_dir)
        path.mkdir(parents=True, exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            path / "bot.log", maxBytes=5_000_000, backupCount=5, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    # Library chatter that adds nothing at INFO.
    for noisy in ("LiteLLM", "litellm", "httpx", "httpcore", "openai", "asyncio", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    # The library logs a warning on every option-list normalisation; keep real errors only.
    logging.getLogger("forecasting_tools.data_models.multiple_choice_report").setLevel(logging.ERROR)
