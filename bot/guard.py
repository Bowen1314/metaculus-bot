"""
Time guards for code that asyncio cannot interrupt.

`asyncio.wait_for` only cancels at an `await`. The library's CDF code is synchronous, and it
has at least one input (several percentiles tied at an open bound) on which it loops forever;
bot/numeric.py::sanitize_percentiles removes that input, and these guards are the backstop for
whatever else exists: a hang must cost one model run, or at worst one timer tick, never the
whole season by holding the run lock indefinitely.
"""

from __future__ import annotations

import logging
import os
import signal
import threading
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

EXIT_WATCHDOG = 4


@contextmanager
def time_limit(seconds: float, what: str) -> Iterator[None]:
    """Raises TimeoutError if the (synchronous, main-thread) block runs longer than `seconds`."""
    usable = hasattr(signal, "setitimer") and threading.current_thread() is threading.main_thread()
    if not usable:
        yield
        return

    def on_alarm(signum, frame):
        raise TimeoutError(f"{what} took longer than {seconds:g}s")

    previous = signal.signal(signal.SIGALRM, on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def start_watchdog(seconds: float) -> threading.Timer:
    """Ends the process if it is still alive after `seconds`. The lock is released by the OS and
    the ledger is consistent at every point, so the next timer tick simply carries on."""

    def fire() -> None:
        logger.critical("watchdog: run exceeded %.0fs; exiting so the next run can start", seconds)
        for handler in logging.getLogger().handlers:
            try:
                handler.flush()
            except Exception:
                pass
        os._exit(EXIT_WATCHDOG)

    timer = threading.Timer(seconds, fire)
    timer.daemon = True
    timer.start()
    return timer
