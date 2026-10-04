"""Guards against code that hangs where asyncio cannot interrupt it."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from bot.guard import EXIT_WATCHDOG, time_limit

PROJECT_DIR = Path(__file__).resolve().parent.parent


def test_time_limit_interrupts_a_busy_loop():
    started = time.monotonic()
    with pytest.raises(TimeoutError, match="cdf took longer"):
        with time_limit(0.2, "cdf"):
            while True:
                pass
    assert time.monotonic() - started < 3


def test_time_limit_leaves_fast_code_and_the_old_handler_alone():
    import signal

    before = signal.getsignal(signal.SIGALRM)
    with time_limit(5, "quick"):
        value = sum(range(1000))
    assert value == 499500 and signal.getsignal(signal.SIGALRM) is before
    time.sleep(0.05)  # no stray alarm afterwards


def test_time_limit_is_a_no_op_off_the_main_thread():
    import threading

    result = []

    def work():
        with time_limit(0.01, "thread"):
            time.sleep(0.05)
            result.append("finished")

    thread = threading.Thread(target=work)
    thread.start()
    thread.join()
    assert result == ["finished"]


def test_watchdog_ends_a_process_that_never_returns():
    code = (
        "import sys; sys.path.insert(0, %r)\n"
        "from bot.guard import start_watchdog\n"
        "start_watchdog(0.3)\n"
        "while True:\n"
        "    pass\n" % str(PROJECT_DIR)
    )
    started = time.monotonic()
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, timeout=30)
    assert done.returncode == EXIT_WATCHDOG and time.monotonic() - started < 20
