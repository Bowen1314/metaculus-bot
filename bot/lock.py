"""Single-instance lock so timer runs never overlap. The OS releases it if the process dies."""

from __future__ import annotations

import fcntl
import os
from pathlib import Path


class AlreadyRunning(RuntimeError):
    pass


class RunLock:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._handle = None

    def acquire(self) -> "RunLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        handle = open(self.path, "a+")
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            handle.close()
            raise AlreadyRunning(f"another run holds {self.path}") from error
        handle.seek(0)
        handle.truncate()
        handle.write(str(os.getpid()))
        handle.flush()
        self._handle = handle
        return self

    def release(self) -> None:
        if self._handle is not None:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            finally:
                self._handle.close()
                self._handle = None

    def __enter__(self) -> "RunLock":
        return self.acquire()

    def __exit__(self, *exc_info) -> None:
        self.release()
