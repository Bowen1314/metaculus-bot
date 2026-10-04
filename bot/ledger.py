"""
Local record of what has been forecast, in SQLite (stdlib only).

Per-question lifecycle (live mode):

    (none) -> failed* -> submitting -> submitted -> done
                 ^           |
                 +-----------+   (submit raised; retried on a later run)

`submitting` is written BEFORE the HTTP POST. If the process dies between the POST and the
`submitted` write, the next run sees `submitting` and asks the server whether a forecast exists
instead of blindly posting again. `submitted` means the forecast is on Metaculus but the
(prize-eligibility) comment is not yet; `done` means both are.

Dry runs are written to a separate table and never block or count as a live forecast.
"""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

# States in which a question must not be forecast again.
FINAL_STATES = frozenset({"submitted", "done", "server_already", "gave_up"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS questions (
    question_id   INTEGER PRIMARY KEY,
    post_id       INTEGER,
    tournament    TEXT,
    qtype         TEXT,
    title         TEXT,
    url           TEXT,
    close_time    TEXT,
    status        TEXT NOT NULL,
    attempts      INTEGER NOT NULL DEFAULT 0,
    comment_attempts INTEGER NOT NULL DEFAULT 0,
    last_error    TEXT,
    payload_json  TEXT,
    comment_text  TEXT,
    cost_usd      REAL NOT NULL DEFAULT 0,
    created_at    TEXT NOT NULL,
    updated_at    TEXT NOT NULL,
    submitted_at  TEXT,
    commented_at  TEXT
);
CREATE TABLE IF NOT EXISTS dry_runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    question_id   INTEGER,
    tournament    TEXT,
    qtype         TEXT,
    title         TEXT,
    payload_json  TEXT,
    cost_usd      REAL,
    created_at    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS spend (
    day           TEXT PRIMARY KEY,
    usd           REAL NOT NULL DEFAULT 0,
    llm_calls     INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS counters (
    name          TEXT PRIMARY KEY,
    value         REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS runs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at    TEXT NOT NULL,
    finished_at   TEXT,
    mode          TEXT,
    summary_json  TEXT
);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: datetime | None = None) -> str:
    return (value or utcnow()).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), timeout=30, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=FULL")
        self._db.executescript(_SCHEMA)

    def close(self) -> None:
        self._db.close()

    @contextmanager
    def _tx(self) -> Iterator[sqlite3.Connection]:
        self._db.execute("BEGIN IMMEDIATE")
        try:
            yield self._db
        except BaseException:
            self._db.execute("ROLLBACK")
            raise
        else:
            self._db.execute("COMMIT")

    # ---------------------------------------------------------------- questions

    def get(self, question_id: int) -> dict[str, Any] | None:
        row = self._db.execute(
            "SELECT * FROM questions WHERE question_id = ?", (question_id,)
        ).fetchone()
        return dict(row) if row else None

    def status(self, question_id: int) -> str | None:
        row = self.get(question_id)
        return row["status"] if row else None

    def is_final(self, question_id: int) -> bool:
        return self.status(question_id) in FINAL_STATES

    def attempts(self, question_id: int) -> int:
        row = self.get(question_id)
        return int(row["attempts"]) if row else 0

    def _upsert(self, question_id: int, **fields: Any) -> None:
        now = _iso()
        with self._tx() as db:
            exists = db.execute(
                "SELECT 1 FROM questions WHERE question_id = ?", (question_id,)
            ).fetchone()
            if not exists:
                db.execute(
                    "INSERT INTO questions (question_id, status, created_at, updated_at) VALUES (?, ?, ?, ?)",
                    (question_id, fields.get("status", "failed"), now, now),
                )
            fields["updated_at"] = now
            assignments = ", ".join(f"{name} = ?" for name in fields)
            db.execute(
                f"UPDATE questions SET {assignments} WHERE question_id = ?",
                (*fields.values(), question_id),
            )

    def record_seen(self, question_id: int, meta: dict[str, Any]) -> None:
        """Store descriptive fields without changing status of an existing row."""
        allowed = {k: meta.get(k) for k in ("post_id", "tournament", "qtype", "title", "url", "close_time")}
        if self.get(question_id) is None:
            self._upsert(question_id, status="seen", **allowed)
        else:
            self._upsert(question_id, **allowed)

    def record_attempt_failed(self, question_id: int, error: str, cost_usd: float = 0.0) -> int:
        row = self.get(question_id) or {}
        if row.get("status") in FINAL_STATES:
            raise RuntimeError(f"refusing to mark question {question_id} failed: already {row['status']}")
        attempts = int(row.get("attempts") or 0) + 1
        self._upsert(
            question_id,
            status="failed",
            attempts=attempts,
            last_error=error[:2000],
            cost_usd=float(row.get("cost_usd") or 0) + cost_usd,
        )
        return attempts

    def mark_submitting(self, question_id: int, payload: dict[str, Any], comment: str, cost_usd: float) -> None:
        """Write-ahead intent. Raises if this question already has a forecast on record."""
        row = self.get(question_id) or {}
        if row.get("status") in FINAL_STATES:
            raise RuntimeError(f"question {question_id} already {row['status']}; refusing to submit twice")
        self._upsert(
            question_id,
            status="submitting",
            attempts=int(row.get("attempts") or 0) + 1,
            payload_json=json.dumps(payload),
            comment_text=comment,
            cost_usd=float(row.get("cost_usd") or 0) + cost_usd,
            last_error=None,
        )

    def note_submit_error(self, question_id: int, error: str) -> None:
        """The POST raised: it may or may not have landed, so the row stays `submitting`."""
        self._upsert(question_id, last_error=error[:2000])

    def bump_attempts(self, question_id: int) -> int:
        attempts = self.attempts(question_id) + 1
        self._upsert(question_id, attempts=attempts)
        return attempts

    def mark_submitted(self, question_id: int) -> None:
        self._upsert(question_id, status="submitted", submitted_at=_iso())

    def mark_done(self, question_id: int) -> None:
        self._upsert(question_id, status="done", commented_at=_iso())

    def mark_comment_failed(self, question_id: int, error: str) -> int:
        row = self.get(question_id) or {}
        attempts = int(row.get("comment_attempts") or 0) + 1
        self._upsert(question_id, comment_attempts=attempts, last_error=error[:2000])
        return attempts

    def mark_server_already(self, question_id: int) -> None:
        """Metaculus reports a forecast we have no local record of (e.g. state dir was lost)."""
        self._upsert(question_id, status="server_already")

    def mark_gave_up(self, question_id: int, reason: str) -> None:
        self._upsert(question_id, status="gave_up", last_error=reason[:2000])

    def pending_comments(self, max_attempts: int = 6) -> list[dict[str, Any]]:
        rows = self._db.execute(
            "SELECT * FROM questions WHERE status = 'submitted' AND comment_text IS NOT NULL "
            "AND comment_attempts < ? ORDER BY submitted_at",
            (max_attempts,),
        ).fetchall()
        return [dict(r) for r in rows]

    def in_state(self, status: str) -> list[dict[str, Any]]:
        rows = self._db.execute("SELECT * FROM questions WHERE status = ?", (status,)).fetchall()
        return [dict(r) for r in rows]

    def counts_by_status(self) -> dict[str, int]:
        rows = self._db.execute("SELECT status, COUNT(*) AS n FROM questions GROUP BY status").fetchall()
        return {r["status"]: r["n"] for r in rows}

    # ---------------------------------------------------------------- dry runs

    def record_dry_run(self, question_id: int | None, meta: dict[str, Any], payload: dict[str, Any], cost_usd: float) -> None:
        with self._tx() as db:
            db.execute(
                "INSERT INTO dry_runs (question_id, tournament, qtype, title, payload_json, cost_usd, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (question_id, meta.get("tournament"), meta.get("qtype"), meta.get("title"), json.dumps(payload), cost_usd, _iso()),
            )

    # ---------------------------------------------------------------- spend

    def add_spend(self, usd: float, llm_calls: int = 1, day: str | None = None) -> None:
        day = day or utcnow().strftime("%Y-%m-%d")
        with self._tx() as db:
            db.execute(
                "INSERT INTO spend (day, usd, llm_calls) VALUES (?, ?, ?) "
                "ON CONFLICT(day) DO UPDATE SET usd = usd + excluded.usd, llm_calls = llm_calls + excluded.llm_calls",
                (day, usd, llm_calls),
            )

    def spend_on(self, day: str | None = None) -> float:
        day = day or utcnow().strftime("%Y-%m-%d")
        row = self._db.execute("SELECT usd FROM spend WHERE day = ?", (day,)).fetchone()
        return float(row["usd"]) if row else 0.0

    def spend_total(self) -> float:
        row = self._db.execute("SELECT COALESCE(SUM(usd), 0) AS usd FROM spend").fetchone()
        return float(row["usd"])

    # ---------------------------------------------------------------- counters

    def add_counter(self, name: str, amount: float) -> float:
        with self._tx() as db:
            db.execute(
                "INSERT INTO counters (name, value) VALUES (?, ?) "
                "ON CONFLICT(name) DO UPDATE SET value = value + excluded.value",
                (name, amount),
            )
        return self.counter(name)

    def counter(self, name: str) -> float:
        row = self._db.execute("SELECT value FROM counters WHERE name = ?", (name,)).fetchone()
        return float(row["value"]) if row else 0.0

    # ---------------------------------------------------------------- runs

    def start_run(self, mode: str) -> int:
        with self._tx() as db:
            cur = db.execute("INSERT INTO runs (started_at, mode) VALUES (?, ?)", (_iso(), mode))
            return int(cur.lastrowid)

    def finish_run(self, run_id: int, summary: dict[str, Any]) -> None:
        with self._tx() as db:
            db.execute(
                "UPDATE runs SET finished_at = ?, summary_json = ? WHERE id = ?",
                (_iso(), json.dumps(summary), run_id),
            )
