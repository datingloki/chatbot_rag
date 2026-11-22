"""Очистка проверяется на временной SQLite-базе."""

import json
import sqlite3

from polybot.clients.handoff import SQLiteHandoff
from polybot.retention import RETENTION_SECONDS, clean_sqlite
from polybot.storage import SQLiteSessionStore


def test_retention_preview_and_apply(tmp_path):
    path = tmp_path / "pilot.sqlite3"
    now = RETENTION_SECONDS + 10_000
    cutoff = now - RETENTION_SECONDS
    SQLiteSessionStore(path).close()
    SQLiteHandoff(path).close()

    old_state = {"history": [{"ts": cutoff - 1, "text": "старая реплика"}]}
    active_state = {
        "history": old_state["history"] + [{"ts": cutoff + 1, "text": "новая реплика"}],
        "chain": ["старая реплика"],
        "handoff": {"created_at": cutoff - 1, "messages": ["старый черновик"]},
    }
    with sqlite3.connect(path) as db:
        db.executemany("INSERT INTO sessions VALUES (?, ?, ?)", [
            ("old", json.dumps(old_state), cutoff - 1),
            ("current", json.dumps(active_state), cutoff + 1),
        ])
        db.execute("INSERT INTO turns (trace_id, session_id, ts, message, kind, reply, sources, path, "
                   "latency_ms, llm_calls, cost_rub, problem_started_at) "
                   "VALUES (?, ?, ?, '', 'answer', '', '[]', '', 0, 0, 0, 0)",
                   ("old-trace", "current", cutoff))
        db.execute("INSERT INTO feedback VALUES (?, ?, ?, 'up')", ("old-trace", "current", cutoff + 1))
        db.execute("INSERT INTO solved_events VALUES (?, ?, 0, ?)", ("old-trace", "current", cutoff + 1))
        db.execute("INSERT INTO repeat_events VALUES (?, ?, 1, ?)", ("old-trace", "current", cutoff + 2))
        db.executemany("INSERT INTO handoff_requests VALUES (?, ?, ?, NULL, '[]', 'created', ?, ?)", [
            ("old-draft", "old-offer", "current", cutoff - 1, cutoff + 1),
            ("new-draft", "new-offer", "current", cutoff + 1, cutoff + 1),
        ])

    preview = clean_sqlite(path, now=now)
    assert preview == clean_sqlite(path, now=now)
    assert preview == {
        "repeat_events": 1, "solved_events": 1, "feedback": 1, "turns": 1,
        "handoff_requests": 1, "sessions": 1, "sessions_trimmed": 1,
    }
    assert clean_sqlite(path, apply=True, now=now) == preview
    assert not any(clean_sqlite(path, now=now).values())

    with sqlite3.connect(path) as db:
        current, = db.execute("SELECT state FROM sessions").fetchone()
        assert db.execute("SELECT ticket_id FROM handoff_requests").fetchall() == [("new-draft",)]
    state = json.loads(current)
    assert state["history"] == [{"ts": cutoff + 1, "text": "новая реплика"}]
    assert state["chain"] == [] and state["handoff"] is None
