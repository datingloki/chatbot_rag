"""Очистка локальной базы пилота. По умолчанию показывает объём изменений."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from contextlib import closing
from pathlib import Path

RETENTION_SECONDS = 90 * 86400


def clean_sqlite(path: Path, *, apply: bool = False, now: float | None = None) -> dict[str, int]:
    cutoff = (time.time() if now is None else now) - RETENTION_SECONDS
    mode = "rw" if apply else "ro"
    with closing(sqlite3.connect(f"{path.resolve().as_uri()}?mode={mode}", uri=True)) as db, db:
        if apply:
            db.execute("BEGIN IMMEDIATE")
        else:
            db.execute("BEGIN")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if not {"sessions", "turns"}.issubset(tables):
            raise ValueError("В файле нет таблиц sessions и turns бота.")

        expired_turn = "trace_id IN (SELECT trace_id FROM turns WHERE ts <= :cutoff)"
        columns = {row[1] for row in db.execute("PRAGMA table_info(turns)")}
        expired_problems = set()
        if "problem_started_at" in columns:
            expired_problems = set(db.execute(
                "SELECT DISTINCT session_id, problem_started_at FROM turns WHERE ts <= ?", (cutoff,)
            ))
        conditions = {
            "repeat_events": f"repeated_at <= :cutoff OR {expired_turn}",
            "solved_events": f"solved_at <= :cutoff OR {expired_turn}",
            "feedback": f"ts <= :cutoff OR {expired_turn}",
            "turns": "ts <= :cutoff",
            "handoff_requests": "created_at <= :cutoff",
            "sessions": "updated_at <= :cutoff",
        }
        if "solved_events" in tables:
            conditions["repeat_events"] += (
                " OR trace_id IN (SELECT trace_id FROM solved_events WHERE solved_at <= :cutoff)"
            )
        params = {"cutoff": cutoff}
        counts: dict[str, int] = {}
        for table, condition in conditions.items():
            if table not in tables:
                continue
            if apply:
                counts[table] = db.execute(f"DELETE FROM {table} WHERE {condition}", params).rowcount
            else:
                counts[table] = db.execute(f"SELECT COUNT(*) FROM {table} WHERE {condition}", params).fetchone()[0]

        counts["sessions_trimmed"] = 0
        rows = db.execute("SELECT session_id, state FROM sessions WHERE updated_at > ?", (cutoff,)).fetchall()
        for session_id, raw in rows:
            state = json.loads(raw)
            history = state.get("history", [])
            kept = [turn for turn in history if turn.get("ts", 0.0) > cutoff]
            handoff = state.get("handoff")
            expired_offer = handoff is not None and handoff.get("created_at", 0.0) <= cutoff
            problem_started_at = state.get("problem_started_at", 0.0)
            expired_context = (session_id, problem_started_at) in expired_problems or any(
                not turn.get("ts") or problem_started_at < turn["ts"] <= cutoff for turn in history
            )
            has_context = state.get("chain") or state.get("pending") or state.get("clarifications")
            reset_context = expired_context and has_context
            if len(kept) == len(history) and not expired_offer and not reset_context:
                continue
            state["history"] = kept
            if reset_context:
                state.update(chain=[], pending=None, clarifications=0)
            if expired_offer:
                state["handoff"] = None
            counts["sessions_trimmed"] += 1
            if apply:
                db.execute("UPDATE sessions SET state = ? WHERE session_id = ?",
                           (json.dumps(state, ensure_ascii=False), session_id))
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("database", type=Path, help="Путь к существующей SQLite-базе Polybot")
    parser.add_argument("--apply", action="store_true", help="Удалить записи старше 90 дней")
    args = parser.parse_args()
    try:
        counts = clean_sqlite(args.database, apply=args.apply)
    except (sqlite3.Error, ValueError) as exc:
        parser.error(str(exc))
    print(json.dumps({"mode": "apply" if args.apply else "preview", **counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
