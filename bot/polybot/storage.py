"""Хранилище сессий в SQLite: состояние чата переживает перезапуск бота.

Таблицы:
- `sessions` — состояние чата (история, цепочка уточнений, счётчик, ожидание ответа) в JSON;
- `turns` — журнал ходов: что спросили, что ответили, путь по графу, причина, задержка, цена.
  Это минимальная наблюдаемость живого бота до Langfuse и сырьё для разбора провалов;
- `feedback` — оценка 👍/👎 под ответом (M09), ключ — `trace_id` хода; повторное нажатие
  заменяет оценку;
- `solved_events` — подтверждённое пользователем закрытие проблемы (M15).

Тексты пользователей — персональные данные: база лежит в bot/data/ (в .gitignore) и
наружу не уходит.
"""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from dataclasses import asdict

from polybot.schemas import HandoffOffer, PendingClarification, ReplyKind, Response, SessionState, Turn

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    state      TEXT NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS turns (
    trace_id   TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    ts         REAL NOT NULL,
    message    TEXT NOT NULL,
    kind       TEXT NOT NULL,
    reason     TEXT,
    reply      TEXT NOT NULL,
    sources    TEXT NOT NULL,
    path       TEXT NOT NULL,
    latency_ms REAL NOT NULL,
    llm_calls  INTEGER NOT NULL,
    cost_rub   REAL NOT NULL,
    problem_started_at REAL,
    handoff_offer_id TEXT,
    handoff_attempt_result TEXT
);
CREATE INDEX IF NOT EXISTS turns_session ON turns(session_id, ts);
CREATE TABLE IF NOT EXISTS solved_events (
    trace_id           TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL,
    problem_started_at REAL NOT NULL,
    solved_at          REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS repeat_events (
    trace_id           TEXT PRIMARY KEY,
    session_id         TEXT NOT NULL,
    problem_started_at REAL NOT NULL,
    repeated_at        REAL NOT NULL,
    UNIQUE (session_id, problem_started_at)
);
CREATE TABLE IF NOT EXISTS feedback (
    trace_id   TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    ts         REAL NOT NULL,
    rating     TEXT NOT NULL CHECK (rating IN ('up', 'down'))
);
"""


def state_to_json(state: SessionState, keep_history: int) -> str:
    history = state.history[-keep_history:] if keep_history > 0 else []
    return json.dumps({
        "history": [{"role": t.role, "text": t.text, "kind": t.kind and str(t.kind), "ts": t.ts}
                    for t in history],
        "chain": state.chain,
        "clarifications": state.clarifications,
        "pending": state.pending and str(state.pending),
        "handoff": asdict(state.handoff) if state.handoff else None,
        "problem_started_at": state.problem_started_at,
    }, ensure_ascii=False)


def state_from_json(session_id: str, raw: str) -> SessionState:
    d = json.loads(raw)
    return SessionState(
        session_id=session_id,
        history=[Turn(role=t["role"], text=t["text"], kind=ReplyKind(t["kind"]) if t.get("kind") else None,
                      ts=t.get("ts", 0.0)) for t in d.get("history", [])],
        chain=list(d.get("chain", [])),
        clarifications=int(d.get("clarifications", 0)),
        pending=PendingClarification(d["pending"]) if d.get("pending") else None,
        handoff=HandoffOffer(**d["handoff"]) if d.get("handoff") else None,
        problem_started_at=float(d.get("problem_started_at", 0.0)),
    )


class SQLiteSessionStore:
    """Тот же интерфейс, что у SessionStore в памяти: get / save (+ журнал ходов)."""

    def __init__(self, path: Path, keep_history: int = 50):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.keep_history = keep_history
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        columns = {row[1] for row in self._db.execute("PRAGMA table_info(turns)")}
        if "problem_started_at" not in columns:
            self._db.execute("ALTER TABLE turns ADD COLUMN problem_started_at REAL")
        if "handoff_offer_id" not in columns:
            self._db.execute("ALTER TABLE turns ADD COLUMN handoff_offer_id TEXT")
        if "handoff_attempt_result" not in columns:
            self._db.execute("ALTER TABLE turns ADD COLUMN handoff_attempt_result TEXT")
        self._db.commit()

    def get(self, session_id: str) -> SessionState:
        with self._lock:
            row = self._db.execute("SELECT state FROM sessions WHERE session_id = ?",
                                   (session_id,)).fetchone()
        return state_from_json(session_id, row[0]) if row else SessionState(session_id=session_id)

    def save(self, state: SessionState) -> None:
        with self._lock, self._db:
            self._save_state(state)

    def _save_state(self, state: SessionState) -> None:
        self._db.execute(
            "INSERT INTO sessions(session_id, state, updated_at) VALUES (?, ?, ?) "
            "ON CONFLICT(session_id) DO UPDATE SET state = excluded.state, updated_at = excluded.updated_at",
            (state.session_id, state_to_json(state, self.keep_history), time.time()),
        )

    def mark_solved(self, state: SessionState, trace_id: str) -> bool:
        """Записывает закрытие и сбрасывает состояние в одной транзакции."""
        with self._lock, self._db:
            turn = self._db.execute(
                "SELECT 1 FROM turns WHERE trace_id = ? AND session_id = ? "
                "AND kind IN ('answer', 'clarify') AND problem_started_at = ?",
                (trace_id, state.session_id, state.problem_started_at),
            ).fetchone()
            if turn is None:
                return False
            started_at = state.problem_started_at
            state.new_problem()
            self._db.execute(
                "INSERT INTO solved_events VALUES (?, ?, ?, ?)",
                (trace_id, state.session_id, started_at, state.problem_started_at),
            )
            self._save_state(state)
            return True

    def mark_repeated(self, state: SessionState, trace_id: str) -> bool:
        """Связывает текущую проблему с закрытием, которое выбрал пользователь."""
        now = time.time()
        with self._lock, self._db:
            result = self._db.execute(
                "INSERT INTO repeat_events SELECT trace_id, session_id, ?, ? FROM solved_events "
                "WHERE trace_id = ? AND session_id = ? AND solved_at >= ? AND solved_at <= ? "
                "ON CONFLICT DO NOTHING",
                (state.problem_started_at, now, trace_id, state.session_id, now - 7 * 86400,
                 state.problem_started_at),
            )
            return result.rowcount == 1

    def log_turn(self, message: str, r: Response) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT OR REPLACE INTO turns "
                "(trace_id, session_id, ts, message, kind, reason, reply, sources, path, "
                "latency_ms, llm_calls, cost_rub, problem_started_at, handoff_offer_id, handoff_attempt_result) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (r.trace_id, r.session_id, time.time(), message, str(r.kind), r.reason, r.text,
                 json.dumps([s.doc_id or s.url for s in r.sources], ensure_ascii=False),
                 " > ".join(r.path), r.latency_ms, r.llm.calls, r.llm.cost_rub, r.problem_started_at,
                 r.handoff_offer_id, r.handoff_attempt_result))

    def handoff_attempt_summary(self) -> dict[str, int]:
        """M19: подтверждённые сохранения и общее число завершённых попыток."""
        with self._lock:
            saved, attempts = self._db.execute(
                "SELECT COUNT(CASE WHEN handoff_attempt_result = 'saved' THEN 1 END), COUNT(*) "
                "FROM turns WHERE handoff_attempt_result IS NOT NULL"
            ).fetchone()
        return {"saved": saved, "attempts": attempts}

    def log_feedback(self, session_id: str, trace_id: str, rating: str) -> None:
        with self._lock, self._db:
            self._db.execute(
                "INSERT INTO feedback(trace_id, session_id, ts, rating) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(trace_id) DO UPDATE SET rating = excluded.rating, ts = excluded.ts",
                (trace_id, session_id, time.time(), rating))

    def feedback_summary(self) -> dict[str, int]:
        """M09: число 👍 и 👎 и число содержательных ответов (для доли оценённых)."""
        with self._lock:
            rows = dict(self._db.execute("SELECT rating, COUNT(*) FROM feedback GROUP BY rating").fetchall())
            answers = self._db.execute("SELECT COUNT(*) FROM turns WHERE kind = 'answer'").fetchone()[0]
        return {"up": rows.get("up", 0), "down": rows.get("down", 0), "answers": answers}

    def close(self) -> None:
        with self._lock:
            self._db.close()
