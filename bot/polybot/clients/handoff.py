"""Инструмент передачи обращения консультанту (C06).

Контракт — как у поиска и LLM:
- `create(request) -> HandoffTicket` — сохранить обращение; повтор с тем же `offer_id`
  возвращает уже созданное (`duplicate=True`), второй записи нет;
- `lookup(session_id, offer_id)` — проверить статус без записи;
- `cancel(ticket_id) -> HandoffTicket` — отменить сохранённое обращение;
- при сбое — `ToolError("handoff", kind)`, никогда не «что-то пошло не так» текстом;
- таймаут задаёт вызывающий (`HandoffSettings.timeout_s`, Bot оборачивает вызов в wait_for).

Интеграции с OmniDesk в MVP нет (design doc, 11), поэтому инструмент — адаптер-заглушка:
черновик обращения сохраняется в SQLite (`handoff_requests`, в той же базе, что сессии).
Когда появится OmniDesk, его адаптер реализует тот же протокол, а бот не меняется.
В черновик идут только сообщения пользователя по проблеме — они уже замаскированы на входе
(C08) — и причина передачи; контакты не передаются.
"""
from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

from polybot.errors import ToolError, ToolErrorKind


@dataclass
class HandoffRequest:
    offer_id: str
    session_id: str
    reason: str | None
    messages: list[str]


@dataclass
class HandoffTicket:
    ticket_id: str
    status: str                      # created | cancelled
    duplicate: bool = False


@dataclass
class HandoffRecord:
    ticket_id: str
    offer_id: str
    session_id: str
    reason: str | None
    messages: list[str]
    status: str
    created_at: float = field(default_factory=time.time)


class HandoffTool(Protocol):
    async def create(self, request: HandoffRequest) -> HandoffTicket: ...

    async def lookup(self, session_id: str, offer_id: str) -> HandoffTicket | None: ...

    async def cancel(self, ticket_id: str) -> HandoffTicket: ...

    def records(self, session_id: str | None = None) -> list[HandoffRecord]: ...


def ticket_id_for(offer_id: str) -> str:
    return "H-" + offer_id[:8].upper()


class MemoryHandoff:
    """В памяти: тесты, харнесс, сценарии."""

    def __init__(self) -> None:
        self._by_offer: dict[str, HandoffRecord] = {}

    async def create(self, request: HandoffRequest) -> HandoffTicket:
        rec = self._by_offer.get(request.offer_id)
        if rec is not None:
            return HandoffTicket(rec.ticket_id, rec.status, duplicate=True)
        rec = HandoffRecord(ticket_id_for(request.offer_id), request.offer_id, request.session_id,
                            request.reason, list(request.messages), "created")
        self._by_offer[request.offer_id] = rec
        return HandoffTicket(rec.ticket_id, rec.status)

    async def lookup(self, session_id: str, offer_id: str) -> HandoffTicket | None:
        rec = self._by_offer.get(offer_id)
        if rec is not None and rec.session_id == session_id:
            return HandoffTicket(rec.ticket_id, rec.status)
        return None

    async def cancel(self, ticket_id: str) -> HandoffTicket:
        for rec in self._by_offer.values():
            if rec.ticket_id == ticket_id:
                rec.status = "cancelled"
                return HandoffTicket(ticket_id, "cancelled")
        raise ToolError("handoff", ToolErrorKind.BAD_REQUEST, f"нет обращения {ticket_id}")

    def records(self, session_id: str | None = None) -> list[HandoffRecord]:
        return [r for r in self._by_offer.values() if session_id is None or r.session_id == session_id]


SCHEMA = """
CREATE TABLE IF NOT EXISTS handoff_requests (
    ticket_id  TEXT PRIMARY KEY,
    offer_id   TEXT NOT NULL UNIQUE,
    session_id TEXT NOT NULL,
    reason     TEXT,
    messages   TEXT NOT NULL,
    status     TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
"""


class SQLiteHandoff:
    """Черновики обращений в SQLite (Telegram-бот). Повтор с тем же offer_id — та же запись."""

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(path, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(SCHEMA)
        self._db.commit()

    def _create(self, request: HandoffRequest) -> HandoffTicket:
        with self._lock, self._db:
            row = self._db.execute("SELECT ticket_id, status FROM handoff_requests WHERE offer_id = ?",
                                   (request.offer_id,)).fetchone()
            if row:
                return HandoffTicket(row[0], row[1], duplicate=True)
            tid, now = ticket_id_for(request.offer_id), time.time()
            self._db.execute("INSERT INTO handoff_requests VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                             (tid, request.offer_id, request.session_id, request.reason,
                              json.dumps(request.messages, ensure_ascii=False), "created", now, now))
            return HandoffTicket(tid, "created")

    def _lookup(self, session_id: str, offer_id: str) -> HandoffTicket | None:
        with self._lock:
            row = self._db.execute(
                "SELECT ticket_id, status FROM handoff_requests WHERE session_id = ? AND offer_id = ?",
                (session_id, offer_id),
            ).fetchone()
        return HandoffTicket(*row) if row else None

    def _cancel(self, ticket_id: str) -> HandoffTicket:
        with self._lock, self._db:
            n = self._db.execute("UPDATE handoff_requests SET status = 'cancelled', updated_at = ? "
                                 "WHERE ticket_id = ?", (time.time(), ticket_id)).rowcount
        if not n:
            raise ToolError("handoff", ToolErrorKind.BAD_REQUEST, f"нет обращения {ticket_id}")
        return HandoffTicket(ticket_id, "cancelled")

    async def create(self, request: HandoffRequest) -> HandoffTicket:
        try:
            return await asyncio.to_thread(self._create, request)
        except sqlite3.Error as exc:
            raise ToolError("handoff", ToolErrorKind.UNAVAILABLE, str(exc)) from exc

    async def lookup(self, session_id: str, offer_id: str) -> HandoffTicket | None:
        try:
            return await asyncio.to_thread(self._lookup, session_id, offer_id)
        except sqlite3.Error as exc:
            raise ToolError("handoff", ToolErrorKind.UNAVAILABLE, str(exc)) from exc

    async def cancel(self, ticket_id: str) -> HandoffTicket:
        try:
            return await asyncio.to_thread(self._cancel, ticket_id)
        except sqlite3.Error as exc:
            raise ToolError("handoff", ToolErrorKind.UNAVAILABLE, str(exc)) from exc

    def records(self, session_id: str | None = None) -> list[HandoffRecord]:
        q = "SELECT ticket_id, offer_id, session_id, reason, messages, status, created_at FROM handoff_requests"
        args: tuple = ()
        if session_id is not None:
            q, args = q + " WHERE session_id = ?", (session_id,)
        with self._lock:
            rows = self._db.execute(q, args).fetchall()
        return [HandoffRecord(r[0], r[1], r[2], r[3], json.loads(r[4]), r[5], r[6]) for r in rows]

    def close(self) -> None:
        with self._lock:
            self._db.close()
