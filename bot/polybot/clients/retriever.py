"""Поиск по базе знаний.

Контракт инструмента `retrieve`:
- сигнатура: `await search(query, top_k) -> list[Chunk]`, чанки по убыванию `score`;
- таймаут `settings.retrieval.timeout_s` (3 с), 1 ретрай;
- пустой список — «ничего не найдено», это не ошибка;
- при сбое бросает `ToolError(tool="retrieve", ...)`.

На этапе 0 реального индекса нет: `NullRetriever` всегда возвращает пустой список, поэтому
доменные вопросы идут в `no_answer`. Поиск по Qdrant появится на этапе 1.
"""
from __future__ import annotations

from typing import Protocol

from polybot.errors import ToolError, ToolErrorKind
from polybot.schemas import Chunk


class Retriever(Protocol):
    async def search(self, query: str, top_k: int) -> list[Chunk]: ...


class NullRetriever:
    async def search(self, query: str, top_k: int) -> list[Chunk]:
        return []


class StaticRetriever:
    """Возвращает заданные чанки (или бросает ошибку). Все запросы сохраняются в `queries`."""

    def __init__(self, chunks: list[Chunk] | None = None, error: ToolErrorKind | None = None):
        self.chunks = chunks or []
        self.error = error
        self.queries: list[str] = []

    async def search(self, query: str, top_k: int) -> list[Chunk]:
        self.queries.append(query)
        if self.error is not None:
            raise ToolError("retrieve", self.error, "задано в тесте")
        return sorted(self.chunks, key=lambda c: c.score, reverse=True)[:top_k]
