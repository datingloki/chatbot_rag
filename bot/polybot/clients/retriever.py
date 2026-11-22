"""Поиск по базе знаний.

Контракт инструмента `retrieve`:
- сигнатура: `await search(query, top_k, exclude_doc_ids=()) -> list[Chunk]`;
- таймаут `settings.retrieval.timeout_s` (3 с), 1 ретрай, если помещается в дедлайн хода (polybot/deadline.py);
- пустой список — «ничего не найдено», это не ошибка;
- при сбое бросает `ToolError(tool="retrieve", ...)`.

Реализации: `QdrantRetriever` (clients/qdrant_retriever.py) — гибридный поиск по индексу;
`NullRetriever` — пустой поиск (индекс не собран); `StaticRetriever` — заданные чанки для тестов.
`exclude_doc_ids` — документы, которые нельзя возвращать (харнесс исключает обращение OmniDesk,
из которого сделан кейс dev).
"""
from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

from polybot.errors import ToolError, ToolErrorKind
from polybot.schemas import Chunk


class Retriever(Protocol):
    async def search(self, query: str, top_k: int,
                     exclude_doc_ids: Iterable[str] = ()) -> list[Chunk]: ...


class NullRetriever:
    version = "null"

    async def search(self, query: str, top_k: int,
                     exclude_doc_ids: Iterable[str] = ()) -> list[Chunk]:
        return []


class StaticRetriever:
    """Возвращает заданные чанки (или бросает ошибку). Все запросы сохраняются в `queries`."""

    version = "static"

    def __init__(self, chunks: list[Chunk] | None = None, error: ToolErrorKind | None = None):
        self.chunks = chunks or []
        self.error = error
        self.queries: list[str] = []
        self.excluded: list[list[str]] = []

    async def search(self, query: str, top_k: int,
                     exclude_doc_ids: Iterable[str] = ()) -> list[Chunk]:
        self.queries.append(query)
        exclude = set(exclude_doc_ids)
        self.excluded.append(sorted(exclude))
        if self.error is not None:
            raise ToolError("retrieve", self.error, "задано в тесте")
        kept = [c for c in self.chunks
                if c.meta.get("doc_id", c.chunk_id.split("#")[0]) not in exclude]
        return sorted(kept, key=lambda c: c.score, reverse=True)[:top_k]
