"""Общий дедлайн хода (C09, design doc 3.2): от получения сообщения до ответа — не больше
`settings.limits.turn_deadline_s` (25 с; цель P95 — 15 с), включая ожидание в очереди чата, поиск и LLM.

`Bot.respond` открывает дедлайн на весь ход (`with turn(15)`), клиенты и узлы спрашивают
`remaining()`:
- поиск: попытка — min(timeout_s, остаток), повтор — только если полная попытка ещё помещается;
- LLM: попытка — min(timeout_s, остаток), повтор — только если остаётся ≥ `min_retry_s`;
- узлы `retrieve` и `generate` дополнительно обрывают вызов по остатку (страховка для любых
  реализаций клиента) и уходят в fallback.

ContextVar, а не аргумент: дедлайн виден клиентам без изменения их протоколов, а ходы разных
чатов (разные asyncio-задачи) друг другу не мешают.
"""
from __future__ import annotations

import asyncio
import contextvars
import math
import time
from collections.abc import Awaitable, Iterator
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")

_deadline: contextvars.ContextVar[float | None] = contextvars.ContextVar("turn_deadline", default=None)


@contextmanager
def turn(seconds: float | None) -> Iterator[None]:
    """Дедлайн на время блока. Вложенный блок не продлевает внешний дедлайн."""
    outer = _deadline.get()
    new = None if seconds is None else time.monotonic() + seconds
    if outer is not None and (new is None or outer < new):
        new = outer
    token = _deadline.set(new)
    try:
        yield
    finally:
        _deadline.reset(token)


def remaining() -> float:
    """Сколько секунд осталось до дедлайна хода; бесконечность, если дедлайна нет."""
    d = _deadline.get()
    return math.inf if d is None else d - time.monotonic()


def left_for_trace() -> float | None:
    """Остаток до дедлайна для трассы при сбое инструмента: ≈ 0 — ход оборван дедлайном
    (SLI «обрыв по дедлайну», design doc 3.5); None — дедлайна нет (тесты без Bot.respond)."""
    left = remaining()
    return None if math.isinf(left) else round(max(left, 0.0), 2)


async def bounded(aw: Awaitable[T], tool: str) -> T:
    """Обрывает вызов инструмента по остатку дедлайна хода: `ToolError(tool, timeout)`."""
    from polybot.errors import ToolError, ToolErrorKind

    left = remaining()
    if left <= 0:
        if hasattr(aw, "close"):
            aw.close()                                  # корутину так и не запустили
        raise ToolError(tool, ToolErrorKind.TIMEOUT, "дедлайн хода истёк")
    if math.isinf(left):
        return await aw
    try:
        return await asyncio.wait_for(aw, timeout=left)
    except TimeoutError as exc:
        raise ToolError(tool, ToolErrorKind.TIMEOUT, "дедлайн хода истёк") from exc
