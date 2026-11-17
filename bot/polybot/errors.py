"""Типизированные ошибки инструментов (контракт инструмента, КТ3).

Инструмент не возвращает текст «что-то пошло не так»: он бросает `ToolError` с видом ошибки,
а узел превращает её в решение графа (обычно переход в `fallback`).
"""
from __future__ import annotations

from enum import StrEnum


class ToolErrorKind(StrEnum):
    TIMEOUT = "timeout"
    UNAVAILABLE = "unavailable"        # сервис не отвечает / соединение отклонено
    BAD_RESPONSE = "bad_response"      # ответ пришёл, но не по контракту


class ToolError(Exception):
    def __init__(self, tool: str, kind: ToolErrorKind, detail: str = ""):
        self.tool = tool
        self.kind = kind
        self.detail = detail
        super().__init__(f"{tool}: {kind}{': ' + detail if detail else ''}")
