"""Типы данных, которыми обмениваются узлы, бот и харнесс."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ReplyKind(StrEnum):
    """Что бот отправил пользователю. Используется в метриках (доля отказов, уточнений, передач)."""

    ANSWER = "answer"          # содержательный ответ с источником
    CLARIFY = "clarify"        # уточняющий вопрос (считается в лимит C04)
    REFUSE = "refuse"          # вне домена или контракта (C05)
    HANDOFF = "handoff"        # предложение консультанта (C06)
    REJECT = "reject"          # вход не принят (C01)
    FALLBACK = "fallback"      # сбой инструмента / неподтверждённый ответ / исчерпан бюджет (C09)


class PendingClarification(StrEnum):
    DOMAIN = "domain"          # спросили, относится ли вопрос к 1С:Полиграфии
    QUESTION = "question"      # вопрос в домене, но неполный


@dataclass
class Chunk:
    """Фрагмент базы знаний, найденный поиском."""

    chunk_id: str
    text: str
    title: str
    url: str
    score: float
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Source:
    title: str
    url: str


@dataclass
class Turn:
    role: str                  # "user" | "assistant"
    text: str
    kind: ReplyKind | None = None
    ts: float = field(default_factory=time.time)


@dataclass
class SessionState:
    """Состояние одного чата. На этапе 0 живёт в памяти, на этапе 2 — в SQLite.

    - `history` — эпизодическая память: все реплики чата;
    - `chain` — сообщения пользователя текущей цепочки уточнений: исходный вопрос + ответы
      на уточнения. По ним заново определяется домен и строится поисковый запрос;
    - `clarifications` — сколько уточнений уже отправлено по текущей проблеме (лимит C04).
    """

    session_id: str
    history: list[Turn] = field(default_factory=list)
    chain: list[str] = field(default_factory=list)
    clarifications: int = 0
    pending: PendingClarification | None = None

    def new_problem(self) -> None:
        """«Проблема решена», /new или передача консультанту: счётчик и цепочка сбрасываются."""
        self.chain = []
        self.clarifications = 0
        self.pending = None


@dataclass
class StepRecord:
    """Запись об одном шаге графа. Эти поля на КТ4 становятся span'ом в Langfuse."""

    node: str
    started_at: float
    duration_ms: float = 0.0
    decision: str | None = None
    input: dict[str, Any] = field(default_factory=dict)
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


@dataclass
class LLMUsage:
    model: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    calls: int = 0
    latency_ms: float = 0.0


@dataclass
class Response:
    """Результат одного хода. Один и тот же для Telegram и харнесса."""

    trace_id: str
    session_id: str
    kind: ReplyKind
    text: str
    sources: list[Source]
    steps: list[StepRecord]
    latency_ms: float
    llm: LLMUsage
    reason: str | None = None          # почему именно этот ответ: out_of_domain, retrieval_error, ...
    path: list[str] = field(default_factory=list)
