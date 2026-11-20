"""Контекст одного хода: что знает граф о текущем сообщении, сессии и инструментах."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from polybot.schemas import (
    Chunk,
    LLMUsage,
    PendingClarification,
    ReplyKind,
    SessionState,
    Source,
    StepRecord,
)
from polybot.settings import Settings

if TYPE_CHECKING:
    from polybot.clients.domain import DomainService
    from polybot.clients.llm import LLMClient
    from polybot.clients.handoff import HandoffTool
    from polybot.clients.retriever import Retriever


@dataclass
class Services:
    """Инструменты, к которым обращаются узлы. В тестах подменяются заглушками."""

    domain: DomainService
    llm: LLMClient
    retriever: Retriever
    handoff: HandoffTool | None = None        # None → в Bot подставляется MemoryHandoff


@dataclass
class Context:
    trace_id: str
    message: str
    session: SessionState
    settings: Settings
    services: Services

    # Харнесс: документы, которые поиск не должен возвращать (см. Retriever)
    exclude_doc_ids: tuple[str, ...] = ()
    # Что замаскировано во входном сообщении (C08): только виды и количество, без значений
    pii_masked: dict[str, int] = field(default_factory=dict)

    # Заполняют узлы по ходу графа
    domain: dict[str, Any] | None = None
    chunks: list[Chunk] = field(default_factory=list)
    evidence: list[Chunk] = field(default_factory=list)
    llm_raw: str | None = None
    llm_usage: LLMUsage = field(default_factory=LLMUsage)
    prompt_version: str | None = None

    # Итог хода
    reply_kind: ReplyKind | None = None
    reply_text: str = ""
    sources: list[Source] = field(default_factory=list)
    reason: str | None = None
    set_pending: PendingClarification | None = None

    # Трасса
    steps: list[StepRecord] = field(default_factory=list)
    path: list[str] = field(default_factory=list)
    current_step: StepRecord | None = None

    @property
    def continues_chain(self) -> bool:
        """Сообщение — ответ на уточнение бота."""
        return self.session.pending is not None

    @property
    def problem_messages(self) -> list[str]:
        """Сообщения пользователя по текущему вопросу: при ответе на уточнение — исходный
        вопрос + все уточнения + текущее сообщение; иначе только текущее сообщение."""
        if self.continues_chain:
            return [*self.session.chain, self.message]
        return [self.message]

    @property
    def clarifications_left(self) -> int:
        return self.settings.limits.max_clarifications - self.session.clarifications

    def note(self, **fields: Any) -> None:
        """Записать поля в выход текущего шага (они попадут в трассу)."""
        if self.current_step is not None:
            self.current_step.output.update(fields)

    def reply(self, kind: ReplyKind, text: str, reason: str | None = None) -> None:
        self.reply_kind = kind
        self.reply_text = text
        if reason is not None:
            self.reason = reason

    def fail(self, reason: str) -> None:
        """Причина, по которой ход уходит в fallback."""
        self.reason = reason
