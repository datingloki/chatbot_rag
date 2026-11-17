"""Узлы, которые отвечают готовыми текстами без вызова LLM (design doc, 2.2)."""
from __future__ import annotations

from polybot import texts
from polybot.nodes.base import BaseNode
from polybot.pipeline.context import Context
from polybot.schemas import PendingClarification, ReplyKind


class Reject(BaseNode):
    """Текст ответа уже выставил input_check; узел нужен, чтобы путь был виден в трассе."""

    name = "reject"

    async def run(self, ctx: Context) -> None:
        ctx.note(reason=ctx.reason)


class Refuse(BaseNode):
    name = "refuse"

    async def run(self, ctx: Context) -> None:
        ctx.reply(ReplyKind.REFUSE, texts.REFUSE, reason="out_of_domain")


class ClarifyDomain(BaseNode):
    name = "clarify_domain"

    async def run(self, ctx: Context) -> None:
        ctx.reply(ReplyKind.CLARIFY, texts.CLARIFY_DOMAIN, reason="domain_unclear")
        ctx.set_pending = PendingClarification.DOMAIN


class NoAnswer(BaseNode):
    """Вопрос в домене, но надёжных фрагментов нет. Пустая выдача ≠ вне домена (2.2)."""

    name = "no_answer"

    async def run(self, ctx: Context) -> None:
        ctx.note(clarifications_left=ctx.clarifications_left)
        if ctx.clarifications_left > 0:
            ctx.reply(ReplyKind.CLARIFY, texts.NO_ANSWER_CLARIFY, reason="no_evidence")
            ctx.set_pending = PendingClarification.QUESTION
        else:
            ctx.reply(ReplyKind.HANDOFF, f"{texts.HANDOFF_LIMIT} {texts.HANDOFF_TAIL}",
                      reason="no_evidence_clarify_limit")


class Handoff(BaseNode):
    """C06: предлагаем консультанта; без подтверждённой интеграции не заявляем о заявке."""

    name = "handoff"

    async def run(self, ctx: Context) -> None:
        lead = texts.HANDOFF_LIMIT if ctx.reason == "clarify_limit" else texts.HANDOFF
        ctx.reply(ReplyKind.HANDOFF, f"{lead} {texts.HANDOFF_TAIL}", reason=ctx.reason or "handoff")


TOOL_FAILURES = ("retrieval_", "llm_")


class Fallback(BaseNode):
    """Сбой инструмента, неподтверждённый ответ или исчерпанный бюджет шагов (C09).
    Готовый текст, без повторной генерации."""

    name = "fallback"

    async def run(self, ctx: Context) -> None:
        reason = ctx.reason or "unknown"
        text = texts.FALLBACK_TOOL if reason.startswith(TOOL_FAILURES) else texts.FALLBACK_UNVERIFIED
        ctx.reply(ReplyKind.FALLBACK, text, reason=reason)
        ctx.sources = []
        ctx.set_pending = None
        ctx.note(reason=reason)
