"""Узлы-развилки маршрутизации: проверка входа, домен, лимит уточнений."""
from __future__ import annotations

from enum import StrEnum

from polybot import texts
from polybot.nodes.base import BaseNode
from polybot.pipeline.context import Context
from polybot.schemas import ReplyKind


class InputDecision(StrEnum):
    OK = "ok"
    REJECT = "reject"


class InputCheck(BaseNode):
    """C01: текст на русском до 2000 символов. Вложения отсекает Telegram-слой."""

    name = "input_check"
    decisions = InputDecision

    def describe_input(self, ctx: Context):
        return {"chars": len(ctx.message), "pending": ctx.session.pending}

    async def run(self, ctx: Context) -> InputDecision:
        text = ctx.message.strip()
        limit = ctx.settings.limits.max_input_chars
        if not text:
            ctx.reply(ReplyKind.REJECT, texts.REJECT_EMPTY, reason="empty")
            return InputDecision.REJECT
        if len(text) > limit:
            ctx.reply(ReplyKind.REJECT, texts.REJECT_TOO_LONG.format(limit=limit), reason="too_long")
            return InputDecision.REJECT
        ctx.message = text
        return InputDecision.OK


class DomainDecision(StrEnum):
    IN = "in"
    OUT = "out"
    UNCLEAR = "unclear"


_LABELS = {"in_domain": DomainDecision.IN, "out_of_domain": DomainDecision.OUT,
           "unclear": DomainDecision.UNCLEAR}


class Domain(BaseNode):
    """Классификатор домена. Если сообщение — ответ на уточнение бота, классификатор получает
    предыдущие сообщения цепочки и решает заново; дальше граф идёт по новому решению."""

    name = "domain"
    decisions = DomainDecision

    async def run(self, ctx: Context) -> DomainDecision:
        history = (
            [{"role": "user", "text": t} for t in ctx.session.chain] if ctx.continues_chain else None
        )
        result = ctx.services.domain.predict(ctx.message, history=history)
        ctx.domain = result
        ctx.note(**result, with_history=history is not None,
                 classifier_version=getattr(ctx.services.domain, "version", None))
        return _LABELS[result["label"]]


class LimitDecision(StrEnum):
    ALLOWED = "allowed"
    EXHAUSTED = "exhausted"


class ClarifyLimit(BaseNode):
    """C04: не более 5 уточняющих сообщений по одной проблеме."""

    name = "clarify_limit"
    decisions = LimitDecision

    async def run(self, ctx: Context) -> LimitDecision:
        ctx.note(clarifications=ctx.session.clarifications, left=ctx.clarifications_left)
        if ctx.clarifications_left > 0:
            return LimitDecision.ALLOWED
        ctx.reason = "clarify_limit"
        return LimitDecision.EXHAUSTED
