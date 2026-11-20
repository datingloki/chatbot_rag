"""Узлы-развилки маршрутизации: проверка входа, домен, лимит уточнений."""
from __future__ import annotations

import re
from enum import StrEnum

from polybot import texts
from polybot.nodes.base import BaseNode
from polybot.pipeline.context import Context
from polybot.schemas import PendingClarification, ReplyKind


class InputDecision(StrEnum):
    OK = "ok"
    REJECT = "reject"


class InputCheck(BaseNode):
    """C01: текст на русском до 2000 символов. Вложения отсекает Telegram-слой."""

    name = "input_check"
    decisions = InputDecision

    def describe_input(self, ctx: Context):
        return {"chars": len(ctx.message), "pending": ctx.session.pending, "pii_masked": ctx.pii_masked}

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


# Правила уровня диалога — их не видит классификатор, который судит по тексту сообщений.
# «Вашу программу» в чате бота 1С:Полиграфии — это и есть продукт.
PRODUCT_REF_RE = re.compile(r"\bваш(?:а|у|ей|ем|ей|его)?\s+(?:программ|продукт|конфигурац|систем|1с)", re.I)
# Ответ на вопрос бота «вопрос касается 1С:Полиграфии?»
AFFIRM_RE = re.compile(r"\b(?:да|ага|угу|конечно|верно|именно|точно|касается|относится)\b", re.I)
DENY_RE = re.compile(r"^\W*(?:нет|не касается|не относится)\W*$", re.I)


def dialogue_rule(ctx: Context, label: str) -> tuple[str, str] | None:
    text = ctx.message
    if ctx.session.pending is PendingClarification.QUESTION and label == "unclear":
        # Мы уже в домене и задали вопрос по существу («какая редакция и ОС?»). Ответ вида
        # «Windows 10, лицензия программная» классификатору неясен, но это не смена темы;
        # уйти из домена можно только явным «вне домена».
        return "in_domain", "answer_to_question"
    if ctx.session.pending is PendingClarification.DOMAIN:
        if DENY_RE.match(text):
            return "out_of_domain", "user_denied_domain"
        if AFFIRM_RE.search(text) and not re.search(r"\bне\s+(?:касается|относится)", text, re.I):
            return "in_domain", "user_confirmed_domain"
    if PRODUCT_REF_RE.search(text):
        return "in_domain", "refers_to_product"
    return None


class Domain(BaseNode):
    """Классификатор домена. Если сообщение — ответ на уточнение бота, классификатор получает
    предыдущие сообщения цепочки и решает заново; дальше граф идёт по новому решению.

    Поверх классификатора — правила диалога: пользователь подтвердил домен в ответ на наш
    вопрос («да, касается»), отвечает на наш вопрос по существу и классификатор не уверен,
    или прямо говорит о «вашей программе» → в домене. Если вопрос
    всё же посторонний, его отсечёт второй рубеж — проверка домена в промпте генерации."""

    name = "domain"
    decisions = DomainDecision

    async def run(self, ctx: Context) -> DomainDecision:
        history = (
            [{"role": "user", "text": t} for t in ctx.session.chain] if ctx.continues_chain else None
        )
        result = dict(ctx.services.domain.predict(ctx.message, history=history))
        override = dialogue_rule(ctx, result["label"])
        if override is not None and override[0] != result["label"]:
            result.update(classifier_label=result["label"], label=override[0], dialogue_rule=override[1])
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
