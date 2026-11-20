"""Финальный узел: записывает реплики и обновляет состояние обращения.

Правила обновления состояния:
- уточнение (CLARIFY) → цепочка = сообщения текущего вопроса, `pending` выставлен, счётчик +1;
- передача консультанту (HANDOFF) → проблема закрыта: цепочка, счётчик и `pending` сброшены;
  появляется предложение передачи (`handoff`, ждёт согласия — его обрабатывает Bot, C06);
- ответ или отказ (ANSWER, REFUSE) → цепочка уточнений завершена, счётчик сохраняется до конца
  проблемы («Проблема решена», /new);
- REJECT и FALLBACK → состояние не меняется: пользователь может повторить сообщение.
"""
from __future__ import annotations

from polybot.nodes.base import BaseNode
from polybot.pipeline.context import Context
import uuid

from polybot.schemas import HandoffOffer, ReplyKind, Turn


class Final(BaseNode):
    name = "final"

    async def run(self, ctx: Context) -> None:
        s = ctx.session
        kind = ctx.reply_kind or ReplyKind.FALLBACK
        before = {"clarifications": s.clarifications, "pending": s.pending, "chain": len(s.chain),
                  "handoff": s.handoff and s.handoff.status}

        if kind is ReplyKind.CLARIFY:
            s.chain = ctx.problem_messages
            s.pending = ctx.set_pending
            s.clarifications += 1
        elif kind is ReplyKind.HANDOFF:
            messages = ctx.problem_messages
            s.new_problem()
            s.handoff = HandoffOffer(offer_id=uuid.uuid4().hex, reason=ctx.reason, messages=messages)
        elif kind in (ReplyKind.ANSWER, ReplyKind.REFUSE):
            s.chain = []
            s.pending = None

        s.history.append(Turn(role="user", text=ctx.message))
        s.history.append(Turn(role="assistant", text=ctx.reply_text, kind=kind))
        if kind is ReplyKind.HANDOFF:
            s.close_problem()            # предложение консультанта — тоже часть старой проблемы
        ctx.note(state_before=before,
                 state_after={"clarifications": s.clarifications, "pending": s.pending,
                              "chain": len(s.chain), "handoff": s.handoff and s.handoff.status})
