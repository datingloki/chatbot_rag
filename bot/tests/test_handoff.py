"""Передача консультанту (C06): предложение, согласие, отказ, сбой, повтор, отмена — без сети."""
import asyncio

import pytest

from polybot import texts
from polybot.clients.handoff import HandoffRequest, MemoryHandoff, SQLiteHandoff
from polybot.clients.llm import ScriptedLLM
from polybot.errors import ToolError, ToolErrorKind
from polybot.schemas import ReplyKind
from polybot.storage import SQLiteSessionStore
from tests.conftest import ask, make_bot, reply

CLIENT_CASE = "После доработки в нашей базе не работает расчет"
HANDOFF = reply("handoff", "Нужна проверка доработанной конфигурации.")
ANSWER = reply("answer", "Ответ.", ["faq:poly2-261#0"])


def offer(bot, session="s1"):
    r = ask(bot, CLIENT_CASE, session)
    assert r.kind is ReplyKind.HANDOFF and r.actions == ["handoff_confirm", "handoff_decline"]
    return r


def act(bot, action, session="s1"):
    return asyncio.run(bot.handoff_action(session, action))


def test_offer_asks_consent_and_sends_nothing():
    bot = make_bot(llm=ScriptedLLM(HANDOFF))
    r = offer(bot)
    assert texts.HANDOFF_TAIL in r.text and "сохранён" not in r.text
    assert bot.services.handoff.records() == []
    assert bot.store.get("s1").handoff.status == "offered"


def test_new_message_without_consent_expires_offer():
    bot = make_bot(llm=ScriptedLLM(HANDOFF, ANSWER))
    offer(bot)
    r = ask(bot, "Как учесть замену материала?")
    assert r.kind is ReplyKind.ANSWER and r.steps[0].node == "handoff_offer"
    assert bot.store.get("s1").handoff is None and bot.services.handoff.records() == []


def test_short_yes_is_consent_but_longer_message_is_not():
    bot = make_bot(llm=ScriptedLLM(HANDOFF, HANDOFF, ANSWER))
    offer(bot)
    assert ask(bot, "Да, передайте").reason == "handoff_sent"
    assert len(bot.services.handoff.records("s1")) == 1
    offer(bot, "s2")
    assert ask(bot, "да, но как учесть замену материала?", "s2").kind is ReplyKind.ANSWER
    assert bot.services.handoff.records("s2") == []


def test_decline_sends_nothing_and_dialog_continues():
    bot = make_bot(llm=ScriptedLLM(HANDOFF, ANSWER))
    offer(bot)
    r = ask(bot, "нет, не надо")
    assert r.reason == "handoff_declined" and r.text == texts.HANDOFF_DECLINED
    assert bot.services.handoff.records() == []
    assert ask(bot, "Как учесть замену материала?").kind is ReplyKind.ANSWER


def test_tool_error_then_retry_creates_exactly_one_record():
    class Flaky(MemoryHandoff):
        fail = True

        async def create(self, request):
            if self.fail:
                self.fail = False
                raise ToolError("handoff", ToolErrorKind.UNAVAILABLE, "нет связи")
            return await super().create(request)

    bot = make_bot(llm=ScriptedLLM(HANDOFF))
    bot.services.handoff = Flaky()
    offer(bot)
    r = act(bot, "handoff_confirm")
    assert r.reason == "handoff_unavailable" and r.text == texts.HANDOFF_FAILED
    assert r.actions == ["handoff_confirm", "handoff_decline"] and bot.services.handoff.records() == []
    assert act(bot, "handoff_confirm").reason == "handoff_sent"
    assert act(bot, "handoff_confirm").reason == "handoff_duplicate"
    recs = bot.services.handoff.records()
    assert len(recs) == 1 and recs[0].messages == [CLIENT_CASE]


def test_tool_timeout_is_reported_honestly():
    class Slow(MemoryHandoff):
        async def create(self, request):
            await asyncio.sleep(1)

    bot = make_bot(llm=ScriptedLLM(HANDOFF))
    bot.settings.handoff.timeout_s = 0.05
    bot.services.handoff = Slow()
    offer(bot)
    r = act(bot, "handoff_confirm")
    assert r.reason == "handoff_timeout" and "ничего не отправлено" in r.text


def test_cancel_after_send():
    bot = make_bot(llm=ScriptedLLM(HANDOFF))
    offer(bot)
    ticket = bot.store.get("s1").handoff.ticket_id if act(bot, "handoff_confirm") else None
    r = act(bot, "handoff_cancel")
    assert r.reason == "handoff_cancelled" and ticket in r.text
    assert bot.services.handoff.records()[0].status == "cancelled"
    assert act(bot, "handoff_confirm").reason == "handoff_nothing"


def test_action_without_offer():
    bot = make_bot()
    assert act(bot, "handoff_confirm").reason == "handoff_nothing"
    assert act(bot, "handoff_cancel").text == texts.HANDOFF_NOTHING


def test_sqlite_tool_is_idempotent_and_cancels(tmp_path):
    tool = SQLiteHandoff(tmp_path / "h.sqlite3")
    req = HandoffRequest("abc123def", "tg:1", "client_specific", ["вопрос с [PHONE]"])
    t1 = asyncio.run(tool.create(req))
    t2 = asyncio.run(tool.create(req))
    assert t1.ticket_id == t2.ticket_id == "H-ABC123DE" and t2.duplicate
    assert asyncio.run(tool.cancel(t1.ticket_id)).status == "cancelled"
    (rec,) = tool.records("tg:1")
    assert rec.status == "cancelled" and rec.messages == ["вопрос с [PHONE]"]
    with pytest.raises(ToolError):
        asyncio.run(tool.cancel("H-NOPE"))
    tool.close()


def test_offer_survives_restart(tmp_path):
    bot = make_bot(llm=ScriptedLLM(HANDOFF))
    bot.store = SQLiteSessionStore(tmp_path / "s.sqlite3")
    offer(bot)
    bot.store.close()
    bot.store = SQLiteSessionStore(tmp_path / "s.sqlite3")
    assert bot.store.get("s1").handoff.status == "offered"
    assert act(bot, "handoff_confirm").reason == "handoff_sent"
