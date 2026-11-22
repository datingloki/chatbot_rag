"""Передача консультанту (C06): предложение, согласие, отказ, сбой, повтор, отмена — без сети."""
import asyncio

import pytest

from polybot import texts
from polybot.clients.handoff import HandoffRequest, MemoryHandoff, SQLiteHandoff
from polybot.clients.llm import ScriptedLLM
from polybot.errors import ToolError, ToolErrorKind
from polybot.schemas import ReplyKind
from polybot.storage import SQLiteSessionStore
from polybot.telegram_bot import keyboard_rows
from tests.conftest import ask, make_bot, reply

CLIENT_CASE = "После доработки в нашей базе не работает расчет"
HANDOFF = reply("handoff", "Нужна проверка доработанной конфигурации.")
ANSWER = reply("answer", "Ответ.", ["faq:poly2-261#0"])


def offer(bot, session="s1"):
    r = ask(bot, CLIENT_CASE, session)
    assert r.kind is ReplyKind.HANDOFF and r.actions == ["handoff_confirm", "handoff_decline"]
    return r


def act(bot, action, session="s1", *, offer_id=None):
    return asyncio.run(bot.handoff_action(session, action, offer_id=offer_id))


def test_offer_asks_consent_and_sends_nothing():
    bot = make_bot(llm=ScriptedLLM(HANDOFF))
    r = offer(bot)
    assert texts.HANDOFF_TAIL in r.text and "сохранён" not in r.text
    assert bot.services.handoff.records() == []
    assert bot.store.get("s1").handoff.status == "offered"


def test_old_buttons_cannot_confirm_or_decline_the_new_offer():
    bot = make_bot(llm=ScriptedLLM(HANDOFF, HANDOFF))
    previous = offer(bot)
    current = offer(bot)
    assert previous.handoff_offer_id != current.handoff_offer_id
    assert current.handoff_offer_id
    refreshed = asyncio.run(bot.refresh_handoff_offer("s1"))
    assert refreshed.handoff_offer_id == current.handoff_offer_id
    assert refreshed.actions == current.actions
    assert bot.services.handoff.records() == []
    buttons = keyboard_rows(current.actions, current.trace_id, current.handoff_offer_id)[0]
    assert [value for _, value in buttons] == [
        f"handoff_confirm:{current.handoff_offer_id}", f"handoff_decline:{current.handoff_offer_id}",
    ]
    assert all(len(value.encode()) <= 64 for _, value in buttons)

    history = list(bot.store.get("s1").history)
    for action in ("handoff_confirm", "handoff_decline"):
        assert act(bot, action, offer_id=previous.handoff_offer_id).reason == "handoff_stale"
    state = bot.store.get("s1")
    assert state.history == history
    assert state.handoff.offer_id == current.handoff_offer_id and state.handoff.status == "offered"
    assert bot.services.handoff.records() == []

    assert act(bot, "handoff_confirm", offer_id=current.handoff_offer_id).reason == "handoff_sent"
    assert act(bot, "handoff_confirm", offer_id=current.handoff_offer_id).reason == "handoff_duplicate"
    assert len(bot.services.handoff.records()) == 1


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


def test_tool_error_then_retry_creates_exactly_one_record(tmp_path):
    class Flaky(MemoryHandoff):
        fail = True

        async def create(self, request):
            if self.fail:
                self.fail = False
                raise ToolError("handoff", ToolErrorKind.UNAVAILABLE, "нет связи")
            return await super().create(request)

    bot = make_bot(llm=ScriptedLLM(HANDOFF))
    bot.services.handoff = Flaky()
    bot.store = SQLiteSessionStore(tmp_path / "s.sqlite3")
    offered = offer(bot)
    r = act(bot, "handoff_confirm")
    assert r.reason == "handoff_unavailable" and r.text == texts.HANDOFF_FAILED
    assert r.actions == ["handoff_confirm", "handoff_decline"] and bot.services.handoff.records() == []
    assert act(bot, "handoff_confirm").reason == "handoff_sent"
    assert act(bot, "handoff_confirm").reason == "handoff_duplicate"
    recs = bot.services.handoff.records()
    assert len(recs) == 1 and recs[0].messages == [CLIENT_CASE]
    assert bot.store.handoff_attempt_summary() == {"saved": 1, "attempts": 2}
    attempts = bot.store._db.execute(
        "SELECT handoff_offer_id, handoff_attempt_result FROM turns "
        "WHERE handoff_attempt_result IS NOT NULL ORDER BY ts"
    ).fetchall()
    assert attempts == [(offered.handoff_offer_id, "failed"), (offered.handoff_offer_id, "saved")]
    bot.store.close()


def test_tool_timeout_is_reported_honestly():
    class Slow(MemoryHandoff):
        async def create(self, request):
            await asyncio.sleep(1)

    bot = make_bot(llm=ScriptedLLM(HANDOFF))
    bot.settings.handoff.timeout_s = 0.05
    bot.services.handoff = Slow()
    offer(bot)
    r = act(bot, "handoff_confirm")
    assert r.reason == "handoff_timeout" and r.text == texts.HANDOFF_TIMEOUT
    assert r.handoff_attempt_result == "timeout_unconfirmed"
    assert r.handoff_offer_id == bot.store.get("s1").handoff.offer_id


def test_timeout_checks_sqlite_before_reporting_the_result(tmp_path):
    class SavedBeforeTimeout(SQLiteHandoff):
        async def create(self, request):
            await super().create(request)
            raise TimeoutError

    bot = make_bot(llm=ScriptedLLM(HANDOFF))
    tool = SavedBeforeTimeout(tmp_path / "h.sqlite3")
    bot.services.handoff = tool
    offered = offer(bot)
    r = act(bot, "handoff_confirm", offer_id=offered.handoff_offer_id)
    assert r.reason == "handoff_sent"
    assert r.handoff_attempt_result == "saved"
    assert r.steps[0].output["confirmed_after_timeout"] is True
    assert tool.records("s1")[0].ticket_id in r.text
    assert len(tool.records()) == 1
    tool.close()


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
