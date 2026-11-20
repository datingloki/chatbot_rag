"""Telegram-обёртка и SQLite-хранилище — без сети."""
import asyncio

from polybot.clients.llm import ScriptedLLM
from polybot.schemas import LLMUsage, PendingClarification, ReplyKind, Response, Source
from polybot.storage import SQLiteSessionStore
from polybot.telegram_bot import (OMNIDESK_SOURCE, build_dispatcher, format_reply, session_id,
                                  split_message)
from tests.conftest import ask, make_bot, reply


def _bot(tmp_path, **kw):
    bot = make_bot(**kw)
    bot.store = SQLiteSessionStore(tmp_path / "s.sqlite3")
    return bot


def test_state_survives_restart(tmp_path):
    bot = _bot(tmp_path)
    ask(bot, "??? не считается бумага")                  # уточнение домена
    bot.store.close()

    store = SQLiteSessionStore(tmp_path / "s.sqlite3")
    s = store.get("s1")
    assert s.pending is PendingClarification.DOMAIN
    assert s.clarifications == 1 and s.chain == ["??? не считается бумага"]
    assert [t.role for t in s.history] == ["user", "assistant"]
    assert s.history[1].kind is ReplyKind.CLARIFY
    assert store.get("другой").clarifications == 0


def test_history_is_trimmed(tmp_path):
    bot = _bot(tmp_path)
    bot.store.keep_history = 4
    for i in range(5):
        ask(bot, f"??? {i}")
    assert len(bot.store.get("s1").history) == 4


def test_turns_are_logged(tmp_path):
    bot = _bot(tmp_path, llm=ScriptedLLM(reply("answer", "Ответ.", ["faq:poly2-261#0"])))
    r = ask(bot, "Как учесть замену материала?")
    row = bot.store._db.execute("SELECT session_id, kind, reason, sources, path FROM turns").fetchone()
    assert row[:3] == ("s1", "answer", r.reason)
    assert "faq:poly2-261" in row[3] and row[4].startswith("input_check")


def test_messages_of_one_chat_are_serialized():
    bot = make_bot()

    async def two():
        await asyncio.gather(bot.respond("c", "??? a"), bot.respond("c", "??? b"))

    asyncio.run(two())
    s = bot.store.get("c")
    assert s.clarifications == 2 and len(s.history) == 4    # без гонки ни один ход не потерян


def _resp(text, sources):
    return Response(trace_id="t", session_id="tg:1", kind=ReplyKind.ANSWER, text=text, sources=sources,
                    steps=[], latency_ms=1.0, llm=LLMUsage())


def test_format_reply_escapes_and_links():
    r = _resp("Поле <Сумма> & «Итог»", [
        Source(title="Замена материала", url="https://1cpoly.ru/q?a=1&id=2", doc_id="faq:poly2-261"),
        Source(title="6.2 Заказ", url="Руководство «УНФ 8. Полиграфия 2», ред. 3.0", doc_id="manual:6.2"),
        Source(title="Клиент ООО Ромашка", url="", doc_id="omnidesk:42"),
        Source(title="Другое обращение", url="", doc_id="omnidesk:43"),
    ])
    out = format_reply(r)
    assert "Поле &lt;Сумма&gt; &amp; «Итог»" in out
    assert '<a href="https://1cpoly.ru/q?a=1&amp;id=2">Замена материала</a>' in out
    assert "6.2 Заказ — Руководство" in out
    assert "Ромашка" not in out and out.count(OMNIDESK_SOURCE) == 1


def test_markdown_residue_is_cleaned():
    text = ("### Способы\n1. **В документе «Расчет заказа»:**\n   - На закладке «Материалы» выберите `Замена`.\n"
            "* Второй пункт с a*b и **незакрытым")
    out = format_reply(_resp(text, []))
    assert "*" not in out and "#" not in out and "`" not in out
    assert "<b>Способы</b>" in out and "<b>В документе «Расчет заказа»:</b>" in out
    assert "   – На закладке «Материалы» выберите Замена." in out
    assert "– Второй пункт" in out


def test_split_long_message_keeps_lines_whole():
    lines = [f"строка {i} " + "x" * 50 for i in range(200)]
    parts = split_message("\n".join(lines), limit=1000)
    assert all(len(p) <= 1000 for p in parts)
    assert "\n".join(parts).split("\n") == lines
    assert split_message("y" * 2500, limit=1000) == ["y" * 1000, "y" * 1000, "y" * 500]


def test_dispatcher_builds():
    dp = build_dispatcher(make_bot(), allowed={1})
    assert dp.sub_routers and session_id(5) == "tg:5"
