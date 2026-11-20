"""Общий дедлайн хода (C09, polybot/deadline.py) и оценка 👍/👎 (M09) — без сети."""
import asyncio
import math
import time

import pytest

from polybot import deadline
from polybot.clients.llm import OpenAICompatibleLLM, ScriptedLLM
from polybot.errors import ToolError, ToolErrorKind
from polybot.schemas import ReplyKind
from polybot.settings import LLMSettings
from polybot.storage import SQLiteSessionStore
from polybot.telegram_bot import keyboard_rows
from tests.conftest import CHUNK, StaticRetriever, ask, make_bot, reply

MSG = [{"role": "user", "content": "привет"}]


def test_no_deadline_outside_turn():
    assert math.isinf(deadline.remaining())


def test_nested_turn_does_not_extend_outer():
    with deadline.turn(1.0):
        with deadline.turn(100.0):
            assert deadline.remaining() <= 1.0
    assert math.isinf(deadline.remaining())


class SlowLLM(ScriptedLLM):
    """Отвечает дольше, чем позволяет дедлайн хода."""

    def __init__(self, delay: float, *script):
        super().__init__(*script)
        self.delay = delay

    async def complete(self, messages, json_schema=None):
        await asyncio.sleep(self.delay)
        return await super().complete(messages, json_schema)


def test_slow_llm_cut_by_turn_deadline():
    bot = make_bot(llm=SlowLLM(5.0, reply("answer", "Ответ.", [CHUNK.chunk_id])), turn_deadline_s=0.3)
    t0 = time.perf_counter()
    r = ask(bot, "Как заменить материал в заказе?")
    assert time.perf_counter() - t0 < 2.0
    assert r.kind is ReplyKind.FALLBACK and r.reason == "llm_timeout"
    step = next(st for st in r.steps if st.node == "generate")
    assert step.output["deadline_left_s"] < 0.1             # по трассе видно, что оборвал дедлайн


def test_slow_search_cut_by_turn_deadline():
    class SlowRetriever(StaticRetriever):
        async def search(self, query, top_k, exclude_doc_ids=()):
            await asyncio.sleep(5.0)
            return await super().search(query, top_k, exclude_doc_ids)

    bot = make_bot(retriever=SlowRetriever([CHUNK]), turn_deadline_s=0.3)
    r = ask(bot, "Как заменить материал в заказе?")
    assert r.kind is ReplyKind.FALLBACK and r.reason == "retrieval_timeout"


class FakeAttempts(OpenAICompatibleLLM):
    """Попытки клиента без сети: каждая «длится» `cost` секунд и падает с `kind`."""

    def __init__(self, cfg, kind, cost=0.0):
        super().__init__(cfg)
        self.kind, self.cost, self.timeouts = kind, cost, []

    async def _attempt(self, params, timeout):
        self.timeouts.append(timeout)
        await asyncio.sleep(self.cost)
        raise ToolError("llm", self.kind, "тест")


def run_with_deadline(llm, seconds):
    async def go():
        with deadline.turn(seconds):
            await llm.complete(MSG)
    with pytest.raises(ToolError):
        asyncio.run(go())


def test_llm_attempt_timeout_capped_by_deadline_and_sdk_retries_off():
    llm = FakeAttempts(LLMSettings(timeout_s=12, min_retry_s=5), ToolErrorKind.TIMEOUT)
    assert llm._client.max_retries == 0
    run_with_deadline(llm, 8.0)
    assert len(llm.timeouts) == 2 and llm.timeouts[0] <= 8.0


def test_llm_no_retry_when_less_than_min_retry_left():
    llm = FakeAttempts(LLMSettings(timeout_s=12, min_retry_s=5), ToolErrorKind.TIMEOUT, cost=0.2)
    run_with_deadline(llm, 5.1)                      # после первой попытки остаётся < 5 с
    assert len(llm.timeouts) == 1


def test_llm_auth_error_not_retried():
    llm = FakeAttempts(LLMSettings(), ToolErrorKind.AUTH)
    run_with_deadline(llm, 15.0)
    assert len(llm.timeouts) == 1


# ---------------------------------------------------------------- M09: оценка 👍/👎
def test_answer_offers_feedback_clarify_does_not():
    bot = make_bot(llm=ScriptedLLM(reply("answer", "Ответ.", [CHUNK.chunk_id]),
                                   reply("clarify", "Уточните тип заказа.")))
    a = ask(bot, "Как заменить материал в заказе?", session="a")
    c = ask(bot, "Как заменить материал в заказе?", session="c")
    assert a.actions == ["feedback_up", "feedback_down", "solved"]
    assert c.actions == ["solved"]
    rows = keyboard_rows(a.actions, a.trace_id)
    assert [d for _, d in rows[0]] == [f"fb:up:{a.trace_id}", f"fb:down:{a.trace_id}"]
    assert all(len(d.encode()) <= 64 for row in rows for _, d in row)   # лимит Telegram


def test_feedback_stored_and_replaced(tmp_path):
    bot = make_bot(llm=ScriptedLLM(reply("answer", "Ответ.", [CHUNK.chunk_id])))
    bot.store = SQLiteSessionStore(tmp_path / "s.sqlite3")
    r = ask(bot, "Как заменить материал в заказе?")
    assert bot.feedback("s1", r.trace_id, "down")
    assert bot.feedback("s1", r.trace_id, "up")      # передумал — оценка заменяется
    assert bot.store.feedback_summary() == {"up": 1, "down": 0, "answers": 1}
    with pytest.raises(ValueError):
        bot.feedback("s1", r.trace_id, "meh")


def test_feedback_without_sqlite_is_noop():
    assert make_bot().feedback("s1", "t", "up") is False


# ---------------------------------------------------------------- выключатель генерации (7.4)
def test_generation_disabled_offers_consultant_without_llm():
    llm = ScriptedLLM()                               # пустой: любой вызов LLM — ошибка теста
    bot = make_bot(llm=llm)
    bot.settings.llm.enabled = False
    r = ask(bot, "Как заменить материал в заказе?")
    assert r.kind is ReplyKind.HANDOFF and r.reason == "generation_disabled"
    assert r.actions == ["handoff_confirm", "handoff_decline"] and llm.calls == []


# ---------------------------------------------------------------- история новой проблемы (6)
def test_prompt_history_reset_after_new_problem():
    llm = ScriptedLLM(reply("answer", "Ответ про материалы.", [CHUNK.chunk_id]),
                      reply("answer", "Ответ про склад.", [CHUNK.chunk_id]))
    bot = make_bot(llm=llm)
    ask(bot, "Как заменить материал в заказе?")
    bot.new_problem("s1")
    ask(bot, "Как вести склад материалов?")
    second_prompt = llm.calls[1][1]["content"]
    assert "Ответ про материалы" not in second_prompt and "Как заменить материал" not in second_prompt.split("Фрагменты")[0]
    assert len(bot.store.get("s1").history) == 4     # в истории реплики остаются


def test_prompt_history_kept_within_problem():
    llm = ScriptedLLM(reply("answer", "Ответ про материалы.", [CHUNK.chunk_id]),
                      reply("answer", "Уточнённый ответ.", [CHUNK.chunk_id]))
    bot = make_bot(llm=llm)
    ask(bot, "Как заменить материал в заказе?")
    ask(bot, "А если заказ уже рассчитан?")            # без /new — та же проблема
    assert "Ответ про материалы" in llm.calls[1][1]["content"]


def test_problem_marker_survives_restart(tmp_path):
    store = SQLiteSessionStore(tmp_path / "s.sqlite3")
    st = store.get("s1")
    st.new_problem()
    store.save(st)
    assert store.get("s1").problem_started_at == st.problem_started_at > 0
