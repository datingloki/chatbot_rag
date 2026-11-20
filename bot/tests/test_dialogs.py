"""Полные диалоги: поле dialog и прогон с симулятором пользователя — без сети и индекса."""
import asyncio
import json

from harness.dialogs import NO_MORE, ScriptedUser, build, derive_dialog, run_dialog, summarize
from harness.run import SourceChecker
from polybot.clients.llm import ScriptedLLM
from polybot.clients.retriever import StaticRetriever
from tests.conftest import CHUNK, make_bot, reply

K5 = {"case_id": "D1", "primary_class": "K5", "expected_action": "answer", "clarification_limit": 5,
      "origin": "synthetic_from_manual", "origin_type": "synthetic_from_manual",
      "input": {"history": [{"role": "user", "text": "Подскажите по теме материалов."},
                            {"role": "assistant", "text": "Что именно нужно?"}],
                "user_message": "Как учесть замену материала после расчета заказа?"},
      "evidence": [{"faq_id": "poly2-261"}]}
K6_CHAIN = {**K5, "case_id": "D2", "primary_class": "K6", "expected_action": "offer_handoff",
            "input": {"history": [{"role": "user", "text": "Не работает обновление."},
                                  *[t for i in range(4) for t in ({"role": "assistant", "text": f"Вопрос {i}?"},
                                                                  {"role": "user", "text": f"Ответ {i}."})],
                                  {"role": "assistant", "text": "Вопрос 4?"}],
                      "user_message": "Текста ошибки нет."}}


def test_dialog_derived_from_case_fields():
    d = derive_dialog(K5)
    assert d["first_message"] == "Подскажите по теме материалов."
    assert d["user_replies"] == ["Как учесть замену материала после расчета заказа?"]
    assert d["reference_questions"] == ["Что именно нужно?"]
    assert d["expected_outcome"] == "answer" and d["max_clarifications"] == 5
    assert len(derive_dialog(K6_CHAIN)["user_replies"]) == 5
    assert derive_dialog({**K5, "primary_class": "K3"}) is None      # у K3 нет ответа пользователя и итога


def test_scripted_user_runs_out_of_replies():
    d = derive_dialog(K5)
    assert asyncio.run(ScriptedUser().reply(d, "?", 0))[0] == d["user_replies"][0]
    assert asyncio.run(ScriptedUser().reply(d, "?", 1))[0] == NO_MORE


def test_bot_asks_then_answers():
    llm = ScriptedLLM(reply("clarify", "Что именно нужно?"),
                      reply("answer", "Замените материал на закладке «Материалы».", [CHUNK.chunk_id]))
    bot = make_bot(llm=llm, retriever=StaticRetriever([CHUNK]))
    case = {**K5, "dialog": derive_dialog(K5)}
    row = asyncio.run(run_dialog(bot, case, ScriptedUser(), 0, SourceChecker()))
    assert row["outcome_ok"] and row["final_action"] == "answer"
    assert row["clarifications"] == 1 and not row["over_limit"] and row["after_replies_exhausted"] == 0
    assert row["source_verdict"] == "source_ok"
    assert row["bot_questions"] == ["Что именно нужно?"]
    # второй ход бот получил ответ пользователя, а не готовую историю
    assert row["turns"][1]["user"] == K5["input"]["user_message"]


def test_bot_that_keeps_asking_stops_at_limit():
    bot = make_bot(llm=ScriptedLLM(*[reply("clarify", f"Ещё вопрос {i}?") for i in range(8)]),
                   retriever=StaticRetriever([CHUNK]))
    case = {**K6_CHAIN, "dialog": derive_dialog(K6_CHAIN)}
    row = asyncio.run(run_dialog(bot, case, ScriptedUser(), 0, SourceChecker()))
    assert row["final_action"] == "offer_handoff" and row["final_reason"] == "clarify_limit"
    assert row["clarifications"] == 5 and not row["over_limit"] and row["outcome_ok"]


def test_answer_without_asking_is_counted():
    bot = make_bot(llm=ScriptedLLM(reply("answer", "Ответ.", [CHUNK.chunk_id])), retriever=StaticRetriever([CHUNK]))
    row = asyncio.run(run_dialog(bot, {**K5, "dialog": derive_dialog(K5)}, ScriptedUser(), 0, SourceChecker()))
    s = summarize([row])
    assert s["D1_outcome_ok"]["n"] == 1 and s["asked_on_first_turn"]["n"] == 0
    assert s["D3_clarifications"]["distribution"] == {0: 1}


def test_build_keeps_file_format(tmp_path):
    p = tmp_path / "cases.jsonl"
    rows = [K5, {**K5, "case_id": "D3", "primary_class": "K2"}]
    p.write_text("".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n" for r in rows),
                 encoding="utf-8")
    rep = build(p, write=True)
    assert rep["dialogs"] == {"K5": 1}
    lines = p.read_text(encoding="utf-8").splitlines()
    assert '"dialog":{' in lines[0] and "dialog" not in lines[1] and ", " not in lines[1]
    assert build(p, write=True)["dialogs"] == {"K5": 1}                # повторный запуск ничего не ломает
