"""Харнесс: подготовка состояния кейса, проверка источника, метрики — без LLM и без индекса."""
import asyncio

from harness.run import SourceChecker, summarize, wilson
from harness.systems import RagSystem, SystemReply, origin_omnidesk, seed_session
from polybot.clients.llm import ScriptedLLM
from polybot.clients.retriever import StaticRetriever
from polybot.schemas import PendingClarification
from polybot.settings import Settings
from tests.conftest import CHUNK, keyword_domain, reply
from polybot.clients.domain import FixedDomain

CASE = {"case_id": "D1", "origin": "real_anonymized_candidate:omnidesk:42:message:2",
        "primary_class": "K5", "expected_action": "answer", "origin_type": "real_anonymized_candidate",
        "clarification_count": 1, "context": {},
        "input": {"history": [{"role": "user", "text": "Не считается бумага"},
                              {"role": "assistant", "text": "В каком документе?"}],
                  "user_message": "В заказе"},
        "evidence": [{"source_case_id": "42"}, {"faq_id": "poly2-261"}]}


def test_seed_session_from_case_history():
    s = seed_session(CASE)
    assert s.clarifications == 1
    assert s.pending is PendingClarification.QUESTION
    assert s.chain == ["Не считается бумага"]
    assert [t.role for t in s.history] == ["user", "assistant"]


def test_origin_case_detected():
    assert origin_omnidesk(CASE) == "42"
    assert origin_omnidesk({"origin": "official_faq_entry:poly2-260"}) is None


def test_rag_system_excludes_origin_and_uses_chain():
    retriever = StaticRetriever([CHUNK])
    rag = RagSystem(Settings(_env_file=None), llm=ScriptedLLM(reply("answer", "Ответ.", [CHUNK.chunk_id])),
                    retriever=retriever, domain=FixedDomain(keyword_domain))
    r = asyncio.run(rag.reply(CASE))
    assert r.action == "answer"
    assert retriever.excluded == [["omnidesk:42"]]
    assert retriever.queries == ["Не считается бумага\nВ заказе"]
    assert r.sources[0]["doc_id"] == "faq:poly2-261"
    assert rag.info()["prompt_version"].startswith("generate_v0@")


def _reply(action, docs=()):
    return SystemReply(action=action, text="", reason=None, latency_ms=1.0,
                       sources=[{"doc_id": d, "title": "", "url": ""} for d in docs])


def test_source_verdicts():
    ch = SourceChecker()
    gold = ch.gold_docs(CASE, exclude="42")
    assert gold == {"faq:poly2-261"}                         # своё обращение исключено
    assert ch.verdict(_reply("answer", ["faq:poly2-261"]), gold) == "source_ok"
    assert ch.verdict(_reply("answer", ["faq:poly2-1"]), gold) == "source_mismatch"
    assert ch.verdict(_reply("answer", ["manual:6.2"]), gold) == "needs_review"   # другой тип источника
    assert ch.verdict(_reply("answer", ["faq:x"]), set()) == "needs_review"
    assert ch.verdict(_reply("clarify"), gold) is None


def _row(cls, expected, action, **kw):
    return {"class": cls, "expected_action": expected, "action": action, "route_ok": expected == action,
            "origin_type": "x", "source_verdict": kw.get("verdict"), "sources": kw.get("sources", []),
            "reason": kw.get("reason"), "latency_ms": kw.get("latency", 10.0),
            "llm": kw.get("llm", {"calls": 1, "cost_rub": 0.5})}


def test_summary_metrics():
    rows = [_row("K1", "refuse", "refuse"), _row("K1", "refuse", "answer", verdict="needs_review"),
            _row("K2", "answer", "answer", verdict="source_ok", sources=[{"doc_id": "faq:1"}]),
            _row("K3", "clarify", "clarify"), _row("K6", "offer_handoff", "offer_handoff"),
            _row("K4", "offer_handoff", "fallback", reason="retrieval_timeout")]
    s = summarize(rows)
    assert s["M01r_route_ok_all"]["n"] == 4 and s["M01r_route_ok_all"]["N"] == 6
    assert s["M17_handoff_offered_domain"]["n"] == 1 and s["M17_handoff_offered_domain"]["N"] == 4
    assert s["M04r_K1_refused"]["share"] == 0.5
    assert s["M06r_K4_K6_handoff_offered"]["n"] == 1
    assert s["answers"]["M02r_source_ok_among_verifiable"]["share"] == 1.0
    assert s["fallback_reasons"] == {"retrieval_timeout": 1}
    assert s["llm"]["cost_rub"] == 3.0


def test_wilson_interval():
    lo, hi = wilson(50, 100)
    assert lo < 0.5 < hi and wilson(0, 0) is None


# ---------------------------------------------------------------- сценарии
def test_scenario_with_fault_and_expectations():
    from harness.scenarios import TurnScriptLLM, run_scenario
    from polybot.bot import Bot
    from polybot.pipeline.context import Services

    spec = {"id": "T1", "faults": [{"turn": 1, "tool": "retrieve", "kind": "timeout"}], "turns": [
        {"user": "Как учесть замену материала?", "expect": {"kind": "fallback", "reason": "retrieval_timeout"}},
        {"user": "Как учесть замену материала?",
         "llm": reply("answer", "Ответ по закладке.", [CHUNK.chunk_id]),
         "expect": {"kind": "answer", "sources_any": ["faq:poly2-261"], "must_contain": ["закладке"]}},
        {"command": "new_problem"},
        {"user": "??? ну это", "expect": {"kind": "clarify", "clarifications": 1}},
        {"user": "Как учесть замену материала?", "llm": reply("answer", "Ответ.", ["faq:nope#0"]),
         "expect": {"kind": "answer"}},      # намеренный провал: неизвестная ссылка → fallback
    ]}
    scripted = TurnScriptLLM()
    bot = Bot(Settings(_env_file=None), Services(domain=FixedDomain(keyword_domain), llm=scripted,
                                                 retriever=StaticRetriever([CHUNK])))
    res = asyncio.run(run_scenario(spec, bot, scripted))
    assert [t["failures"] == [] for t in res["turns"]] == [True, True, True, False]
    assert "kind: ждали answer" in res["turns"][3]["failures"][0]
    assert res["passed"] is False


def test_m01a_route_and_source():
    rows = [_row("K2", "answer", "answer", verdict="source_ok"),
            _row("K2", "answer", "answer", verdict="source_mismatch"),     # маршрут верен, источник нет
            _row("K2", "answer", "answer", verdict="needs_review"),
            _row("K1", "refuse", "refuse"), _row("K3", "clarify", "answer", verdict="source_ok")]
    s = summarize(rows)
    assert s["M01a_route_and_source"]["n"] == 2 and s["M01a_needs_review"] == 1
    assert s["M01r_route_ok_all"]["n"] == 4


def test_canary_frozen_list_matches_rule():
    from collections import Counter
    from harness.canary import CLASSES, canary_cases, load_dev, load_ids, select

    dev, ids = load_dev(), load_ids()
    assert ids == select(dev)                        # список воспроизводится правилом и seed
    cases = canary_cases(dev, ids)
    assert Counter(c["primary_class"] for c in cases) == {k: 10 for k in CLASSES}
