"""Маршрутизация по графу: каждый тест — сценарий из design doc 2.2 / plan_rag_bot.md."""
from polybot.clients.llm import ScriptedLLM
from polybot.clients.retriever import StaticRetriever
from polybot.errors import ToolError, ToolErrorKind
from polybot.schemas import PendingClarification, ReplyKind
from polybot import texts
from tests.conftest import CHUNK, WEAK, ask, make_bot, reply


# ---------------------------------------------------------------- вход и домен
def test_empty_and_too_long_rejected_without_tools():
    llm = ScriptedLLM()
    bot = make_bot(llm=llm)
    assert ask(bot, "   ").kind is ReplyKind.REJECT
    r = ask(bot, "а" * 2001)
    assert r.kind is ReplyKind.REJECT and r.reason == "too_long"
    assert r.path == ["input_check", "reject", "final"]
    assert llm.calls == []


def test_out_of_domain_refused_without_search_or_llm():
    retriever = StaticRetriever([CHUNK])
    bot = make_bot(retriever=retriever)
    r = ask(bot, "Подскажи рецепт борща")
    assert r.kind is ReplyKind.REFUSE
    assert r.path == ["input_check", "domain", "refuse", "final"]
    assert retriever.queries == []


def test_unclear_asks_domain_clarification():
    bot = make_bot()
    r = ask(bot, "??? печать")
    assert r.kind is ReplyKind.CLARIFY and r.text == texts.CLARIFY_DOMAIN
    s = bot.store.get("s1")
    assert s.pending is PendingClarification.DOMAIN
    assert s.clarifications == 1
    assert s.chain == ["??? печать"]


# ---------------------------------------------------------------- после уточнения домен решается заново
def test_after_clarification_in_domain_goes_to_rag_with_full_question():
    retriever = StaticRetriever([CHUNK])
    llm = ScriptedLLM(reply("answer", "Замените материал на закладке «Материалы».", ["faq-261#0"]))
    bot = make_bot(retriever=retriever, llm=llm)
    ask(bot, "??? не считается бумага")
    r = ask(bot, "да, в заказе")

    assert r.kind is ReplyKind.ANSWER
    assert r.path == ["input_check", "domain", "retrieve", "evidence_gate", "generate",
                      "validate", "final"]
    # домен определён по ответу вместе с исходным вопросом
    domain_step = next(s for s in r.steps if s.node == "domain")
    assert domain_step.output["with_history"] is True
    # искали по исходному вопросу + уточнению, а не по «да, в заказе»
    assert retriever.queries == ["??? не считается бумага\nда, в заказе"]
    assert r.sources[0].url == CHUNK.url
    s = bot.store.get("s1")
    assert s.pending is None and s.chain == []


def test_after_clarification_out_of_domain_refused():
    bot = make_bot()
    ask(bot, "??? помогите")
    r = ask(bot, "какая погода завтра")
    assert r.kind is ReplyKind.REFUSE
    assert bot.store.get("s1").pending is None


def test_after_clarification_still_unclear_asks_again():
    bot = make_bot()
    ask(bot, "??? помогите")
    r = ask(bot, "??? ну это")
    assert r.kind is ReplyKind.CLARIFY
    s = bot.store.get("s1")
    assert s.clarifications == 2
    assert s.chain == ["??? помогите", "??? ну это"]


def test_sixth_clarification_is_never_sent():
    bot = make_bot()
    kinds = [ask(bot, f"??? {i}").kind for i in range(6)]
    assert kinds == [ReplyKind.CLARIFY] * 5 + [ReplyKind.HANDOFF]
    s = bot.store.get("s1")
    assert s.clarifications == 0 and s.pending is None   # передача закрывает проблему


def test_clarification_limit_is_shared_between_domain_and_no_answer():
    bot = make_bot(retriever=StaticRetriever([WEAK]))   # ниже порога → no_answer
    for i in range(3):
        assert ask(bot, f"??? {i}").kind is ReplyKind.CLARIFY
    for i in range(2):
        assert ask(bot, f"вопрос {i}").kind is ReplyKind.CLARIFY
    r = ask(bot, "ещё вопрос")
    assert r.kind is ReplyKind.HANDOFF and r.reason == "no_evidence_clarify_limit"


# ---------------------------------------------------------------- RAG и проверка ответа
def test_no_evidence_is_not_out_of_domain():
    bot = make_bot(retriever=StaticRetriever([WEAK]))
    r = ask(bot, "Как учесть замену материала?")
    assert r.kind is ReplyKind.CLARIFY and r.reason == "no_evidence"
    assert "no_answer" in r.path


def test_grounded_answer_has_source():
    llm = ScriptedLLM(reply("answer", "Замените материал на закладке «Материалы».", ["faq-261#0"]))
    bot = make_bot(llm=llm)
    r = ask(bot, "Как учесть замену материала?")
    assert r.kind is ReplyKind.ANSWER
    assert [s.url for s in r.sources] == [CHUNK.url]
    assert r.llm.calls == 1
    # слабый фрагмент в промпт не попал
    prompt = llm.calls[0][1]["content"]
    assert "faq-261#0" in prompt and "faq-7#0" not in prompt


def test_answer_with_unknown_citation_goes_to_fallback():
    bot = make_bot(llm=ScriptedLLM(reply("answer", "Ответ.", ["faq-999#0"])))
    r = ask(bot, "Как учесть замену материала?")
    assert r.kind is ReplyKind.FALLBACK and r.reason == "validation_unknown_citation"
    assert r.text == texts.FALLBACK_UNVERIFIED
    assert r.sources == []


def test_answer_without_citation_goes_to_fallback():
    bot = make_bot(llm=ScriptedLLM(reply("answer", "Ответ.", [])))
    assert ask(bot, "вопрос").reason == "validation_uncited_answer"


def test_sql_advice_blocked():
    text = "Выполните UPDATE Документы SET Сумма = 0 WHERE Номер = 5."
    bot = make_bot(llm=ScriptedLLM(reply("answer", text, ["faq-261#0"])))
    assert ask(bot, "вопрос").reason == "validation_unsafe_db_advice"


def test_false_ticket_claim_blocked():
    bot = make_bot(llm=ScriptedLLM(reply("handoff", "Заявка создана, консультант свяжется.")))
    assert ask(bot, "вопрос").reason == "validation_false_handoff_claim"


def test_invalid_json_goes_to_fallback():
    bot = make_bot(llm=ScriptedLLM("Вот ответ без JSON"))
    assert ask(bot, "вопрос").reason == "validation_bad_json"


def test_llm_clarify_counts_and_sets_pending():
    bot = make_bot(llm=ScriptedLLM(reply("clarify", "Какой тип заказа?")))
    r = ask(bot, "Не считается заказ")
    assert r.kind is ReplyKind.CLARIFY
    s = bot.store.get("s1")
    assert s.pending is PendingClarification.QUESTION and s.clarifications == 1


def test_llm_clarify_after_limit_becomes_handoff():
    bot = make_bot(llm=ScriptedLLM(reply("clarify", "Какой тип заказа?")))
    bot.store.get("s1").clarifications = 5
    r = ask(bot, "Не считается заказ")
    assert r.kind is ReplyKind.HANDOFF and r.reason == "clarify_limit"


def test_llm_handoff_for_client_specific_case():
    bot = make_bot(llm=ScriptedLLM(reply("handoff", "Нужна диагностика вашей базы.")))
    r = ask(bot, "После обновления ошибка в нашей рабочей базе")
    assert r.kind is ReplyKind.HANDOFF
    assert texts.HANDOFF_TAIL in r.text


# ---------------------------------------------------------------- второй рубеж домена в промпте
def test_prompt_contains_domain_guard():
    llm = ScriptedLLM(reply("answer", "Ответ.", ["faq-261#0"]))
    ask(make_bot(llm=llm), "вопрос")
    system = llm.calls[0][0]["content"]
    assert '"out_of_domain"' in system and '"professional_advice"' in system


def test_llm_out_of_domain_refused_with_template():
    # классификатор пропустил посторонний вопрос (keyword_domain считает его доменным)
    bot = make_bot(llm=ScriptedLLM(reply("out_of_domain", "")))
    r = ask(bot, "Напиши рекламный текст для типографии")
    assert r.kind is ReplyKind.REFUSE
    assert r.reason == "llm_out_of_domain"
    assert r.text == texts.REFUSE
    assert r.sources == []
    validate = next(s for s in r.steps if s.node == "validate")
    assert validate.output["classifier_label"] == "in_domain"


def test_llm_refusal_text_is_never_shown():
    # даже если модель «заодно» выполнила постороннюю задачу, пользователь видит только шаблон
    bot = make_bot(llm=ScriptedLLM(reply("out_of_domain", "Вот ваш рекламный текст: ...")))
    r = ask(bot, "Напиши рекламный текст")
    assert r.text == texts.REFUSE


def test_llm_professional_advice_refused():
    bot = make_bot(llm=ScriptedLLM(reply("professional_advice", "")))
    r = ask(bot, "Можно ли уволить сотрудника за опоздание?")
    assert r.kind is ReplyKind.REFUSE
    assert r.reason == "llm_professional_advice"
    assert r.text == texts.REFUSE_PROFESSIONAL


def test_llm_refusal_closes_clarification_chain():
    bot = make_bot(llm=ScriptedLLM(reply("out_of_domain", "")))
    ask(bot, "??? помогите")
    ask(bot, "с отчётом в Excel")
    s = bot.store.get("s1")
    assert s.pending is None and s.chain == []


# ---------------------------------------------------------------- сбои инструментов и бюджет
def test_retrieval_failure_gives_tool_fallback_and_keeps_state():
    bot = make_bot()
    ask(bot, "??? не считается")
    bot.services.retriever = StaticRetriever(error=ToolErrorKind.UNAVAILABLE)
    r = ask(bot, "да, в заказе")
    assert r.kind is ReplyKind.FALLBACK and r.reason == "retrieval_unavailable"
    assert r.text == texts.FALLBACK_TOOL
    # цепочка уточнения сохранилась: после восстановления поиска можно продолжить
    s = bot.store.get("s1")
    assert s.pending is PendingClarification.DOMAIN and s.chain == ["??? не считается"]

    bot.services.retriever = StaticRetriever([CHUNK])
    bot.services.llm = ScriptedLLM(reply("answer", "Ответ по документации.", ["faq-261#0"]))
    assert ask(bot, "да, в заказе").kind is ReplyKind.ANSWER


def test_llm_timeout_gives_tool_fallback():
    bot = make_bot(llm=ScriptedLLM(ToolError("llm", ToolErrorKind.TIMEOUT)))
    r = ask(bot, "вопрос")
    assert r.kind is ReplyKind.FALLBACK and r.reason == "llm_timeout"
    assert r.text == texts.FALLBACK_TOOL


def test_step_budget_exhaustion_ends_in_fallback():
    llm = ScriptedLLM(reply("answer", "Ответ.", ["faq-261#0"]))
    bot = make_bot(llm=llm, max_steps=3)
    r = ask(bot, "вопрос")
    assert r.kind is ReplyKind.FALLBACK and r.reason == "budget_exhausted"
    assert r.path[-2:] == ["fallback", "final"]
    assert llm.calls == []          # до генерации не дошли


def test_unexpected_node_error_ends_in_fallback():
    class Broken:
        def predict(self, text, history=None):
            raise RuntimeError("модель не загрузилась")

    r = ask(make_bot(domain=Broken()), "вопрос")
    assert r.kind is ReplyKind.FALLBACK and r.reason == "internal_error"
    assert r.steps[1].error.startswith("RuntimeError")


def test_every_step_is_recorded():
    bot = make_bot(llm=ScriptedLLM(reply("answer", "Ответ.", ["faq-261#0"])))
    r = ask(bot, "вопрос")
    assert [s.node for s in r.steps] == r.path
    gen = next(s for s in r.steps if s.node == "generate")
    assert gen.output["prompt_version"].startswith("generate_v0@")
    final = r.steps[-1]
    assert "state_before" in final.output and "state_after" in final.output
