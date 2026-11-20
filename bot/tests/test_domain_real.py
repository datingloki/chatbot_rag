"""Настоящий классификатор домена из ../domain_classifier (без LLM и без поиска)."""
import pytest

from polybot.bot import Bot
from polybot.clients.llm import ScriptedLLM
from polybot.clients.retriever import NullRetriever
from polybot.schemas import ReplyKind
from polybot.settings import Settings
from tests.conftest import ask

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def bot():
    return Bot.from_settings(Settings(_env_file=None), llm=ScriptedLLM(), retriever=NullRetriever())


def test_borscht_refused(bot):
    assert ask(bot, "Подскажи рецепт борща", "a").kind is ReplyKind.REFUSE


def test_install_without_object_is_unclear(bot):
    # «что поставить?» — может быть наша программа: уточняем, а не отказываем (R5)
    r = ask(bot, "Как поставить на компьютер?", "inst")
    assert r.kind is ReplyKind.CLARIFY and r.reason == "domain_unclear"
    assert ask(bot, "Как установить Python на компьютер?", "inst2").kind is ReplyKind.REFUSE


def test_product_question_goes_to_search(bot):
    r = ask(bot, "Как обновить релиз 1С:Полиграфии?", "b")
    assert "retrieve" in r.path
    # поиска на этапе 0 нет → честное «нет ответа в базе», а не отказ
    assert r.kind is ReplyKind.CLARIFY and r.reason == "no_evidence"


def test_short_answer_to_clarification_uses_history(bot):
    ask(bot, "Не считается бумага", "c")
    r = ask(bot, "Да, в заказе", "c")
    domain = next(s for s in r.steps if s.node == "domain")
    assert domain.output["with_history"] is True
    assert r.kind is not ReplyKind.REFUSE


# ---------------------------------------------------------------- загрузка и версия
SAMPLES = [
    ("Подскажи рецепт борща", None),
    ("Как обновить релиз 1С:Полиграфии?", None),
    ("Можно ли уволить сотрудника за прогул по ТК?", None),
    ("Как настроить роутер дома?", None),
    ("Консультация по лицензии ИТС", None),
    ("принтер", None),
    ("Да, в заказе", [{"role": "user", "text": "Не считается бумага"}]),
]


def test_same_decisions_as_classifier_with_rules(bot):
    """Обёртка (модель + rules.py по пути) решает так же, как DomainClassifier.predict()."""
    import sys

    svc = bot.services.domain
    clf_module = sys.modules["domain_clf"]
    original = clf_module.DomainClassifier.load()
    # Эталонный predict делает `from rules import apply_rules`: на время теста подставляем
    # тот же rules.py, загруженный по пути.
    sys.modules["rules"] = sys.modules["polybot_domain_rules"]
    try:
        for text, history in SAMPLES:
            assert svc.predict(text, history) == original.predict(text, history=history), text
    finally:
        del sys.modules["rules"]
    assert any(svc.predict(t, h)["rule"] for t, h in SAMPLES), "ни одно правило не сработало"


def test_classifier_dir_not_added_to_sys_path(bot):
    import sys

    assert not any(p.rstrip("/").endswith("domain_classifier") for p in sys.path)
    assert "rules" not in sys.modules


def test_version_and_dev_check(bot):
    svc = bot.services.domain
    assert svc.version.startswith("domain_clf@") and len(svc.version) == len("domain_clf@") + 12
    assert svc.dev_check["status"] == "ok", svc.dev_check
    r = ask(bot, "Подскажи рецепт борща", "v")
    domain = next(s for s in r.steps if s.node == "domain")
    assert domain.output["classifier_version"] == svc.version
