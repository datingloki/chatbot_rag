"""C08: маскирование до LLM, журнала и состояния сессии."""
from polybot.clients.llm import ScriptedLLM
from polybot.privacy import mask_pii
from polybot.storage import SQLiteSessionStore
from tests.conftest import ask, make_bot, reply

RAW = "Иванов Иван Иванович, тел. +7 (916) 123-45-67, ivan@mail.ru, ИНН 7701234567"


def test_masks_contacts_ids_and_names():
    out, counts = mask_pii(RAW + ", договор № 15/2024-П, р/с 40702810900000012345, пароль: qwerty123")
    for raw in ("Иванов", "916", "ivan@", "7701234567", "15/2024", "40702810900000012345", "qwerty123"):
        assert raw not in out
    assert {"person", "phone", "email", "inn", "contract", "account", "secret"} <= set(counts)


def test_keeps_product_details():
    for text in ("Обновили до релиза 3.0.12.34 от 12.11.2026, сумма 150000 руб",
                 "В Расчете Заказа на закладке «Материалы» нет Материалов",
                 "Вера в Печать, заказ на 5000 листов",
                 "Как учесть замену материала после расчета заказа?"):
        assert mask_pii(text) == (text, {})


def test_names_inside_capitalized_run_and_initials():
    assert mask_pii("Менеджер Иванов Иван Иванович не видит заказ")[0] == "Менеджер [ФИО] не видит заказ"
    assert mask_pii("Звонил Петров И.И.")[0] == "Звонил [ФИО]"
    assert mask_pii("Релиз 3.0.12.34, IP сервера: 10.0.0.5")[0] == "Релиз 3.0.12.34, IP сервера: [IP_ADDRESS]"


def test_raw_values_never_reach_llm_state_or_log(tmp_path):
    llm = ScriptedLLM(reply("answer", "Ответ.", ["faq:poly2-261#0"]))
    bot = make_bot(llm=llm)
    bot.store = SQLiteSessionStore(tmp_path / "s.sqlite3")
    r = ask(bot, f"Как учесть замену материала? {RAW}")
    prompt = "\n".join(m["content"] for m in llm.calls[0])
    state = bot.store._db.execute("select state from sessions").fetchone()[0]
    log = bot.store._db.execute("select message from turns").fetchone()[0]
    for where in (prompt, state, log):
        for raw in ("Иванов", "123-45-67", "ivan@mail.ru", "7701234567"):
            assert raw not in where
    assert "[PHONE]" in log
    assert r.steps[0].input["pii_masked"] == {"person": 1, "phone": 1, "email": 1, "inn": 1}
