"""Слой правил поверх модели (дизайн-док 2.2: «правила по темам и терминам, история
диалога и явные исключения; совпадения по одному слову недостаточно для уверенного решения»).

Правила не выносят отказ по одному слову сами по себе, а корректируют решение модели:

R1 domain_term   — модель отказывает, но в сообщении есть термин продукта/сопровождения
                   (1С, ИТС, КП, СЛК, лицензия, релиз, техподдержка…) и нет признаков
                   проф. консультации → unclear (C05: без уточнения не отказываем).
R2 short_ambig   — короткое сообщение (≤ 4 значимых слов) с объектом, который есть в
                   программе (принтер, отчёт, зарплата, штрихкод…) → unclear вместо отказа.
R3 prof_consult  — признаки проф. юридической/налоговой/кадровой консультации («по закону»,
                   «ТК РФ», штраф, суд…) без указания на программу → out_of_domain (C05).
R4 third_party   — модель не уверена (unclear), в сообщении названо стороннее ПО/сервис/
                   устройство, и нет ни терминов продукта, ни указания на программу, ни
                   интеграций из базы знаний → out_of_domain. Решение модели «in_domain»
                   R4 не меняет, поэтому новых ложных отказов на уверенных доменных не даёт.

Словари подбирались только по dev. Названия, встречающиеся только в eval (Outlook, CorelDRAW),
в словарь R4 сознательно не включены, чтобы eval оставался проверкой обобщения.
"""
from __future__ import annotations

from domain_clf import _lemma, tokens

DOMAIN_TERMS = {
    "1с", "полиграфия", "унф", "итс", "кп", "слк", "лицензия", "лицензионный", "релиз", "конфигурация",
    "конфигуратор", "техподдержка", "поддержка", "армекс", "обновление", "обновить", "платформа",
    "рег", "регистрационный", "консультация",
}
AMBIGUOUS_OBJECTS = {
    "принтер", "печать", "распечатать", "сканер", "штрихкод", "штрих", "отчёт", "отчет", "зарплата",
    "себестоимость", "копия", "резервный", "база", "данные", "заказ", "склад", "документ", "счёт", "счет",
    "оплата", "сотрудник", "клиент", "менеджер", "налоговый", "цена", "стоимость", "права", "доступ",
}
PROF_MARKERS = {
    "закон", "законно", "законный", "тк", "кодекс", "юридический", "юрист", "штраф", "пеня", "неустойка",
    "суд", "иск", "исковый", "претензия", "правомерно", "оспорить", "налоговик", "трудовой", "декретный",
    "больничный", "патент", "увольнение", "уволить", "налог", "усн", "осно",
}
PROGRAM_MARKERS = {"программа", "1с", "полиграфия", "унф", "база", "документ", "справочник", "конфигурация"}
THIRD_PARTY = {  # стороннее ПО, сервисы, устройства
    "word", "ворд", "excel", "эксель", "powerpoint", "google", "гугл", "gmail", "яндекс", "zoom", "зум",
    "teams", "skype", "telegram", "телеграм", "whatsapp", "youtube", "ютуб", "anydesk", "photoshop", "фотошоп",
    "illustrator", "indesign", "figma", "canva", "android", "андроид", "iphone", "айфон", "ios", "wi", "wifi",
    "роутер", "tp", "vpn", "python", "госуслуга", "ноутбук", "браузер", "телефон", "смартфон",
}
INTEGRATIONS = {  # интеграции и платформы, о которых есть ответы в базе знаний — R4 не применяется
    "битрикс24", "битрикс", "манго", "касперский", "kaspersky", "sql", "mssql", "postgresql", "rdp",
    "remoteapp", "teamviewer", "mac", "macos", "os", "слк", "сервер",
}
STOP = {"как", "где", "что", "почему", "можно", "ли", "в", "на", "и", "а", "у", "с", "по", "для", "мы",
        "я", "мой", "наш", "это", "не", "нет", "да", "же", "бы", "то", "так", "мне", "нам", "вы", "ваш"}


def lemmas(text: str) -> list[str]:
    return [_lemma(t) for t in tokens(text)]


def features(text: str) -> dict:
    lem = lemmas(text)
    s = set(lem) | set(tokens(text))
    content = [l for l in lem if l not in STOP]
    return {
        "domain": bool(s & DOMAIN_TERMS),
        "ambig": bool(s & AMBIGUOUS_OBJECTS),
        "prof": bool(s & PROF_MARKERS),
        "program": bool(s & PROGRAM_MARKERS),
        "third_party": bool(s & THIRD_PARTY),
        "integration": bool(s & INTEGRATIONS),
        "n_content": len(content),
    }


def apply_rules(text: str, label: str) -> tuple[str, str | None]:
    f = features(text)
    prof_only = f["prof"] and not f["program"]
    if label == "out_of_domain":
        if f["domain"] and not prof_only:
            return "unclear", "R1_domain_term"
        if f["n_content"] <= 4 and f["ambig"] and not f["prof"]:
            return "unclear", "R2_short_ambig"
    elif prof_only and not f["domain"]:
        return "out_of_domain", "R3_prof_consult"
    elif (label == "unclear" and f["third_party"] and not f["domain"] and not f["program"]
          and not f["integration"]):
        return "out_of_domain", "R4_third_party"
    return label, None
