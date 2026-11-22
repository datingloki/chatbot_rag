"""Маскирование персональных и коммерческих данных (C08, design doc 1.2 и 7.2).

Сообщение маскируется один раз — на входе в `Bot.respond`, до классификатора, поиска,
LLM, журнала и трасс. Поэтому исходные значения не попадают ни к провайдеру LLM, ни в
SQLite, ни в результаты харнесса. Для ответа они боту не нужны: база знаний общая, а
диагностика конкретной базы клиента — это передача консультанту.

Что маскируется (метки — как в выгрузке OmniDesk, omnidesk-export/prepare_rag_candidates.py):

- [EMAIL], [PHONE] (10–15 цифр), [IP_ADDRESS] (только после «IP»/«адрес» — иначе это релиз 1С);
- [INN] — 10/12 цифр после «ИНН»; [OGRN], [KPP], [ACCOUNT] (20 цифр: р/с, к/с), [CARD] (16 цифр);
- [CONTRACT] — номер после «договор/счёт/заказ … №»;
- [ФИО] — 2–3 слова с заглавной буквы, которые морфология опознаёт как имя/фамилию/отчество
  (одиночное «Мария» не маскируется — слишком много ложных срабатываний вроде «Вера»), и
  фамилия с инициалами («Иванов И.И.», «И. И. Иванов»);
- [LONG_NUMBER] — прочие числа от 10 цифр; [REDACTED] — пароли и токены.

Регулярки консервативны: адреса, названия организаций и одиночные имена не ловятся.
Номера релизов (3.0.12.34), даты и суммы не трогаются.
"""
from __future__ import annotations

import re
from collections import Counter
from functools import lru_cache

EMAIL = re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.I)
SECRET = re.compile(r"\b(password|passwd|token|api[ _-]?key|secret|парол[ьяеи]?|токен|ключ api)\b"
                    r"(\s*(?:[:=]|is|есть|—|-)?\s*)([^\s,;]{4,})", re.I)
# Только с подписью: «3.0.12.34» без неё — номер релиза 1С, его маскировать нельзя.
IP = re.compile(r"(\b(?:ip|айпи|ip-адрес|адрес)\w*\s*(?:сервера\s*)?:?\s*)((?:\d{1,3}\.){3}\d{1,3})(?![\d.])", re.I)
LABELLED = [
    ("INN", re.compile(r"(\bИНН\s*(?:№|:)?\s*)(\d{12}|\d{10})\b", re.I)),
    ("OGRN", re.compile(r"(\bОГРН(?:ИП)?\s*(?:№|:)?\s*)(\d{13}|\d{15})\b", re.I)),
    ("KPP", re.compile(r"(\bКПП\s*(?:№|:)?\s*)(\d{9})\b", re.I)),
    ("CONTRACT", re.compile(
        r"(\b(?:договор|контракт|сч[её]т|заказ|заявк|обращени)\w*\s*(?:№|N|номер)\s*)([\w/-]*\d[\w/-]*)", re.I)),
]
ACCOUNT = re.compile(r"(?<!\d)\d{20}(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d{4}[ -]?){3}\d{4}(?!\d)")
PHONE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{8,}\d(?![\w.])")
LONG_NUMBER = re.compile(r"(?<!\d)\d{10,}(?!\d)")

CAP = r"[А-ЯЁ][а-яё]+(?:-[А-ЯЁ][а-яё]+)?"
INITIALS = re.compile(rf"\b{CAP}\s+[А-ЯЁ]\.\s?[А-ЯЁ]\.|\b[А-ЯЁ]\.\s?[А-ЯЁ]\.\s?{CAP}")
CAP_RUN = re.compile(rf"\b{CAP}(?:\s+{CAP})+\b")


@lru_cache(maxsize=1)
def _morph():
    import pymorphy3

    return pymorphy3.MorphAnalyzer()


def _name_tags(word: str) -> set[str]:
    tags: set[str] = set()
    for p in _morph().parse(word)[:4]:
        tags |= {"Name", "Surn", "Patr"} & p.tag.grammemes
    return tags


def _is_fio(words: list[str]) -> bool:
    tags = [_name_tags(w) for w in words]
    if not all(tags):
        return False
    union = set().union(*tags)
    return "Patr" in union or ("Name" in union and "Surn" in union)


def _phone(m: re.Match) -> str:
    digits = re.sub(r"\D", "", m.group(0))
    return "[PHONE]" if 10 <= len(digits) <= 15 else m.group(0)


def mask_pii(text: str) -> tuple[str, Counter]:
    """→ (текст с метками, сколько чего замаскировано). Значения нигде не сохраняются."""
    counts: Counter = Counter()

    def sub(pattern: re.Pattern, repl, s: str, kind: str) -> str:
        new, n = pattern.subn(repl, s)
        if n:
            counts[kind] += n
        return new

    text = sub(SECRET, lambda m: f"{m.group(1)}{m.group(2)}[REDACTED]", text, "secret")
    text = sub(EMAIL, "[EMAIL]", text, "email")
    text = sub(IP, lambda m: f"{m.group(1)}[IP_ADDRESS]", text, "ip_address")
    for kind, pattern in LABELLED:
        text = sub(pattern, lambda m, k=kind: f"{m.group(1)}[{k}]", text, kind.lower())
    text = sub(ACCOUNT, "[ACCOUNT]", text, "account")
    text = sub(CARD, "[CARD]", text, "card")

    before = text
    text = PHONE.sub(_phone, text)
    if text != before:
        counts["phone"] += text.count("[PHONE]") - before.count("[PHONE]")
    text = sub(LONG_NUMBER, "[LONG_NUMBER]", text, "long_number")

    text = sub(INITIALS, "[ФИО]", text, "person")

    def fio(m: re.Match) -> str:
        # окно 3, затем 2 слова: «Менеджер Иванов Иван Иванович» → «Менеджер [ФИО]»
        words, out, i = m.group(0).split(), [], 0
        while i < len(words):
            for n in (3, 2):
                if i + n <= len(words) and _is_fio(words[i:i + n]):
                    out.append("[ФИО]")
                    counts["person"] += 1
                    i += n
                    break
            else:
                out.append(words[i])
                i += 1
        return " ".join(out)

    text = CAP_RUN.sub(fio, text)
    return text, counts
