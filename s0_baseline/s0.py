"""S0 — бейзлайн-бот на интентах (design doc, 6.1).

Порядок работы:
  1. Проверка входа и явного выхода за домен (формат входа; классификатор домена
     domain_classifier: только явный `out_of_domain` - отказ; `unclear` идёт дальше).
  2. Классификация доменного вопроса по утверждённому списку интентов (= все FAQ).
  3. Уверенный интент -> проверенный шаблон ответа с источником.
  4. Низкая уверенность, нет шаблона или клиент-специфичная проблема → предложение
     оператора, без выдуманного ответа.
  5. Передача (заявка) — только после согласия пользователя И подтверждённой интеграции.
     Интеграция с OmniDesk не подтверждена, поэтому S0 заявку не создаёт и не
     сообщает о ней.

Генеративной LLM нет. Свободный текст бот не составляет: только шаблоны ниже.

Использование:
    .venv/bin/python s0.py "Какой срок действия лицензии на программу?"
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import joblib
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "domain_classifier"))
sys.path.insert(0, str(HERE))

from data import ROOT, load_intents  # noqa: E402
from domain_clf import DomainClassifier, _lemma, tokens  # noqa: E402
from intent_matcher import conf_features  # noqa: E402

MODEL_PATH = HERE / "model" / "s0.joblib"
MAX_INPUT_CHARS = 2000
INTEGRATION_CONFIRMED = False   # OmniDesk: доступ не подтверждён (design doc, 3.5 / M06)

# ---------------------------------------------------------------- клиент-специфичные проблемы
# Признаки того, что ответ требует работы с базой/данными/учётной записью конкретного клиента
# (K4/K6). Подобраны по dev. Применяются только как блокировка ответа шаблоном → оператор.
CLIENT_LEMMAS = {
    "доработка", "доработать", "доработанный", "дорабатывать", "расширение", "филиал",
    "инн", "кпп", "огрн",
}
CLIENT_PHRASES = [
    r"\b(наш|нашей|нашу|рабоч\w+|клиентск\w+|своей|моей)\s+баз",
    r"\bкопи\w*\s+баз",
    r"\bв\s+баз\w*\s+клиента",
    r"\bжурнал\w*\s+регистрац",
    r"\bтекст\w*\s+ошибк\w*\s+не\s+сохран",
    r"\bпосле\s+(переноса|обмена|загрузки)\s+данных",
    r"\bкод\s+партн[её]р",
    r"\bрег(истрационн\w*)?\.?\s*номер",
    r"\bномер\s+(заявк|договор|сч[её]т)",
    r"\bмы\s+(звонили|писали|оставляли|отправляли)",
    r"\bоставлял\w*\s+заявк",
    r"\bвышлите|\bвыслать|\bпришлите|\bвыставьте|\bвыставить\s+сч[её]т",
    r"\[(инн|ip_address|long_number|phone|email|fio|name)\]",
]


def client_specific(text: str) -> str | None:
    low = text.lower().replace("ё", "е")
    lem = {_lemma(t) for t in tokens(text)}
    hit = lem & CLIENT_LEMMAS
    if hit:
        return f"lemma:{sorted(hit)[0]}"
    for p in CLIENT_PHRASES:
        if re.search(p, low):
            return f"phrase:{p}"
    return None


# ---------------------------------------------------------------- шаблоны текста
REFUSE_TEXT = ("Я отвечаю по 1С:Полиграфии и связанным с ней вопросам по КП/ИТС, лицензированию "
               "и поддержке на основе официальных материалов. С этим вопросом помочь не смогу.")
REJECT_EMPTY = "Пожалуйста, напишите вопрос текстом."
REJECT_LONG = (f"Сообщение длиннее {MAX_INPUT_CHARS} символов. Пожалуйста, сократите его и опишите "
               "главный вопрос.")
HANDOFF_TEXT = {
    "low_confidence": "В базе готовых ответов нет проверенного ответа на этот вопрос.",
    "no_template": "В базе готовых ответов нет проверенного ответа на этот вопрос.",
    "client_specific": ("Вопрос касается вашей конкретной базы или учётной записи, поэтому готовой "
                        "инструкции здесь нет."),
}
# Интеграция не подтверждена (design doc 2.2): предлагаем консультанта, но не обещаем заявку.
HANDOFF_TAIL = "По этому вопросу нужен консультант. Автоматическая передача обращения пока не подтверждена."
HANDOFF_TAIL_INTEGRATED = "Могу передать вопрос консультанту. Если согласны, ответьте «да»."
HANDOFF_DONE = "Обращение передано консультанту."


def answer_text(intent: dict) -> str:
    src = intent["source_title"] or "FAQ 1cpoly.ru"
    return (f"По вашему вопросу есть ответ в FAQ «{intent['title']}».\n\n{intent['answer_template']}\n\n"
            f"Источник: {src}, {intent['source_url']}\n"
            "Если это не решает вашу задачу, напишите — предложу передать вопрос консультанту.")


# ---------------------------------------------------------------- бот
class S0Bot:
    def __init__(self, bundle: dict, domain: DomainClassifier | None = None):
        self.cfg = bundle["config"]
        # Интенты и шаблоны — только из версионированного файла, с которым обучена модель.
        path = ROOT / bundle["intents_file"]
        intents, sha = load_intents(path)
        if sha != bundle["intents_sha256"]:
            raise RuntimeError(
                f"{bundle['intents_file']} изменён после обучения (sha256 {sha[:12]} ≠ "
                f"{bundle['intents_sha256'][:12]}). Порог подобран для другой версии интентов: "
                "перезапустите train.py или выпустите новую версию файла.")
        self.intents_version = bundle["intents_version"]
        self.intents = {i["intent_id"]: i for i in intents}
        self.matcher = bundle["matcher"]          # векторы примеров уже посчитаны в train.py
        self.calib = bundle["calibrator"]
        self.threshold = bundle["threshold"]
        self.domain = domain or DomainClassifier.load()

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "S0Bot":
        return cls(joblib.load(path))

    def intent(self, texts: list[str]) -> list[dict]:
        tops = self.matcher.top(self.matcher.intent_scores(texts))
        probs = self.calib.predict_proba(conf_features(tops))[:, 1]
        for t, p in zip(tops, probs):
            t["p_intent"] = float(p)
        return tops

    def respond(self, user_message: str, history: list[dict] | None = None,
                handoff_consent: bool = False) -> dict:
        history = history or []
        msg = (user_message or "").strip()
        out = {"action": None, "reason": None, "intent_id": None, "p_intent": None,
               "domain": None, "client_specific": None, "ticket_created": False, "source_url": None}

        # (1) вход и явный выход за домен
        if not msg:
            return {**out, "action": "reject_input", "reason": "empty", "text": REJECT_EMPTY}
        if len(msg) > MAX_INPUT_CHARS:
            return {**out, "action": "reject_input", "reason": "too_long", "text": REJECT_LONG}
        dom = self.domain.predict(msg, history=history)
        out["domain"] = dom["label"]
        if dom["label"] == "out_of_domain":
            return {**out, "action": "refuse", "reason": "out_of_domain", "text": REFUSE_TEXT}

        # (2) интент: по текущему сообщению; если не уверен и есть история — вместе с ней
        top = self.intent([msg])[0]
        prev = [h["text"] for h in history if h.get("role") == "user"]
        if top["p_intent"] < self.threshold and prev:
            ctx_top = self.intent(["\n".join(prev + [msg])])[0]
            if ctx_top["p_intent"] > top["p_intent"]:
                top = {**ctx_top, "used_history": True}
        out.update(intent_id=top["intent_id"], p_intent=round(top["p_intent"], 4),
                   intent_score=round(top["score"], 4), intent_margin=round(top["margin"], 4))

        # (4) клиент-специфичная проблема — шаблон не показываем
        cs = client_specific("\n".join(prev + [msg]))
        out["client_specific"] = cs
        reason = None
        if cs:
            reason = "client_specific"
        elif top["p_intent"] < self.threshold:
            reason = "low_confidence"
        elif not self.intents[top["intent_id"]].get("answer_template", "").strip():
            reason = "no_template"

        # (3) уверенный интент → шаблон с источником
        if reason is None:
            intent = self.intents[top["intent_id"]]
            return {**out, "action": "answer", "reason": "confident_intent",
                    "intents_version": self.intents_version, "template_status": intent["status"],
                    "source_url": intent["source_url"], "text": answer_text(intent)}

        # (4)/(5) предложение оператора; заявку создаём только при согласии И интеграции
        ticket = bool(handoff_consent and INTEGRATION_CONFIRMED)
        if ticket:
            # здесь будет вызов адаптера OmniDesk; статус «создано» — только по его подтверждению
            tail = HANDOFF_DONE
        else:
            tail = HANDOFF_TAIL_INTEGRATED if INTEGRATION_CONFIRMED else HANDOFF_TAIL
        return {**out, "action": "offer_handoff", "reason": reason, "ticket_created": ticket,
                "text": HANDOFF_TEXT[reason] + " " + tail}


if __name__ == "__main__":
    bot = S0Bot.load()
    for text in sys.argv[1:] or [line.strip() for line in sys.stdin if line.strip()]:
        r = bot.respond(text)
        print(json.dumps({"input": text, **{k: v for k, v in r.items() if k != "text"}}, ensure_ascii=False))
        print(r["text"], "\n")
