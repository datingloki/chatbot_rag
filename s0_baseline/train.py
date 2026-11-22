"""Сборка S0 ТОЛЬКО на dev и базе знаний. eval_300_balanced.jsonl здесь не читается.

1. Интенты и их примеры — из версионированного файла intents/intents_v1.jsonl
   (build_intents.py): заголовок, заголовок + начало ответа, перефразы FAQ из dev.
   Векторы примеров считаются здесь и сохраняются в model/s0.joblib вместе с версией
   и sha256 файла интентов.
2. Для каждого доменного кейса dev считаем топ-1 интент, исключая примеры из его же
   группы (split_group_id) — перефразы одного диалога не попадают в «обучение» и «проверку»
   одновременно.
3. Калибровка: логистическая регрессия P(топ-1 интент верный | сходство, отрыв).
   Вероятности для подбора порога — out-of-fold (GroupKFold по split_group_id).
4. Порог: минимальная вероятность, при которой S0 отвечает шаблоном не более чем на
   MAX_FALSE_ANSWER доменных dev-вопросов без подходящего FAQ и не даёт неверного FAQ
   на dev-вопросах с известным FAQ. Отказы K1 не участвуют (их делает классификатор домена).

Запуск:  .venv/bin/python train.py
"""
from __future__ import annotations

import json
import warnings
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from data import INTENTS_PATH, ROOT, load_dev, load_intents
from intent_matcher import IntentMatcher, conf_features
from s0 import MODEL_PATH, client_specific

warnings.filterwarnings("ignore")

CONFIG = {
    "weights": {"char": 1.0, "lemma": 1.0, "emb": 1.0},
    "emb_model": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
}
MAX_FALSE_ANSWER = 0.03     # доля доменных вопросов без FAQ, на которые можно ответить шаблоном
MAX_WRONG_INTENT = 0        # неверных FAQ на вопросах с известным FAQ


def user_text(c: dict) -> str:
    return c["message"]


def main():
    intents, intents_sha = load_intents(INTENTS_PATH)
    key = {i["intent_id"]: i["answer_key"] for i in intents}
    dev = load_dev()
    dom = [c for c in dev if c["primary_class"] != "K1"]
    m = IntentMatcher(intents, CONFIG["weights"], CONFIG["emb_model"])

    # примеры той же группы, что и кейс (перефразы одного диалога), при подборе порога скрыты
    texts = [user_text(c) for c in dom]
    mask = np.array([[g is not None and g == c["group"] for g in m.proto_group] for c in dom], dtype=bool)
    tops = m.top(m.intent_scores(texts, mask=mask))
    X = conf_features(tops)
    gold = [key.get(c["faq_id"]) if c["faq_id"] else None for c in dom]
    y = np.array([g is not None and key[t["intent_id"]] == g for t, g in zip(tops, gold)], dtype=int)
    groups = [c["group"] for c in dom]

    oof = np.zeros(len(dom))
    for tr, te in GroupKFold(n_splits=5).split(X, y, groups):
        oof[te] = LogisticRegression(class_weight="balanced").fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    calib = LogisticRegression(class_weight="balanced").fit(X, y)

    cs = np.array([client_specific(c["message"] + "\n" + "\n".join(
        h["text"] for h in c["history"] if h.get("role") == "user")) is not None for c in dom])
    has_faq = np.array([g is not None for g in gold])
    neg = ~has_faq

    def outcome(t):
        ans = (oof >= t) & ~cs
        return {
            "threshold": round(float(t), 4),
            "false_answer_no_faq": int((ans & neg).sum()), "n_no_faq": int(neg.sum()),
            "wrong_intent_faq": int((ans & has_faq & (y == 0)).sum()),
            "correct_faq_answered": int((ans & has_faq & (y == 1)).sum()), "n_faq": int(has_faq.sum()),
            "faq_paraphrase_answered": int((ans & (y == 1) & np.array(
                [c["origin_type"] != "official_faq_entry" for c in dom])).sum()),
        }

    cands = np.unique(np.round(oof, 4))
    ok = [t for t in cands if outcome(t)["false_answer_no_faq"] <= MAX_FALSE_ANSWER * neg.sum()
          and outcome(t)["wrong_intent_faq"] <= MAX_WRONG_INTENT]
    threshold = float(min(ok)) if ok else 1.0
    curve = [outcome(t) for t in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, threshold)]

    # разбор: какие dev-кейсы без FAQ получили бы ответ и что блокирует клиент-правило
    answered_no_faq = [{"case_id": c["case_id"], "class": c["primary_class"], "message": c["message"][:160],
                        "intent": t["intent_id"], "p": round(float(p), 3)}
                       for c, t, p, n, s in zip(dom, tops, oof, neg, cs) if n and p >= threshold and not s]
    cs_hits = Counter((c["primary_class"], bool(c["faq_id"])) for c, s in zip(dom, cs) if s)
    cs_blocked_faq = [c["case_id"] for c, s, yy in zip(dom, cs, y) if s and c["faq_id"]]

    report = {
        "config": CONFIG, "max_false_answer": MAX_FALSE_ANSWER, "threshold": threshold,
        "calibrator": {"coef": calib.coef_.round(3).tolist(), "intercept": calib.intercept_.round(3).tolist(),
                       "features": ["score", "margin"]},
        "dev_domain_cases": len(dom), "dev_faq_cases": int(has_faq.sum()),
        "top1_accuracy_faq_cases": round(float(y[has_faq].mean()), 3),
        "top1_accuracy_faq_paraphrases": round(float(np.mean([yy for c, yy in zip(dom, y)
                                                              if c["faq_id"] and c["origin_type"] != "official_faq_entry"])), 3),
        "threshold_curve_oof": curve,
        "answered_without_faq_at_threshold": answered_no_faq,
        "client_specific_hits": {f"{k[0]}|faq={k[1]}": v for k, v in sorted(cs_hits.items())},
        "client_specific_blocked_faq_cases": cs_blocked_faq,
        "intents_file": str(INTENTS_PATH.relative_to(ROOT)), "intents_sha256": intents_sha,
        "intents_status": dict(Counter(i["status"] for i in intents)),
        "n_intents": len(intents), "n_answer_keys": len(set(key.values())),
        "n_examples": len(m.proto_text), "examples_by_kind": dict(Counter(m.proto_kind)),
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    (MODEL_PATH.parent / "train_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    joblib.dump({
        "config": CONFIG, "matcher": m, "calibrator": calib, "threshold": threshold,
        "intents_file": str(INTENTS_PATH.relative_to(ROOT)), "intents_version": intents[0]["intents_version"],
        "intents_sha256": intents_sha,
    }, MODEL_PATH)
    print(json.dumps({k: v for k, v in report.items() if k != "answered_without_faq_at_threshold"},
                     ensure_ascii=False, indent=1))
    print("answered without FAQ:", json.dumps(answered_no_faq, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
