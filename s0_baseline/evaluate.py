"""Прогон S0 на наборе кейсов и автоматические метрики маршрута.

    .venv/bin/python evaluate.py --set dev     # разработочный набор (S0 на нём настроен — оптимистично)
    .venv/bin/python evaluate.py --set eval    # закрытый набор; запускать на замороженной модели

Что считается кодом (маршрут, без оценки текста человеком/LLM):
  M17  — предложение оператора / все доменные кейсы (K2–K6), n/N, интервал Уилсона.
  M04r — отказ на K1 (маршрут; содержание отказа — фиксированный шаблон).
  M06r — K4/K6 получили предложение оператора; ни одного «заявка создана».
  M02r/M03r — ответы шаблоном: доля с верным FAQ (по эталонному source_faq_id). Ответ на кейс
         без эталонного FAQ помечается `needs_review`: он может быть верным (FAQ подходит), но
         засчитывается только после ручной проверки.
  M05r — K3 получили уточнение (S0 уточнять не умеет — 0 по построению).
M01, M07, M08, M10 требуют ручной проверки; для них выгружается eval_cases.jsonl.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict

import numpy as np

from data import load_cases, DEV, EVAL
from s0 import MODEL_PATH, S0Bot

DOMAIN = {"K2", "K3", "K4", "K5", "K6"}


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def frac(k: int, n: int) -> dict:
    return {"n": k, "N": n, "share": round(k / n, 3) if n else None, "wilson95": wilson(k, n)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", choices=["dev", "eval"], required=True)
    args = ap.parse_args()
    cases = load_cases(DEV if args.set == "dev" else EVAL)
    bot = S0Bot.load()
    key = {iid: i["answer_key"] for iid, i in bot.intents.items()}
    bot.respond("прогрев")  # загрузка моделей вне замера задержки
    rows, lat = [], []
    for c in cases:
        t0 = time.perf_counter()
        r = bot.respond(c["message"], history=c["history"], handoff_consent=bool(c["context"].get("handoff_consent")))
        lat.append(time.perf_counter() - t0)
        gold = key.get(c["faq_id"]) if c["faq_id"] else None
        if r["action"] != "answer":
            verdict = None
        elif gold is None:
            verdict = "needs_review"
        else:
            verdict = "correct_faq" if key[r["intent_id"]] == gold else "wrong_faq"
        rows.append({
            "case_id": c["case_id"], "class": c["primary_class"], "expected_action": c["expected_action"],
            "origin_type": c["origin_type"], "gold_faq": c["faq_id"], "message": c["message"],
            "n_history": len(c["history"]), "action": r["action"], "reason": r["reason"],
            "intent_id": r["intent_id"], "p_intent": r["p_intent"], "domain": r["domain"],
            "client_specific": r["client_specific"], "ticket_created": r["ticket_created"],
            "source_url": r["source_url"], "answer_verdict": verdict, "route_ok": r["action"] == c["expected_action"],
            "latency_s": round(lat[-1], 4), "text": r["text"],
        })

    dom = [r for r in rows if r["class"] in DOMAIN]
    ans = [r for r in rows if r["action"] == "answer"]
    k1 = [r for r in rows if r["class"] == "K1"]
    k46 = [r for r in rows if r["class"] in {"K4", "K6"}]
    k3 = [r for r in rows if r["class"] == "K3"]
    faq_cases = [r for r in rows if r["gold_faq"]]
    verbatim = [r for r in faq_cases if r["origin_type"] == "official_faq_entry"]
    para = [r for r in faq_cases if r["origin_type"] != "official_faq_entry"]

    by_class = defaultdict(Counter)
    for r in rows:
        by_class[r["class"]][r["action"]] += 1
    m17_by_class = {k: frac(sum(r["action"] == "offer_handoff" for r in rows if r["class"] == k),
                            sum(r["class"] == k for r in rows)) for k in sorted(DOMAIN)}
    verdicts = Counter(r["answer_verdict"] for r in ans)

    report = {
        "set": args.set, "n_cases": len(rows), "model": str(MODEL_PATH.name),
        "threshold": bot.threshold, "config": bot.cfg, "intents_version": bot.intents_version,
        "M17_handoff_offered_domain": frac(sum(r["action"] == "offer_handoff" for r in dom), len(dom)),
        "M17_by_class": m17_by_class,
        "M17_by_reason": dict(Counter(r["reason"] for r in dom if r["action"] == "offer_handoff")),
        "M17_without_verbatim_faq_titles": frac(
            sum(r["action"] == "offer_handoff" for r in dom if r["origin_type"] != "official_faq_entry"),
            sum(r["origin_type"] != "official_faq_entry" for r in dom)),
        "domain_answered_by_template": frac(sum(r["action"] == "answer" for r in dom), len(dom)),
        "domain_refused_false_refusal": frac(sum(r["action"] == "refuse" for r in dom), len(dom)),
        "actions_by_class": {k: dict(v) for k, v in sorted(by_class.items())},
        "route_accuracy_by_class": {k: frac(sum(r["route_ok"] for r in rows if r["class"] == k),
                                            sum(r["class"] == k for r in rows)) for k in sorted(by_class)},
        "M01r_route_ok_all": frac(sum(r["route_ok"] for r in rows), len(rows)),
        "answers": {
            "total": len(ans), "verdicts": dict(verdicts),
            "M02r_M03r_correct_faq_among_verifiable": frac(verdicts["correct_faq"],
                                                          verdicts["correct_faq"] + verdicts["wrong_faq"]),
            "answered_non_answer_class": frac(sum(r["expected_action"] != "answer" for r in ans), len(ans)),
            "with_source": frac(sum(bool(r["source_url"]) for r in ans), len(ans)),
        },
        "faq_coverage": {
            "faq_cases_answered_correctly": frac(sum(r["answer_verdict"] == "correct_faq" for r in faq_cases), len(faq_cases)),
            "verbatim_titles_answered_correctly": frac(sum(r["answer_verdict"] == "correct_faq" for r in verbatim), len(verbatim)),
            "paraphrases_answered_correctly": frac(sum(r["answer_verdict"] == "correct_faq" for r in para), len(para)),
        },
        "M04r_K1_refused": frac(sum(r["action"] == "refuse" for r in k1), len(k1)),
        "M04r_K1_answered_by_template": frac(sum(r["action"] == "answer" for r in k1), len(k1)),
        "M05r_K3_clarified": frac(sum(r["action"] == "clarify" for r in k3), len(k3)),
        "M06r_K4_K6_handoff_offered": frac(sum(r["action"] == "offer_handoff" for r in k46), len(k46)),
        "M06r_ticket_claimed": sum(r["ticket_created"] for r in rows),
        "latency_s": {"p50": round(float(np.percentile(lat, 50)), 4), "p95": round(float(np.percentile(lat, 95)), 4),
                      "max": round(float(np.max(lat)), 4), "hardware": "local CPU (dev laptop), not pilot hardware"},
        "cost_api_rub_per_request": 0,
    }
    out_dir = MODEL_PATH.parent.parent / "results"
    out_dir.mkdir(exist_ok=True)
    (out_dir / f"{args.set}_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    with (out_dir / f"{args.set}_cases.jsonl").open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
