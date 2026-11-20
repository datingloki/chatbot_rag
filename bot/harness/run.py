"""Харнесс: прогон системы на наборе кейсов через один интерфейс (КТ3, раздел 5 design doc).

    .venv/bin/python -m harness.run --system s0  --set dev
    .venv/bin/python -m harness.run --system rag --set dev --repeat 3 --concurrency 4
    .venv/bin/python -m harness.run --system rag --set dev --limit 20 --classes K2,K5   # быстрая проверка

Результат — папка harness/results/run_<система>_<набор>_<время>/:
- meta.json    — версии всего, от чего зависит результат: git (и есть ли незакоммиченные правки),
                 хеш набора, модель и её параметры, версия промпта, индекса, классификатора,
                 параметры поиска и лимиты; время и стоимость прогона;
- cases.jsonl  — по строке на кейс и повтор: ожидание, действие системы, текст, источники,
                 задержка, токены, стоимость;
- traces.jsonl — шаги графа каждого хода (для RAG): по ним разбираются провалы;
- summary.json — M01a (автоматическая часть M01) и метрики маршрута по определениям
                 `s0_baseline/evaluate.py` (M01r, M17, M04r, M05r, M06r, M02r), задержка P50/P95,
                 стоимость; при --repeat — разброс.

M01 (design doc 7.2) = верный маршрут и все обязательные проверки. Код считает M01a: маршрут верен
и, если система ответила, источник совпал с эталоном; ответы, где источник проверить кодом нельзя,
идут в `M01a_needs_review`. Проверки текста (M03, M07, M08, M10) делают человек и LLM по
cases.jsonl; итоговый M01 = M01a минус кейсы, проваленные на них. У S0 ответы — утверждённые
консультантом тексты FAQ, поэтому для него M01 = M01a (+ needs_review после ручной проверки).

Eval закрыт до КТ4: `--set eval` работает только с переменной HARNESS_ALLOW_EVAL=1.
`--set canary` — 60 кейсов dev из harness/canary_v1.json для проверки перед релизом (harness/canary.py).
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import statistics
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from harness.retrieval_eval import excerpt_sections, gold, words
from harness.systems import RagSystem, S0System, System, SystemReply
from polybot.settings import BOT_DIR, REPO_DIR

SETS = {"dev": REPO_DIR / "dev set" / "dev_300_balanced.jsonl",
        "eval": REPO_DIR / "eval set" / "eval_300_balanced.jsonl"}
RESULTS = BOT_DIR / "harness" / "results"
DOMAIN = {"K2", "K3", "K4", "K5", "K6"}


# ---------------------------------------------------------------- статистика
def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def frac(k: int, n: int) -> dict[str, Any]:
    return {"n": k, "N": n, "share": round(k / n, 3) if n else None, "wilson95": wilson(k, n)}


def quantile(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    if len(xs) == 1:
        return xs[0]
    return statistics.quantiles(xs, n=100, method="inclusive")[int(q * 100) - 1]


def git_state() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True, cwd=REPO_DIR).stdout.strip()

    try:
        return {"sha": run("rev-parse", "--short", "HEAD"), "branch": run("rev-parse", "--abbrev-ref", "HEAD"),
                "dirty": bool(run("status", "--porcelain", "--", "bot", "domain_classifier", "s0_baseline"))}
    except Exception:  # noqa: BLE001
        return {"sha": "unknown"}


# ---------------------------------------------------------------- проверка источника
class SourceChecker:
    """Совпадает ли источник ответа с эталоном кейса: документ FAQ/OmniDesk из `evidence` или
    раздел руководства, где есть эталонный фрагмент (автоматическая часть M02)."""

    def __init__(self) -> None:
        self._manual = None

    def _manual_chunks(self):
        if self._manual is None:
            from ingest.chunking import chunk_document
            from ingest.documents import load_manual

            self._manual = [(c.doc_id, words(c.text)) for d in load_manual() for c in chunk_document(d)]
        return self._manual

    def gold_docs(self, case: dict[str, Any], exclude: str | None) -> set[str]:
        docs, excerpts = gold(case)
        if exclude:
            docs.discard(f"omnidesk:{exclude}")
        if excerpts:
            docs |= excerpt_sections(excerpts, self._manual_chunks())
        return docs

    def verdict(self, reply: SystemReply, gold_docs: set[str]) -> str | None:
        if reply.action != "answer":
            return None
        if not gold_docs:
            return "needs_review"           # эталонного источника нет — проверяет человек
        cited = {s.get("doc_id", "") for s in reply.sources}
        if cited & gold_docs:
            return "source_ok"
        # Источник другого типа (FAQ при эталоне из руководства) может быть верным ответом —
        # засчитываем только после ручной проверки, как needs_review в s0_baseline/evaluate.py.
        kind = lambda d: d.split(":", 1)[0]  # noqa: E731
        if not {kind(d) for d in cited} & {kind(d) for d in gold_docs}:
            return "needs_review"
        return "source_mismatch"


# ---------------------------------------------------------------- метрики
def m01a(r: dict[str, Any]) -> str:
    """Автоматическая часть M01 по кейсу: ok / needs_review (ответ, источник не проверить кодом) / fail."""
    if not r["route_ok"]:
        return "fail"
    if r["action"] != "answer" or r["source_verdict"] == "source_ok":
        return "ok"
    return "needs_review" if r["source_verdict"] in (None, "needs_review") else "fail"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    dom = [r for r in rows if r["class"] in DOMAIN]
    k1 = [r for r in rows if r["class"] == "K1"]
    k3 = [r for r in rows if r["class"] == "K3"]
    k46 = [r for r in rows if r["class"] in {"K4", "K6"}]
    ans = [r for r in rows if r["action"] == "answer"]
    exp_ans = [r for r in rows if r["expected_action"] == "answer"]
    verdicts = Counter(r["source_verdict"] for r in ans)
    by_class = defaultdict(Counter)
    for r in rows:
        by_class[r["class"]][r["action"]] += 1
    confusion = defaultdict(Counter)
    for r in rows:
        confusion[r["expected_action"]][r["action"]] += 1
    lat = [r["latency_ms"] for r in rows]
    cost = sum(r["llm"].get("cost_rub", 0) for r in rows)
    return {
        "n_cases": len(rows),
        "M01a_route_and_source": frac(sum(m01a(r) == "ok" for r in rows), len(rows)),
        "M01a_needs_review": sum(m01a(r) == "needs_review" for r in rows),
        "M01a_by_class": {k: frac(sum(m01a(r) == "ok" for r in rows if r["class"] == k),
                                  sum(r["class"] == k for r in rows)) for k in sorted(by_class)},
        "M01r_route_ok_all": frac(sum(r["route_ok"] for r in rows), len(rows)),
        "route_ok_by_class": {k: frac(sum(r["route_ok"] for r in rows if r["class"] == k),
                                      sum(r["class"] == k for r in rows)) for k in sorted(by_class)},
        "actions_by_class": {k: dict(v) for k, v in sorted(by_class.items())},
        "confusion_expected_vs_action": {k: dict(v) for k, v in sorted(confusion.items())},
        "M17_handoff_offered_domain": frac(sum(r["action"] == "offer_handoff" for r in dom), len(dom)),
        "M17_by_class": {k: frac(sum(r["action"] == "offer_handoff" for r in rows if r["class"] == k),
                                 sum(r["class"] == k for r in rows)) for k in sorted(DOMAIN)},
        "M17_without_verbatim_faq_titles": frac(
            sum(r["action"] == "offer_handoff" for r in dom if r["origin_type"] != "official_faq_entry"),
            sum(r["origin_type"] != "official_faq_entry" for r in dom)),
        "domain_false_refusal": frac(sum(r["action"] == "refuse" for r in dom), len(dom)),
        "M04r_K1_refused": frac(sum(r["action"] == "refuse" for r in k1), len(k1)),
        "M04r_K1_answered": frac(sum(r["action"] == "answer" for r in k1), len(k1)),
        "M05r_K3_clarified": frac(sum(r["action"] == "clarify" for r in k3), len(k3)),
        "M06r_K4_K6_handoff_offered": frac(sum(r["action"] == "offer_handoff" for r in k46), len(k46)),
        "answers": {
            "total": len(ans),
            "answered_when_expected": frac(sum(r["action"] == "answer" for r in exp_ans), len(exp_ans)),
            "answered_non_answer_class": frac(sum(r["expected_action"] != "answer" for r in ans), len(ans)),
            "source_verdicts": dict(verdicts),
            "M02r_source_ok_among_verifiable": frac(verdicts["source_ok"],
                                                    verdicts["source_ok"] + verdicts["source_mismatch"]),
            "with_source": frac(sum(bool(r["sources"]) for r in ans), len(ans)),
        },
        "fallback": frac(sum(r["action"] == "fallback" for r in rows), len(rows)),
        "fallback_reasons": dict(Counter(r["reason"] for r in rows if r["action"] == "fallback")),
        "latency_ms": {"p50": round(quantile(lat, 0.50), 1), "p95": round(quantile(lat, 0.95), 1),
                       "max": round(max(lat), 1) if lat else 0.0},
        "llm": {"calls": sum(r["llm"].get("calls", 0) for r in rows),
                "prompt_tokens": sum(r["llm"].get("prompt_tokens", 0) for r in rows),
                "completion_tokens": sum(r["llm"].get("completion_tokens", 0) for r in rows),
                "cost_rub": round(cost, 3),
                "cost_rub_per_case": round(cost / len(rows), 4) if rows else 0.0},
    }


KEY_METRICS = ["M01a_route_and_source", "M01r_route_ok_all", "M17_handoff_offered_domain", "M04r_K1_refused",
               "M05r_K3_clarified", "M06r_K4_K6_handoff_offered"]


def spread(per_repeat: list[dict[str, Any]]) -> dict[str, Any]:
    out = {}
    for m in KEY_METRICS:
        xs = [s[m]["share"] for s in per_repeat if s[m]["share"] is not None]
        if xs:
            out[m] = {"mean": round(statistics.mean(xs), 3), "min": min(xs), "max": max(xs)}
    xs = [s["answers"]["M02r_source_ok_among_verifiable"]["share"] for s in per_repeat
          if s["answers"]["M02r_source_ok_among_verifiable"]["share"] is not None]
    if xs:
        out["M02r_source_ok_among_verifiable"] = {"mean": round(statistics.mean(xs), 3),
                                                  "min": min(xs), "max": max(xs)}
    return out


# ---------------------------------------------------------------- прогон
def load_cases(path: Path, limit: int | None, classes: set[str] | None) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if classes:
        cases = [c for c in cases if c["primary_class"] in classes]
    return cases[:limit] if limit else cases


async def run_cases(system: System, cases: list[dict[str, Any]], repeat: int, concurrency: int,
                    checker: SourceChecker, on_row) -> None:
    from harness.systems import origin_omnidesk

    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(case: dict[str, Any], rep: int) -> None:
        nonlocal done
        async with sem:
            # Свой case_id на повтор: состояние сессии не переходит из повтора в повтор.
            reply = await system.reply({**case, "case_id": f"{case['case_id']}#r{rep}"})
        own = origin_omnidesk(case)
        gdocs = checker.gold_docs(case, own)
        on_row(case, rep, reply, gdocs, checker.verdict(reply, gdocs))
        done += 1
        if done % 25 == 0 or done == len(cases) * repeat:
            print(f"  {done}/{len(cases) * repeat}")

    await asyncio.gather(*(one(c, r) for r in range(repeat) for c in cases))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", choices=["s0", "rag"], required=True)
    ap.add_argument("--set", choices=[*SETS, "canary"], default="dev")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--classes", help="через запятую: K1,K2,…")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    if args.set == "eval" and os.environ.get("HARNESS_ALLOW_EVAL") != "1":
        raise SystemExit("eval закрыт до КТ4 (design doc 5.3). Для прогона на КТ4: HARNESS_ALLOW_EVAL=1")
    path = SETS["dev" if args.set == "canary" else args.set]
    cases = load_cases(path, args.limit, set(args.classes.split(",")) if args.classes else None)
    if args.set == "canary":                     # 60 кейсов dev из harness/canary_v1.json (7.3)
        from harness.canary import canary_cases, load_ids

        cases = canary_cases(cases, load_ids())
    system: System = S0System() if args.system == "s0" else RagSystem()
    concurrency = 1 if args.system == "s0" else args.concurrency

    run_id = f"run_{args.system}_{args.set}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = RESULTS / run_id
    out.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "system": args.system, "set": args.set,
            "set_file": str(path.relative_to(REPO_DIR)),
            "set_sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16],
            "n_cases": len(cases), "repeat": args.repeat, "concurrency": concurrency,
            "filters": {"limit": args.limit, "classes": args.classes}, "note": args.note,
            "git": git_state(), "system_info": system.info(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{run_id}: {len(cases)} кейсов × {args.repeat}, система {args.system}")

    rows: list[dict[str, Any]] = []
    cases_f = (out / "cases.jsonl").open("w", encoding="utf-8")
    traces_f = (out / "traces.jsonl").open("w", encoding="utf-8")

    def on_row(case, rep, reply: SystemReply, gdocs, verdict) -> None:
        row = {"case_id": case["case_id"], "repeat": rep, "class": case["primary_class"],
               "expected_action": case["expected_action"], "origin_type": case["origin_type"],
               "message": case["input"]["user_message"], "n_history": len(case["input"].get("history", [])),
               "action": reply.action, "route_ok": reply.action == case["expected_action"],
               "reason": reply.reason, "text": reply.text, "sources": reply.sources,
               "gold_docs": sorted(gdocs), "source_verdict": verdict, "latency_ms": reply.latency_ms,
               "llm": reply.llm, "path": reply.path, "trace_id": reply.trace_id, "extra": reply.extra}
        rows.append(row)
        cases_f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        if reply.steps:
            traces_f.write(json.dumps({"case_id": case["case_id"], "repeat": rep, "trace_id": reply.trace_id,
                                       "steps": reply.steps}, ensure_ascii=False, default=str) + "\n")

    t0 = time.time()
    try:
        asyncio.run(run_cases(system, cases, args.repeat, concurrency, SourceChecker(), on_row))
    finally:
        cases_f.close()
        traces_f.close()

    per_repeat = [summarize([r for r in rows if r["repeat"] == rep]) for rep in range(args.repeat)]
    summary = {"all_repeats": summarize(rows), "per_repeat": per_repeat if args.repeat > 1 else None,
               "spread": spread(per_repeat) if args.repeat > 1 else None}
    meta.update(finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), duration_s=round(time.time() - t0, 1),
                cost_rub=summary["all_repeats"]["llm"]["cost_rub"])
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summary["all_repeats"], meta, summary["spread"])
    print(f"\nРезультаты: {out}")


def _f(m: dict[str, Any]) -> str:
    return f"{m['n']}/{m['N']} ({100 * m['share']:.0f}%)" if m["N"] else "—"


def print_summary(s: dict[str, Any], meta: dict[str, Any], sp: dict[str, Any] | None) -> None:
    print(f"\n{meta['system']} на {meta['set']}: {s['n_cases']} ходов за {meta['duration_s']} с, "
          f"стоимость LLM {s['llm']['cost_rub']} ₽")
    print(f"  M01a маршрут и источник           {_f(s['M01a_route_and_source'])}"
          f"  + {s['M01a_needs_review']} на ручную проверку")
    print(f"  M01r маршрут верен                {_f(s['M01r_route_ok_all'])}")
    print(f"  M17 передача оператору (домен)    {_f(s['M17_handoff_offered_domain'])}")
    print(f"  M04r K1 отказ                     {_f(s['M04r_K1_refused'])}")
    print(f"  M05r K3 уточнение                 {_f(s['M05r_K3_clarified'])}")
    print(f"  M06r K4/K6 передача               {_f(s['M06r_K4_K6_handoff_offered'])}")
    print(f"  ответ, когда ожидался             {_f(s['answers']['answered_when_expected'])}")
    print(f"  M02r источник совпал с эталоном   {_f(s['answers']['M02r_source_ok_among_verifiable'])}")
    print(f"  ложный отказ в домене             {_f(s['domain_false_refusal'])}")
    print(f"  fallback                          {_f(s['fallback'])}  {s['fallback_reasons'] or ''}")
    print(f"  задержка P50/P95                  {s['latency_ms']['p50']:.0f} / {s['latency_ms']['p95']:.0f} мс")
    print("  маршрут по классам: " + ", ".join(f"{k} {_f(v)}" for k, v in s["route_ok_by_class"].items()))
    if sp:
        print("  разброс по повторам: " + ", ".join(f"{k} {v['min']}–{v['max']}" for k, v in sp.items()))


if __name__ == "__main__":
    main()
