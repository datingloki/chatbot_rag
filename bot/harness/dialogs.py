"""Полные диалоги: от первого неполного сообщения до ответа или предложения консультанта.

Замечание ревью КТ2: в кейсах «после уточнения» (K5, K6) история с вопросом бота написана
заранее — так проверяется решение на готовом контексте, но не то, задаст ли бот сам полезный
вопрос и вовремя ли остановится. Поэтому у тех же кейсов dev/eval есть поле `dialog`, и бот
проходит разговор сам: на его уточнения отвечает симулятор пользователя.

    .venv/bin/python -m harness.dialogs build                 # отчёт: у каких кейсов будет dialog
    .venv/bin/python -m harness.dialogs build --write         # записать поле dialog в dev
    .venv/bin/python -m harness.dialogs run                   # прогон dev, пользователь scripted
    .venv/bin/python -m harness.dialogs run --user llm        # пользователь — LLM по фактам кейса
    .venv/bin/python -m harness.dialogs run --classes K6 --limit 10

eval и all_cases (в нём есть кейсы eval) — только с HARNESS_ALLOW_EVAL=1.

Поле `dialog` выводится из уже размеченных полей кейса, ничего не придумывается:

- `first_message` — первая реплика пользователя в `input.history`;
- `user_replies` — ответы пользователя на уточнения по порядку (следующие реплики пользователя
  в истории и `input.user_message`): то, что пользователь знает и сообщит, если спросят;
- `reference_questions` — уточнения бота из размеченной истории: эталон для ручной оценки
  полезности вопросов, боту не показываются;
- `expected_outcome` — `expected_action` кейса (`answer` для K5, `offer_handoff` для K6);
- `max_clarifications` — `clarification_limit` (5, C04).

K3 («нужно уточнение») полного диалога не получает: у этих кейсов нет ответа пользователя на
уточнение и нет итога, ожидается только сам вопрос — они остаются однократной проверкой.

Симулятор пользователя:
- `scripted` — на каждое уточнение отдаёт следующий ответ из `user_replies`, когда ответы
  кончились — «Больше уточнить не могу». Воспроизводимо и бесплатно; именно так ответил
  пользователь в размеченном кейсе.
- `llm` — модель в роли пользователя отвечает на конкретный вопрос бота, используя только
  `user_replies`; о том, чего там нет, говорит «не знаю». Реалистичнее: видно, понял ли бот
  ответ не на «свой» вопрос и остановился ли, когда пользователь ничего не знает.

Результат — harness/results/dialogs_<набор>_<пользователь>_<время>/: meta.json, cases.jsonl
(все ходы, вопросы бота и эталонные вопросы — для ручной проверки полезности), summary.json.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from harness.normalize_evidence import COMPACT, line_format
from harness.run import RESULTS, SETS, SourceChecker, frac, git_state, quantile
from polybot.schemas import ReplyKind
from polybot.settings import REPO_DIR, Settings

DIALOG_CLASSES = {"K5", "K6"}
NO_MORE = "Больше уточнить не могу."
DIALOG_VERSION = 1


# ---------------------------------------------------------------- поле dialog
def _input(case: dict[str, Any]) -> dict[str, Any]:
    inp = case.get("input") or {}
    return json.loads(inp) if isinstance(inp, str) else inp       # all_cases: JSON-строка


def derive_dialog(case: dict[str, Any]) -> dict[str, Any] | None:
    if case.get("primary_class") not in DIALOG_CLASSES:
        return None
    inp = _input(case)
    turns = list(inp.get("history") or []) + [{"role": "user", "text": inp.get("user_message", "")}]
    if len(turns) < 2 or turns[0].get("role") != "user":
        return None
    return {
        "version": DIALOG_VERSION,
        "first_message": turns[0]["text"],
        "user_replies": [t["text"] for t in turns[1:] if t.get("role") == "user"],
        "reference_questions": [t["text"] for t in turns[1:] if t.get("role") == "assistant"],
        "expected_outcome": case["expected_action"],
        "max_clarifications": int(case.get("clarification_limit") or 5),
        "derived_from": "input.history + input.user_message",
    }


def get_dialog(case: dict[str, Any]) -> dict[str, Any] | None:
    d = case.get("dialog")
    return json.loads(d) if isinstance(d, str) else d


def build(path: Path, write: bool) -> dict[str, Any]:
    lines = [l for l in path.open(encoding="utf-8") if l.strip()]
    sep = line_format(lines)
    rows = [json.loads(l) for l in lines]
    made: Counter = Counter()
    for r in rows:
        d = derive_dialog(r)
        if d is None:
            r.pop("dialog", None)
            continue
        made[r["primary_class"]] += 1
        encoded = isinstance(r.get("input"), str)
        r["dialog"] = json.dumps(d, ensure_ascii=False, separators=COMPACT) if encoded else d
    if write:
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(r, ensure_ascii=False, separators=sep) + "\n" for r in rows),
                       encoding="utf-8")
        tmp.replace(path)
    shown = path.relative_to(REPO_DIR) if path.is_relative_to(REPO_DIR) else path
    return {"file": str(shown), "cases": len(rows), "dialogs": dict(made),
            "replies_per_dialog": dict(Counter(len(derive_dialog(r)["user_replies"])
                                               for r in rows if derive_dialog(r))),
            "written": write}


# ---------------------------------------------------------------- симуляторы пользователя
class ScriptedUser:
    name = "scripted"

    async def reply(self, dialog: dict[str, Any], question: str, i: int) -> tuple[str, float]:
        replies = dialog["user_replies"]
        return (replies[i] if i < len(replies) else NO_MORE), 0.0


USER_PROMPT = """Ты — пользователь программы 1С:Полиграфия и пишешь в чат поддержки. Бот задал тебе вопрос.
Ответь как пользователь: коротко, по-русски, только тем, что есть в «Что ты знаешь». Ничего не придумывай:
если ответа на вопрос в этих сведениях нет — так и скажи («не знаю», «не могу сказать»).
Не задавай встречных вопросов, не благодари, не объясняй, что ты симулятор.

Что ты знаешь (твои реплики, которые ты готов сообщить):
{facts}

Уже сообщил боту: {told}

Вопрос бота:
{question}"""


class LLMUser:
    """Пользователь-модель: отвечает на вопрос бота только по фактам кейса (temperature 0)."""

    name = "llm"

    def __init__(self, settings: Settings) -> None:
        from polybot.clients.llm import OpenAICompatibleLLM

        cfg = settings.llm.model_copy(update={"temperature": 0.0, "response_format": "none",
                                              "max_tokens": 200})
        self.llm, self.cfg = OpenAICompatibleLLM(cfg), cfg

    async def reply(self, dialog: dict[str, Any], question: str, i: int) -> tuple[str, float]:
        facts = "\n".join(f"- {t}" for t in dialog["user_replies"]) or "- (ничего)"
        told = "; ".join(dialog["user_replies"][:i]) or "ничего, кроме первого сообщения"
        r = await self.llm.complete([{"role": "user", "content": USER_PROMPT.format(
            facts=facts, told=told, question=question)}])
        cost = (r.prompt_tokens * self.cfg.price_in_per_1k + r.completion_tokens * self.cfg.price_out_per_1k) / 1000
        return r.text.strip() or NO_MORE, cost


# ---------------------------------------------------------------- прогон
async def run_dialog(bot, case: dict[str, Any], user, rep: int, checker: SourceChecker) -> dict[str, Any]:
    from harness.systems import ACTION, origin_omnidesk

    d = get_dialog(case)
    own = origin_omnidesk(case)
    exclude = (f"omnidesk:{own}",) if own else ()
    session = f"dialog:{case['case_id']}#r{rep}"
    msg, turns, asked, user_cost = d["first_message"], [], 0, 0.0
    t0 = time.perf_counter()
    r = None
    for _ in range(d["max_clarifications"] + 2):        # +2: лишний ход покажет, что бот не остановился
        r = await bot.respond(session, msg, exclude)
        turns.append({"user": msg, "kind": str(r.kind), "reason": r.reason, "bot": r.text,
                      "sources": [s.doc_id for s in r.sources], "path": r.path,
                      "llm_rub": r.llm.cost_rub, "trace_id": r.trace_id})
        if r.kind is not ReplyKind.CLARIFY:
            break
        asked += 1
        msg, c = await user.reply(d, r.text, asked - 1)
        user_cost += c
    final = ACTION[r.kind]
    gdocs = checker.gold_docs(case, own)
    from harness.systems import SystemReply

    sr = SystemReply(action=final, text=r.text, reason=r.reason, latency_ms=0.0,
                     sources=[{"doc_id": s.doc_id, "title": s.title, "url": s.url} for s in r.sources])
    n_replies = len(d["user_replies"])
    return {
        "case_id": case["case_id"], "repeat": rep, "class": case["primary_class"],
        "expected_outcome": d["expected_outcome"], "final_action": final, "final_reason": r.reason,
        "outcome_ok": final == d["expected_outcome"],
        "clarifications": asked, "max_clarifications": d["max_clarifications"],
        "over_limit": asked > d["max_clarifications"],
        # уточнения после того, как пользователь уже сообщил всё, что знает
        "after_replies_exhausted": max(0, asked - n_replies),
        "asked_on_first_turn": turns[0]["kind"] == "clarify",
        "source_verdict": checker.verdict(sr, gdocs), "gold_docs": sorted(gdocs),
        "bot_questions": [t["bot"] for t in turns if t["kind"] == "clarify"],
        "reference_questions": d["reference_questions"],
        "turns": turns, "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
        "bot_rub": round(sum(t["llm_rub"] for t in turns), 4), "user_rub": round(user_cost, 4),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = lambda cls: [r for r in rows if r["class"] == cls]  # noqa: E731
    answers = [r for r in rows if r["final_action"] == "answer"]
    verdicts = Counter(r["source_verdict"] for r in answers)
    return {
        "n_dialogs": len(rows),
        "D1_outcome_ok": frac(sum(r["outcome_ok"] for r in rows), len(rows)),
        "D1_by_class": {c: frac(sum(r["outcome_ok"] for r in by(c)), len(by(c))) for c in sorted(DIALOG_CLASSES)},
        "final_actions": {c: dict(Counter(r["final_action"] for r in by(c))) for c in sorted(DIALOG_CLASSES)},
        "D2_source_ok_among_answers": frac(verdicts["source_ok"], len(answers)),
        "source_verdicts": dict(verdicts),
        "D3_clarifications": {"distribution": dict(sorted(Counter(r["clarifications"] for r in rows).items())),
                              "mean": round(sum(r["clarifications"] for r in rows) / len(rows), 2) if rows else None,
                              "over_limit": frac(sum(r["over_limit"] for r in rows), len(rows))},
        "D4_kept_asking_after_user_told_all": frac(sum(r["after_replies_exhausted"] > 0 for r in rows), len(rows)),
        "asked_on_first_turn": frac(sum(r["asked_on_first_turn"] for r in rows), len(rows)),
        "fallback_reasons": dict(Counter(r["final_reason"] for r in rows if r["final_action"] == "fallback")),
        "latency_ms_per_dialog": {"p50": round(quantile([r["latency_ms"] for r in rows], 0.5), 1),
                                  "p95": round(quantile([r["latency_ms"] for r in rows], 0.95), 1)},
        "cost_rub": {"bot": round(sum(r["bot_rub"] for r in rows), 3),
                     "user_simulator": round(sum(r["user_rub"] for r in rows), 3)},
    }


async def run_all(rag, cases, user, repeat: int, concurrency: int, on_row) -> None:
    checker = SourceChecker()
    sem = asyncio.Semaphore(concurrency)
    done = 0

    async def one(case, rep):
        nonlocal done
        async with sem:
            on_row(await run_dialog(rag.bot, case, user, rep, checker))
        done += 1
        if done % 10 == 0 or done == len(cases) * repeat:
            print(f"  {done}/{len(cases) * repeat}")

    await asyncio.gather(*(one(c, r) for r in range(repeat) for c in cases))


def print_summary(s: dict[str, Any]) -> None:
    f = lambda m: f"{m['n']}/{m['N']} ({100 * m['share']:.0f}%)" if m["N"] else "—"  # noqa: E731
    print(f"\nДиалогов: {s['n_dialogs']}")
    print(f"  D1 итог верен (ответ / консультант)      {f(s['D1_outcome_ok'])}")
    for c, m in s["D1_by_class"].items():
        print(f"     {c}: {f(m)}   итоги: {s['final_actions'][c]}")
    print(f"  D2 источник ответа совпал с эталоном     {f(s['D2_source_ok_among_answers'])}  {s['source_verdicts']}")
    print(f"  D3 уточнений: {s['D3_clarifications']['distribution']}, в среднем {s['D3_clarifications']['mean']}; "
          f"сверх лимита {f(s['D3_clarifications']['over_limit'])}")
    print(f"  D4 спрашивал, когда пользователь всё сказал {f(s['D4_kept_asking_after_user_told_all'])}")
    print(f"  уточнил на первом ходе                   {f(s['asked_on_first_turn'])}")
    print(f"  P50/P95 диалога {s['latency_ms_per_dialog']['p50']}/{s['latency_ms_per_dialog']['p95']} мс, "
          f"LLM: бот {s['cost_rub']['bot']} ₽, симулятор {s['cost_rub']['user_simulator']} ₽")


def cmd_run(args) -> None:
    from harness.systems import RagSystem

    path = SETS[args.set]
    cases = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
    cases = [c for c in cases if get_dialog(c)]
    if not cases:
        raise SystemExit(f"в {path.name} нет поля dialog: сначала python -m harness.dialogs build --write")
    if args.classes:
        cases = [c for c in cases if c["primary_class"] in set(args.classes.split(","))]
    cases = cases[:args.limit] if args.limit else cases
    settings = Settings()
    rag = RagSystem(settings)
    user = LLMUser(settings) if args.user == "llm" else ScriptedUser()
    run_id = f"dialogs_{args.set}_{user.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = RESULTS / run_id
    out.mkdir(parents=True, exist_ok=True)
    meta = {"run_id": run_id, "set": args.set, "set_file": str(path.relative_to(REPO_DIR)),
            "set_sha256": hashlib.sha256(path.read_bytes()).hexdigest()[:16], "n_dialogs": len(cases),
            "repeat": args.repeat, "user_simulator": user.name, "dialog_version": DIALOG_VERSION,
            "filters": {"limit": args.limit, "classes": args.classes}, "git": git_state(),
            "system_info": rag.info(), "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    print(f"{run_id}: {len(cases)} диалогов × {args.repeat}, пользователь {user.name}")
    rows: list[dict[str, Any]] = []
    with (out / "cases.jsonl").open("w", encoding="utf-8") as f:
        def on_row(row):
            rows.append(row)
            f.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        t0 = time.time()
        try:
            asyncio.run(run_all(rag, cases, user, args.repeat, args.concurrency, on_row))
        finally:
            close = getattr(rag.bot.services.retriever, "close", None)
            if close:
                asyncio.run(close())
    s = summarize(rows)
    meta.update(finished_at=time.strftime("%Y-%m-%dT%H:%M:%S%z"), duration_s=round(time.time() - t0, 1),
                cost_rub=s["cost_rub"])
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(s)
    print(f"\nРезультаты: {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--set", choices=list(SETS), default="dev")
    b.add_argument("--file", type=Path, help="произвольный jsonl с кейсами (например, all_cases)")
    b.add_argument("--write", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--set", choices=list(SETS), default="dev")
    r.add_argument("--user", choices=["scripted", "llm"], default="scripted")
    r.add_argument("--classes")
    r.add_argument("--limit", type=int)
    r.add_argument("--repeat", type=int, default=1)
    r.add_argument("--concurrency", type=int, default=4)
    args = ap.parse_args()
    path = (args.file.resolve() if getattr(args, "file", None) else SETS[args.set])
    if path != SETS["dev"] and os.environ.get("HARNESS_ALLOW_EVAL") != "1":
        raise SystemExit("кроме dev — только с HARNESS_ALLOW_EVAL=1 (eval закрыт до КТ4, all_cases содержит eval)")
    if args.cmd == "build":
        print(json.dumps(build(path, args.write), ensure_ascii=False, indent=2))
    else:
        cmd_run(args)


if __name__ == "__main__":
    main()
