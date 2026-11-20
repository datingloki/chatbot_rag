"""Многошаговые сценарии (КТ3): диалог из нескольких ходов с проверкой каждого хода.

    .venv/bin/python -m harness.scenarios                          # все harness/scenarios/*.yaml
    .venv/bin/python -m harness.scenarios harness/scenarios/s2_retrieval_failure.yaml
    .venv/bin/python -m harness.scenarios --llm scripted           # без ключа: ответы LLM из YAML

Формат сценария (YAML):

    id: S1
    title: Уточнение, затем ответ
    requirement: второй шаг зависит от результата первого
    contract: [C04, C02, C03]
    faults:                          # необязательно: сбой инструмента на ходе N (с 1)
      - {turn: 2, tool: retrieve, kind: unavailable}     # tool: retrieve | llm | handoff
    limits: {max_steps: 4}           # необязательно: лимиты на время сценария (бюджет шагов, уточнений)
    turns:
      - user: "Не считается заказ, что делать?"
        llm: '{"type": "clarify", "text": "Какой тип заказа?", "cited_chunk_ids": []}'  # только для --llm scripted
        expect: {kind: clarify}
      - user: "Листовая продукция"
        expect:
          kind: answer                  # answer | clarify | refuse | handoff | reject | fallback
          kind_in: [answer, clarify]    # вместо kind: допустимо любое из списка
          reason: grounded_answer       # необязательно
          sources_any: ["faq:poly2-261"]
          sources_none: ["faq:poly2-181"]   # этих источников в ответе быть не должно
          must_contain: ["закладке"]
          must_not_contain: ["заявка создана"]
          clarifications: 1             # необязательно: счётчик уточнений после хода
          pending: question             # domain | question | null — ждёт ли бот ответа на уточнение
          llm_called: true              # был ли вызов LLM на этом ходе
          prompt_must_contain: ["замен"]        # что должно уйти в промпт LLM
          prompt_must_not_contain: ["123-45-67"] # чего в промпте быть не должно (C08)
      - command: new_problem            # «Проблема решена» / /new между ходами
      - action: handoff_confirm         # нажатие кнопки: handoff_confirm | handoff_decline | handoff_cancel
        expect:
          handoff_calls: 1              # сколько раз вызван инструмент передачи на этом ходе
          handoff_records: 1            # сколько черновиков по сессии в инструменте
          handoff_status: created       # статус последнего черновика: created | cancelled | null
          actions: [handoff_confirm, handoff_decline]   # какие кнопки предложены

Сценарий идёт через `Bot.respond` с одной сессией на сценарий. Результат —
harness/results/scenarios_<время>/: по файлу на сценарий (ходы, ожидание, факт, путь, причина)
и summary.json с итогом.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from polybot.bot import Bot
from polybot.clients.llm import LLMResult
from polybot.errors import ToolError, ToolErrorKind
from polybot.schemas import Response
from polybot.settings import BOT_DIR, Settings

SCENARIOS_DIR = BOT_DIR / "harness" / "scenarios"
RESULTS = BOT_DIR / "harness" / "results"


class Faults:
    """Сбои по номеру хода: обёртки над инструментами бота бросают ToolError на нужном ходе."""

    def __init__(self, faults: list[dict[str, Any]]):
        self.by_turn: dict[tuple[int, str], ToolErrorKind] = {
            (int(f["turn"]), f["tool"]): ToolErrorKind(f.get("kind", "unavailable")) for f in faults or []}
        self.turn = 0

    def check(self, tool: str) -> None:
        kind = self.by_turn.get((self.turn, tool))
        if kind is not None:
            raise ToolError(tool, kind, f"сбой из сценария на ходе {self.turn}")


class FaultyRetriever:
    def __init__(self, inner, faults: Faults):
        self.inner, self.faults = inner, faults
        self.version = getattr(inner, "version", None)

    async def search(self, query, top_k, exclude_doc_ids=()):
        self.faults.check("retrieve")
        return await self.inner.search(query, top_k, exclude_doc_ids)


class FaultyLLM:
    def __init__(self, inner, faults: Faults):
        self.inner, self.faults = inner, faults
        self.model_name = getattr(inner, "model_name", "")
        self.prompts: list[str] = []          # что ушло в LLM на текущем ходе

    async def complete(self, messages, json_schema=None):
        self.prompts.append("\n".join(m["content"] for m in messages))
        self.faults.check("llm")
        return await self.inner.complete(messages, json_schema)


class FaultyHandoff:
    """Инструмент передачи со сбоями из сценария и счётчиком вызовов на ходе."""

    def __init__(self, inner, faults: Faults):
        self.inner, self.faults, self.calls = inner, faults, 0

    async def create(self, request):
        self.calls += 1
        self.faults.check("handoff")
        return await self.inner.create(request)

    async def cancel(self, ticket_id):
        self.calls += 1
        self.faults.check("handoff")
        return await self.inner.cancel(ticket_id)

    def records(self, session_id=None):
        return self.inner.records(session_id)


class TurnScriptLLM:
    """--llm scripted: ответ LLM берётся из поля `llm` текущего хода сценария."""

    model_name = "scripted"

    def __init__(self) -> None:
        self.next: str | None = None

    async def complete(self, messages, json_schema=None):
        if self.next is None:
            raise ToolError("llm", ToolErrorKind.BAD_RESPONSE, "в сценарии нет поля llm для этого хода")
        text, self.next = self.next, None
        return LLMResult(text=text, model="scripted", prompt_tokens=0, completion_tokens=0, latency_ms=0.0)


@dataclass
class TurnResult:
    turn: int
    user: str
    expect: dict[str, Any]
    kind: str
    reason: str | None
    text: str
    sources: list[str]
    path: list[str]
    clarifications: int
    pending: str | None = None
    llm_calls: int = 0
    handoff_calls: int = 0
    latency_ms: float = 0.0
    cost_rub: float = 0.0
    failures: list[str] = field(default_factory=list)


def check(expect: dict[str, Any], r: Response, clarifications: int, pending: str | None = None,
          prompts: list[str] | None = None, handoff: dict[str, Any] | None = None) -> list[str]:
    fails = []
    handoff = handoff or {}
    for key in ("handoff_calls", "handoff_records", "handoff_status"):
        if key in expect and handoff.get(key) != expect[key]:
            fails.append(f"{key}: ждали {expect[key]}, стало {handoff.get(key)}")
    if "actions" in expect and r.actions != list(expect["actions"]):
        fails.append(f"actions: ждали {expect['actions']}, стало {r.actions}")
    prompts = prompts or []
    if "kind" in expect and str(r.kind) != expect["kind"]:
        fails.append(f"kind: ждали {expect['kind']}, получили {r.kind} ({r.reason})")
    if "kind_in" in expect and str(r.kind) not in expect["kind_in"]:
        fails.append(f"kind: ждали одно из {expect['kind_in']}, получили {r.kind} ({r.reason})")
    if "reason" in expect and r.reason != expect["reason"]:
        fails.append(f"reason: ждали {expect['reason']}, получили {r.reason}")
    docs = {s.doc_id for s in r.sources}
    if expect.get("sources_any") and not docs & set(expect["sources_any"]):
        fails.append(f"sources_any: нет ни одного из {expect['sources_any']} в {sorted(docs)}")
    if expect.get("sources_none") and docs & set(expect["sources_none"]):
        fails.append(f"sources_none: в ответе есть {sorted(docs & set(expect['sources_none']))}")
    low = r.text.lower()
    for s in expect.get("must_contain", []):
        if s.lower() not in low:
            fails.append(f"must_contain: нет «{s}»")
    for s in expect.get("must_not_contain", []):
        if s.lower() in low:
            fails.append(f"must_not_contain: есть «{s}»")
    if "clarifications" in expect and clarifications != int(expect["clarifications"]):
        fails.append(f"clarifications: ждали {expect['clarifications']}, стало {clarifications}")
    if "pending" in expect and pending != expect["pending"]:
        fails.append(f"pending: ждали {expect['pending']}, стало {pending}")
    if "llm_called" in expect and bool(prompts) != bool(expect["llm_called"]):
        fails.append(f"llm_called: ждали {expect['llm_called']}, вызовов {len(prompts)}")
    prompt = "\n".join(prompts).lower()
    for s_ in expect.get("prompt_must_contain", []):
        if s_.lower() not in prompt:
            fails.append(f"prompt_must_contain: в промпте нет «{s_}»")
    for s_ in expect.get("prompt_must_not_contain", []):
        if s_.lower() in prompt:
            fails.append(f"prompt_must_not_contain: в промпте есть «{s_}»")
    return fails


async def run_scenario(spec: dict[str, Any], bot: Bot, scripted: TurnScriptLLM | None) -> dict[str, Any]:
    faults = Faults(spec.get("faults"))
    base_llm, base_ret = bot.services.llm, bot.services.retriever
    llm = FaultyLLM(scripted or base_llm, faults)
    bot.services.llm = llm
    bot.services.retriever = FaultyRetriever(base_ret, faults)
    base_handoff = bot.services.handoff
    handoff = FaultyHandoff(base_handoff, faults)
    bot.services.handoff = handoff
    base_limits, base_max_steps = bot.settings.limits, bot.orchestrator.max_steps
    if spec.get("limits"):
        bot.settings.limits = base_limits.model_copy(update=spec["limits"])
        bot.orchestrator.max_steps = bot.settings.limits.max_steps
    session = f"scenario:{spec['id']}:{time.time_ns()}"
    results: list[TurnResult] = []
    try:
        n = 0
        for step in spec["turns"]:
            if step.get("command") == "new_problem":
                bot.new_problem(session)
                continue
            n += 1
            faults.turn = n
            if scripted is not None:
                scripted.next = step.get("llm")
            llm.prompts, handoff.calls = [], 0
            if step.get("action"):
                r = await bot.handoff_action(session, step["action"])
            else:
                r = await bot.respond(session, step["user"])
            recs = handoff.records(session)
            hstate = {"handoff_calls": handoff.calls, "handoff_records": len(recs),
                      "handoff_status": recs[-1].status if recs else None}
            state = bot.store.get(session)
            clar, pending = state.clarifications, state.pending and str(state.pending)
            expect = step.get("expect") or {}
            results.append(TurnResult(
                turn=n, user=step.get("user") or f"[{step.get('action')}]", expect=expect, kind=str(r.kind), reason=r.reason, text=r.text,
                sources=[s.doc_id for s in r.sources], path=r.path, clarifications=clar, pending=pending,
                llm_calls=len(llm.prompts), handoff_calls=handoff.calls, latency_ms=r.latency_ms,
                cost_rub=r.llm.cost_rub, failures=check(expect, r, clar, pending, llm.prompts, hstate)))
    finally:
        bot.services.llm, bot.services.retriever, bot.services.handoff = base_llm, base_ret, base_handoff
        bot.settings.limits, bot.orchestrator.max_steps = base_limits, base_max_steps
    passed = all(not t.failures for t in results)
    return {"id": spec["id"], "title": spec.get("title", ""), "requirement": spec.get("requirement", ""),
            "contract": spec.get("contract", []), "faults": spec.get("faults", []), "passed": passed,
            "turns": [t.__dict__ for t in results]}


def load_specs(paths: list[Path]) -> list[dict[str, Any]]:
    files = [p for path in paths for p in (sorted(path.glob("*.yaml")) if path.is_dir() else [path])]
    specs = []
    for f in files:
        spec = yaml.safe_load(f.read_text(encoding="utf-8"))
        spec["_file"] = str(f.relative_to(BOT_DIR)) if f.is_relative_to(BOT_DIR) else str(f)
        specs.append(spec)
    return specs


async def main_async(args) -> int:
    specs = load_specs([Path(p) for p in args.paths] or [SCENARIOS_DIR])
    if not specs:
        print(f"нет сценариев в {SCENARIOS_DIR}")
        return 1
    from harness.run import git_state
    from harness.systems import RagSystem

    scripted = TurnScriptLLM() if args.llm == "scripted" else None
    rag = RagSystem(Settings(), **({"llm": scripted} if scripted else {}))
    bot = rag.bot
    out = RESULTS / f"scenarios_{args.llm}_{time.strftime('%Y%m%d_%H%M%S')}"
    out.mkdir(parents=True, exist_ok=True)
    summary = []
    for spec in specs:
        res = await run_scenario(spec, bot, scripted)
        res["file"] = spec["_file"]
        (out / f"{spec['id']}.json").write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
        res["cost_rub"] = round(sum(t["cost_rub"] for t in res["turns"]), 4)
        summary.append({k: res[k] for k in ("id", "title", "requirement", "passed", "file", "cost_rub")})
        mark = "ОК " if res["passed"] else "ПРОВАЛ"
        print(f"{mark} {spec['id']}: {spec.get('title', '')}")
        for t in res["turns"]:
            status = "  ✓" if not t["failures"] else "  ✗"
            print(f"{status} ход {t['turn']}: [{t['kind']}] {t['reason']} | {' → '.join(t['path'])}")
            for f in t["failures"]:
                print(f"      {f}")
    meta = {"llm_mode": args.llm, "git": git_state(), "system": rag.info(),
            "started_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (out / "summary.json").write_text(json.dumps({**meta, "passed": sum(s["passed"] for s in summary),
                                                  "total": len(summary), "scenarios": summary},
                                                 ensure_ascii=False, indent=2), encoding="utf-8")
    close = getattr(bot.services.retriever, "close", None)
    if close:
        await close()
    passed = sum(s["passed"] for s in summary)
    cost = sum(s["cost_rub"] for s in summary)
    print(f"\n{passed}/{len(summary)} сценариев прошли, LLM {cost:.2f} ₽. Результаты: {out}")
    return 0 if passed == len(summary) else 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="*", help="файлы или папки со сценариями")
    ap.add_argument("--llm", choices=["real", "scripted"], default="real")
    raise SystemExit(asyncio.run(main_async(ap.parse_args())))


if __name__ == "__main__":
    main()
