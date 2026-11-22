"""Диалог с ботом в терминале — чтобы проверить маршрутизацию без Telegram.

    .venv/bin/python -m polybot.cli            # интерактивно
    .venv/bin/python -m polybot.cli "вопрос"   # один ход

Команды: /new — новая проблема, /state — состояние сессии, /graph — mermaid, /quit.
На этапе 0 поиска нет (NullRetriever), поэтому доменные вопросы идут в no_answer и LLM
не вызывается.
"""
from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import asdict

from polybot.bot import Bot


def show(resp) -> None:
    print(f"\n[{resp.kind}] {resp.text}")
    for s in resp.sources:
        print(f"  источник: {s.title} — {s.url}")
    print(f"  путь: {' → '.join(resp.path)}  | причина: {resp.reason} | {resp.latency_ms} мс")


async def main(args: list[str]) -> None:
    bot = Bot.from_settings()
    session = "cli"
    if args:
        for text in args:
            show(await bot.respond(session, text))
        return
    print("Пишите вопрос. /new, /state, /graph, /quit")
    while True:
        try:
            text = input("\n> ")
        except (EOFError, KeyboardInterrupt):
            return
        if text == "/quit":
            return
        if text == "/new":
            bot.new_problem(session)
            print("новая проблема: счётчик уточнений сброшен")
            continue
        if text == "/state":
            state = asdict(bot.store.get(session))
            state["history"] = len(state["history"])
            print(json.dumps(state, ensure_ascii=False, indent=2, default=str))
            continue
        if text == "/graph":
            print(bot.graph.to_mermaid())
            continue
        show(await bot.respond(session, text))


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
