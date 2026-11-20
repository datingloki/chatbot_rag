"""Проверка подключения к LLM из `bot/.env`.

    .venv/bin/python -m polybot.check_llm

1. Короткий вызов в каждом режиме response_format (json_schema → json_object → none):
   какой поддерживает провайдер, сколько длится, сколько токенов.
2. Полный ход бота на реальной статье FAQ (поиск подменён этой статьёй): доходит ли ответ
   через generate → validate до пользователя и с каким источником.

Ничего не пишет на диск. Стоимость — несколько тысяч токенов (≈ 1–2 ₽ для Qwen3.6 в Yandex).
"""
from __future__ import annotations

import asyncio
import json
import sys

from polybot.bot import Bot
from polybot.clients.llm import OpenAICompatibleLLM
from polybot.clients.retriever import StaticRetriever
from polybot.errors import ToolError
from polybot.nodes.rag import REPLY_SCHEMA, strip_fences
from polybot.schemas import Chunk
from polybot.settings import REPO_DIR, Settings

PING = [
    {"role": "system", "content": "Ответь только JSON по схеме."},
    {"role": "user", "content": 'Верни {"type": "clarify", "text": "Проверка связи.", '
                                '"cited_chunk_ids": []}'},
]
FAQ_ID = "poly2-261"
QUESTION = "Как учесть замену материала после расчета заказа?"


def faq_chunk() -> Chunk:
    path = REPO_DIR / "knowledge base" / "faq_articles.jsonl"
    for line in path.open(encoding="utf-8"):
        a = json.loads(line)
        if a["id"] == FAQ_ID:
            return Chunk(chunk_id=f"{FAQ_ID}#0", text=a["text"][:2500], title=a["title"],
                         url=a["source_url"], score=0.9)
    raise SystemExit(f"статья {FAQ_ID} не найдена в {path}")


async def ping(settings: Settings) -> str | None:
    """Возвращает первый работающий режим response_format."""
    for mode in ("json_schema", "json_object", "none"):
        cfg = settings.llm.model_copy(update={"response_format": mode})
        llm = OpenAICompatibleLLM(cfg)
        try:
            r = await llm.complete(PING, json_schema=REPLY_SCHEMA)
        except ToolError as exc:
            print(f"  {mode:12s} ошибка: {exc}")
            if exc.kind in ("auth", "unavailable", "timeout"):
                return None        # дело не в режиме: дальше пробовать бессмысленно
            continue
        try:
            json.loads(strip_fences(r.text))
            ok = "JSON ок"
        except json.JSONDecodeError:
            ok = f"не JSON: {r.text[:80]!r}"
        print(f"  {mode:12s} {r.latency_ms:7.0f} мс  in={r.prompt_tokens} out={r.completion_tokens}  {ok}")
        if ok == "JSON ок":
            return mode
    return None


async def main() -> int:
    settings = Settings()
    cfg = settings.llm
    print(f"LLM: {cfg.base_url}  модель: {cfg.model_uri}  каталог: {cfg.project or '—'}  "
          f"логирование провайдера: {'выключено' if cfg.disable_data_logging else 'включено'}")
    if "yandex" in cfg.base_url:
        missing = [name for name, ok in (("LLM__API_KEY", cfg.api_key.get_secret_value()
                                          not in ("", "not-needed")),
                                         ("LLM__PROJECT", bool(cfg.project))) if not ok]
        if missing:
            print(f"\nДля Yandex AI Studio заполните в bot/.env: {', '.join(missing)} (см. bot/README.md)")
            return 1
    print("\n1. response_format")
    mode = await ping(settings)
    if mode is None:
        print("\nНе удалось получить JSON ни в одном режиме — проверьте ключ, каталог и модель в .env")
        return 1
    if mode != cfg.response_format:
        print(f"\n  → в .env укажите LLM__RESPONSE_FORMAT={mode}")
        settings.llm = cfg.model_copy(update={"response_format": mode})

    print(f"\n2. Ход бота на статье FAQ {FAQ_ID}")
    bot = Bot.from_settings(settings, retriever=StaticRetriever([faq_chunk()]))
    r = await bot.respond("check", QUESTION)
    print(f"  вопрос:   {QUESTION}")
    print(f"  результат: [{r.kind}] причина={r.reason}")
    print(f"  путь:     {' → '.join(r.path)}")
    print(f"  LLM:      {r.llm.latency_ms:.0f} мс, in={r.llm.prompt_tokens} out={r.llm.completion_tokens}, "
          f"{r.llm.cost_rub:.3f} ₽;  весь ход {r.latency_ms:.0f} мс")
    for s in r.sources:
        print(f"  источник: {s.title} — {s.url}")
    print(f"\n{r.text}")
    return 0 if r.kind == "answer" else 2


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
