from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from polybot.bot import Bot  # noqa: E402
from polybot.clients.domain import FixedDomain  # noqa: E402
from polybot.clients.llm import ScriptedLLM  # noqa: E402
from polybot.clients.retriever import StaticRetriever  # noqa: E402
from polybot.pipeline.context import Services  # noqa: E402
from polybot.schemas import Chunk  # noqa: E402
from polybot.settings import Settings  # noqa: E402

CHUNK = Chunk(chunk_id="faq:poly2-261#0", text="На закладке «Материалы» можно заменить материал.",
              title="Как учесть замену материала?",
              url="https://1cpoly.ru/questions/?action=show&id=261", score=0.82)
WEAK = Chunk(chunk_id="faq:license-007#0", text="Про лицензии.", title="Лицензии",
             url="https://1cpoly.ru/questions/faq-license/", score=0.2)


def reply(kind: str, text: str, cited: list[str] | None = None) -> str:
    return json.dumps({"type": kind, "text": text, "cited_chunk_ids": cited or []},
                      ensure_ascii=False)


def keyword_domain(text: str, history: list[dict]) -> str:
    """Тестовое правило домена: «борщ», «погода» — вне домена; «???» — неясно;
    «да, в заказе» — в домене только с историей; остальное — в домене."""
    t = text.lower()
    if "борщ" in t or "погод" in t:
        return "out_of_domain"
    if t.startswith("???"):
        return "unclear"
    if t.startswith("да, в заказе"):
        return "in_domain" if history else "unclear"
    return "in_domain"


def make_bot(*, llm=None, retriever=None, domain=None, **limits) -> Bot:
    settings = Settings(_env_file=None)
    for k, v in limits.items():
        setattr(settings.limits, k, v)
    services = Services(
        domain=domain or FixedDomain(keyword_domain),
        llm=llm or ScriptedLLM(),
        retriever=retriever if retriever is not None else StaticRetriever([CHUNK, WEAK]),
    )
    return Bot(settings, services)


def ask(bot: Bot, text: str, session: str = "s1"):
    return asyncio.run(bot.respond(session, text))


@pytest.fixture
def chunk() -> Chunk:
    return CHUNK
