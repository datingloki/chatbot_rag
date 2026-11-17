"""Единая точка входа: `await Bot.respond(session_id, message) -> Response`.

Её вызывают и Telegram-обработчик, и харнесс — поэтому прогоны на наборе и живой бот идут
одним и тем же путём (требование КТ3: «один вход»).
"""
from __future__ import annotations

import time
import uuid

from polybot.clients.domain import DomainClassifierService, DomainService
from polybot.clients.llm import LLMClient, OpenAICompatibleLLM
from polybot.clients.retriever import NullRetriever, Retriever
from polybot.graph_config import build_graph
from polybot.pipeline.context import Context, Services
from polybot.pipeline.graph import Graph
from polybot.pipeline.orchestrator import Orchestrator
from polybot.schemas import ReplyKind, Response, SessionState
from polybot.settings import Settings


class SessionStore:
    """Состояние чатов в памяти. На этапе 2 заменяется SQLite с тем же интерфейсом."""

    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        return self._sessions.setdefault(session_id, SessionState(session_id=session_id))

    def save(self, state: SessionState) -> None:
        self._sessions[state.session_id] = state


class Bot:
    def __init__(
        self,
        settings: Settings,
        services: Services,
        graph: Graph | None = None,
        store: SessionStore | None = None,
    ):
        self.settings = settings
        self.services = services
        self.graph = graph or build_graph()
        self.orchestrator = Orchestrator(self.graph, max_steps=settings.limits.max_steps)
        self.store = store or SessionStore()

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        retriever: Retriever | None = None,
        domain: DomainService | None = None,
    ) -> Bot:
        settings = settings or Settings()
        services = Services(
            domain=domain or DomainClassifierService(settings.domain_classifier_dir),
            llm=llm or OpenAICompatibleLLM(settings.llm),
            retriever=retriever or NullRetriever(),
        )
        return cls(settings, services)

    def new_problem(self, session_id: str) -> None:
        """Кнопка «Проблема решена» или команда /new."""
        state = self.store.get(session_id)
        state.new_problem()
        self.store.save(state)

    async def respond(self, session_id: str, message: str) -> Response:
        t0 = time.perf_counter()
        state = self.store.get(session_id)
        ctx = Context(
            trace_id=uuid.uuid4().hex,
            message=message,
            session=state,
            settings=self.settings,
            services=self.services,
        )
        await self.orchestrator.run(ctx)
        self.store.save(state)
        return Response(
            trace_id=ctx.trace_id,
            session_id=session_id,
            kind=ctx.reply_kind or ReplyKind.FALLBACK,
            text=ctx.reply_text,
            sources=ctx.sources,
            steps=ctx.steps,
            latency_ms=round((time.perf_counter() - t0) * 1000, 2),
            llm=ctx.llm_usage,
            reason=ctx.reason,
            path=ctx.path,
        )
