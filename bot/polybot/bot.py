"""Единая точка входа: `await Bot.respond(session_id, message) -> Response`.

Её вызывают и Telegram-обработчик, и харнесс — поэтому прогоны на наборе и живой бот идут
одним и тем же путём (требование КТ3: «один вход»).
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
import uuid
from collections import defaultdict

from polybot import deadline, texts
from polybot.clients.domain import DomainClassifierService, DomainService
from polybot.clients.handoff import HandoffRequest, HandoffTicket, HandoffTool, MemoryHandoff
from polybot.clients.llm import LLMClient, OpenAICompatibleLLM
from polybot.clients.retriever import NullRetriever, Retriever
from polybot.graph_config import build_graph
from polybot.pipeline.context import Context, Services
from polybot.pipeline.graph import Graph
from polybot.pipeline.orchestrator import Orchestrator
from polybot.privacy import mask_pii
from polybot.errors import ToolError, ToolErrorKind
from polybot.schemas import LLMUsage, ReplyKind, Response, SessionState, StepRecord, Turn
from polybot.settings import Settings

logger = logging.getLogger(__name__)

# Ответ на предложение консультанта текстом. Только короткое однозначное «да» или «нет»;
# всё остальное — новое сообщение, а предложение без согласия снимается (C06).
CONSENT_RE = re.compile(
    r"^\W*(да|давайте|передайте|передай|передавайте|согласен|согласна|ок|окей|хорошо)\b[\w\s,.!]{0,25}$", re.I)
DECLINE_RE = re.compile(
    r"^\W*(нет|не надо|не нужно|не передавайте|не передавай|отказываюсь)\b[\w\s,.!]{0,25}$", re.I)
HANDOFF_STATUS_TIMEOUT_S = 0.2
FEEDBACK_RATINGS = ("up", "down")              # M09: 👍 / 👎 под ответом
OPEN = ("offered", "failed")                    # предложение ждёт согласия или повтора
ACTION_LABELS = {"handoff_confirm": "[Сохранить черновик]", "handoff_decline": "[Не нужно]",
                 "handoff_cancel": "[Отменить обращение]"}


def default_retriever(settings: Settings) -> Retriever:
    """Поиск по Qdrant, если индекс собран; иначе пустой поиск с предупреждением."""
    from polybot.retrieval.store import read_manifest

    if not read_manifest(settings.retrieval):
        logger.warning("индекс базы знаний не собран (python -m ingest.build_index): поиск пустой")
        return NullRetriever()
    from polybot.clients.qdrant_retriever import QdrantRetriever

    return QdrantRetriever(settings.retrieval)


class SessionStore:
    """Состояние чатов в памяти (тесты, харнесс). Для Telegram — polybot.storage.SQLiteSessionStore."""

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
        if services.handoff is None:
            services.handoff = MemoryHandoff()
        # Два сообщения одного чата подряд не должны обрабатываться одновременно:
        # оба прочитали бы одно состояние, и второе затёрло бы первое.
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        llm: LLMClient | None = None,
        retriever: Retriever | None = None,
        domain: DomainService | None = None,
        store: SessionStore | None = None,
        handoff: HandoffTool | None = None,
    ) -> Bot:
        settings = settings or Settings()
        services = Services(
            domain=domain or DomainClassifierService(settings.domain_classifier_dir),
            llm=llm or OpenAICompatibleLLM(settings.llm),
            retriever=retriever or default_retriever(settings),
            handoff=handoff,
        )
        return cls(settings, services, store=store)

    def new_problem(self, session_id: str) -> None:
        """Сбрасывает состояние текущей проблемы."""
        state = self.store.get(session_id)
        state.new_problem()
        self.store.save(state)

    async def solve_problem(self, session_id: str, trace_id: str) -> bool:
        async with self._locks[session_id]:
            mark_solved = getattr(self.store, "mark_solved", None)
            if mark_solved is None:
                self.new_problem(session_id)
                return True
            return mark_solved(self.store.get(session_id), trace_id)

    async def mark_repeated_problem(self, session_id: str, trace_id: str) -> bool:
        async with self._locks[session_id]:
            mark_repeated = getattr(self.store, "mark_repeated", None)
            return mark_repeated is not None and mark_repeated(self.store.get(session_id), trace_id)

    def feedback(self, session_id: str, trace_id: str, rating: str) -> bool:
        """Оценка 👍/👎 под содержательным ответом (M09). Повторное нажатие заменяет оценку.
        False — хранилище оценки не пишет (харнесс, тесты в памяти)."""
        if rating not in FEEDBACK_RATINGS:
            raise ValueError(f"неизвестная оценка {rating}")
        log = getattr(self.store, "log_feedback", None)
        if log is None:
            return False
        log(session_id, trace_id, rating)
        return True

    async def respond(self, session_id: str, message: str,
                      exclude_doc_ids: tuple[str, ...] = ()) -> Response:
        # C08: маскируем один раз на входе — дальше по графу, к LLM, в состояние сессии,
        # журнал и трассы идёт только замаскированный текст.
        message, masked = mask_pii(message)
        # C09: общий дедлайн хода считается с получения сообщения, включая очередь чата.
        with deadline.turn(self.settings.limits.turn_deadline_s):
            return await self._respond_locked(session_id, message, exclude_doc_ids, masked)

    async def _respond_locked(self, session_id: str, message: str,
                              exclude_doc_ids: tuple[str, ...], masked) -> Response:
        t0, started = time.perf_counter(), time.time()
        lock = self._locks[session_id]
        try:
            await deadline.bounded(lock.acquire(), "queue")
        except ToolError:
            reason = "queue_timeout"
            waited_ms = round((time.perf_counter() - t0) * 1000, 2)
            step = StepRecord(
                node="queue", started_at=started, duration_ms=waited_ms,
                decision=reason, output={"deadline_left_s": deadline.left_for_trace()},
            )
            resp = Response(
                trace_id=uuid.uuid4().hex, session_id=session_id, kind=ReplyKind.FALLBACK,
                text=texts.FALLBACK_QUEUE, sources=[], steps=[step], latency_ms=waited_ms,
                llm=LLMUsage(), reason=reason, path=["queue"],
            )
        else:
            try:
                state = self.store.get(session_id)
                problem_started_at = state.problem_started_at
                offer = state.handoff
                if offer is not None and offer.status in OPEN and CONSENT_RE.match(message):
                    resp = await self._handoff(session_id, "handoff_confirm", message)
                elif offer is not None and offer.status in OPEN and DECLINE_RE.match(message):
                    resp = await self._handoff(session_id, "handoff_decline", message)
                else:
                    expired = offer.offer_id if offer is not None and offer.status in OPEN else None
                    resp = await self._respond(session_id, message, exclude_doc_ids, dict(masked), expired)
            finally:
                lock.release()
            resp.latency_ms = round((time.perf_counter() - t0) * 1000, 2)
            resp.problem_started_at = problem_started_at
        self._log(message, resp)
        return resp

    def _log(self, message: str, resp: Response) -> None:
        log = getattr(self.store, "log_turn", None)
        if log is not None:
            try:
                log(message, resp)
            except Exception:                      # журнал не должен ломать ответ
                logger.exception("не удалось записать ход %s", resp.trace_id)

    async def _respond(self, session_id: str, message: str, exclude_doc_ids: tuple[str, ...],
                       pii_masked: dict[str, int], expired_offer: str | None = None) -> Response:
        t0 = time.perf_counter()
        state = self.store.get(session_id)
        if expired_offer:
            state.handoff = None
        ctx = Context(
            trace_id=uuid.uuid4().hex,
            message=message,
            session=state,
            settings=self.settings,
            services=self.services,
            exclude_doc_ids=tuple(exclude_doc_ids),
            pii_masked=pii_masked,
        )
        if expired_offer:
            ctx.steps.append(StepRecord(node="handoff_offer", started_at=time.time(), decision="expired",
                                        output={"offer_id": expired_offer, "consent": False}))
        await self.orchestrator.run(ctx)
        self.store.save(state)
        kind = ctx.reply_kind or ReplyKind.FALLBACK
        offer = state.handoff if kind is ReplyKind.HANDOFF else None
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
            handoff_offer_id=offer.offer_id if offer else None,
            actions=(["handoff_confirm", "handoff_decline"] if kind is ReplyKind.HANDOFF
                     else ["feedback_up", "feedback_down", "solved"] if kind is ReplyKind.ANSWER
                     else ["solved"] if kind is ReplyKind.CLARIFY else []),
        )

    # ---------------------------------------------------------------- передача консультанту (C06)
    async def refresh_handoff_offer(self, session_id: str) -> Response | None:
        """Показывает актуальное предложение по старой кнопке без ID."""
        async with self._locks[session_id]:
            offer = self.store.get(session_id).handoff
            if offer is None or offer.status not in OPEN:
                return None
            return Response(
                trace_id=uuid.uuid4().hex, session_id=session_id, kind=ReplyKind.HANDOFF,
                text=f"{texts.HANDOFF} {texts.HANDOFF_TAIL}", sources=[], steps=[],
                latency_ms=0.0, llm=LLMUsage(), actions=["handoff_confirm", "handoff_decline"],
                handoff_offer_id=offer.offer_id,
            )

    async def handoff_action(self, session_id: str, action: str, *, offer_id: str | None = None) -> Response:
        """Кнопка или сценарий: handoff_confirm (согласие, повтор), handoff_decline, handoff_cancel."""
        if action not in ACTION_LABELS:
            raise ValueError(f"неизвестное действие {action}")
        async with self._locks[session_id]:
            resp = await self._handoff(session_id, action, ACTION_LABELS[action], offer_id=offer_id)
        self._log(ACTION_LABELS[action], resp)
        return resp

    async def _handoff(self, session_id: str, action: str, user_text: str, *,
                       offer_id: str | None = None) -> Response:
        t0, started = time.perf_counter(), time.time()
        state = self.store.get(session_id)
        offer, tool = state.handoff, self.services.handoff
        if offer_id is not None and (offer is None or offer_id != offer.offer_id):
            return Response(
                trace_id=uuid.uuid4().hex, session_id=session_id, kind=ReplyKind.HANDOFF,
                text=texts.HANDOFF_STALE, sources=[], steps=[], latency_ms=0.0, llm=LLMUsage(),
                reason="handoff_stale", path=[action],
            )
        timeout = min(self.settings.handoff.timeout_s, deadline.remaining())
        out: dict = {"offer_id": offer and offer.offer_id, "status_before": offer and offer.status}
        error = None
        attempt_result = None
        actions: list[str] = []
        if action == "handoff_decline":
            if offer is not None and offer.status in OPEN:
                state.handoff, reason, text = None, "handoff_declined", texts.HANDOFF_DECLINED
            else:
                reason, text = "handoff_nothing", texts.HANDOFF_NOTHING
        elif action == "handoff_confirm":
            if offer is not None and offer.status == "sent":            # двойное нажатие
                reason, text = "handoff_duplicate", texts.HANDOFF_SENT.format(ticket=offer.ticket_id)
            elif offer is None or offer.status not in OPEN:
                reason, text = "handoff_nothing", texts.HANDOFF_NOTHING
            else:
                req = HandoffRequest(offer.offer_id, session_id, offer.reason, offer.messages)
                ticket = None
                try:
                    ticket = await asyncio.wait_for(tool.create(req), timeout)
                except (ToolError, TimeoutError) as exc:
                    kind = exc.kind if isinstance(exc, ToolError) else ToolErrorKind.TIMEOUT
                    if kind == ToolErrorKind.TIMEOUT:
                        ticket = await self._lookup_handoff_after_timeout(session_id, offer.offer_id)
                    if ticket is None:
                        timed_out = kind == ToolErrorKind.TIMEOUT
                        attempt_result = "timeout_unconfirmed" if timed_out else "failed"
                        offer.status, reason = "failed", f"handoff_{kind}"
                        text = texts.HANDOFF_TIMEOUT if timed_out else texts.HANDOFF_FAILED
                        error, actions = str(exc) or kind, ["handoff_confirm", "handoff_decline"]
                    else:
                        out["confirmed_after_timeout"] = True
                if ticket is not None:
                    if ticket.status == "cancelled":
                        offer.status, offer.ticket_id = "cancelled", ticket.ticket_id
                        reason = "handoff_cancelled"
                        text = texts.HANDOFF_CANCELLED.format(ticket=ticket.ticket_id)
                        attempt_result = "cancelled"
                    else:
                        offer.status, offer.ticket_id = "sent", ticket.ticket_id
                        reason = "handoff_duplicate" if ticket.duplicate else "handoff_sent"
                        text = texts.HANDOFF_SENT.format(ticket=ticket.ticket_id)
                        attempt_result = "saved"
                    out["ticket_id"] = ticket.ticket_id
        else:                                                            # handoff_cancel
            if offer is None or offer.status != "sent":
                reason, text = "handoff_nothing", texts.HANDOFF_NOTHING
            else:
                try:
                    await asyncio.wait_for(tool.cancel(offer.ticket_id), timeout)
                    offer.status, reason = "cancelled", "handoff_cancelled"
                    text = texts.HANDOFF_CANCELLED.format(ticket=offer.ticket_id)
                except (ToolError, asyncio.TimeoutError) as exc:
                    kind = exc.kind if isinstance(exc, ToolError) else ToolErrorKind.TIMEOUT
                    reason, text = f"handoff_cancel_{kind}", texts.HANDOFF_CANCEL_FAILED.format(ticket=offer.ticket_id)
                    error = str(exc) or kind
        state.history.append(Turn(role="user", text=user_text))
        state.history.append(Turn(role="assistant", text=text, kind=ReplyKind.HANDOFF))
        state.close_problem()                    # действия с передачей — про закрытую проблему
        self.store.save(state)
        out.update(status_after=state.handoff and state.handoff.status, reason=reason)
        step = StepRecord(node=action, started_at=started, duration_ms=round((time.perf_counter() - t0) * 1000, 2),
                          decision=reason, input={"text": user_text}, output=out, error=error)
        return Response(trace_id=uuid.uuid4().hex, session_id=session_id, kind=ReplyKind.HANDOFF, text=text,
                        sources=[], steps=[step], latency_ms=step.duration_ms, llm=LLMUsage(), reason=reason,
                        path=[action], actions=actions, handoff_offer_id=offer.offer_id if offer else None,
                        handoff_attempt_result=attempt_result)

    async def _lookup_handoff_after_timeout(self, session_id: str, offer_id: str) -> HandoffTicket | None:
        timeout = min(HANDOFF_STATUS_TIMEOUT_S, deadline.remaining())
        if timeout <= 0:
            return None
        try:
            return await asyncio.wait_for(self.services.handoff.lookup(session_id, offer_id), timeout)
        except (ToolError, TimeoutError):
            return None
