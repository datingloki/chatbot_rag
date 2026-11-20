"""Системы, которые харнесс гоняет на наборе через один интерфейс (требование КТ3).

`await system.reply(case) -> SystemReply`, `system.info() -> dict` (версии для meta.json).

- `S0System` — бейзлайн на интентах (`../s0_baseline`), без LLM.
- `RagSystem` — наш бот (`polybot.Bot.respond`): классификатор домена, поиск, LLM, проверка.

Действия приводятся к словарю разметки dev (`expected_action`):
answer | clarify | refuse | offer_handoff, плюс reject_input и fallback.
"""
from __future__ import annotations

import asyncio
import hashlib
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from polybot.schemas import PendingClarification, ReplyKind, SessionState, Turn
from polybot.settings import REPO_DIR, Settings

ACTION = {ReplyKind.ANSWER: "answer", ReplyKind.CLARIFY: "clarify", ReplyKind.REFUSE: "refuse",
          ReplyKind.HANDOFF: "offer_handoff", ReplyKind.REJECT: "reject_input",
          ReplyKind.FALLBACK: "fallback"}


@dataclass
class SystemReply:
    action: str
    text: str
    reason: str | None
    sources: list[dict[str, str]]
    latency_ms: float
    llm: dict[str, Any] = field(default_factory=dict)
    path: list[str] = field(default_factory=list)
    trace_id: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)


class System(Protocol):
    name: str

    def info(self) -> dict[str, Any]: ...

    async def reply(self, case: dict[str, Any]) -> SystemReply: ...


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def origin_omnidesk(case: dict[str, Any]) -> str | None:
    """id обращения OmniDesk, из которого сделан кейс (его нельзя находить поиском)."""
    parts = case.get("origin", "").split(":")
    if len(parts) >= 3 and parts[0] == "real_anonymized_candidate" and parts[1] == "omnidesk":
        return parts[2]
    return None


# ---------------------------------------------------------------- S0
class S0System:
    name = "s0"

    def __init__(self) -> None:
        s0_dir = REPO_DIR / "s0_baseline"
        sys.path.insert(0, str(s0_dir))               # s0.py сам добавляет domain_classifier
        from s0 import MODEL_PATH, S0Bot  # type: ignore[import-not-found]

        self._bot = S0Bot.load()
        self._model_path = MODEL_PATH
        self._bot.respond("прогрев")                  # загрузка моделей вне замера задержки

    def info(self) -> dict[str, Any]:
        return {"system": "s0", "model_file": self._model_path.name,
                "model_sha256": _sha(self._model_path), "intents_version": self._bot.intents_version,
                "threshold": self._bot.threshold, "config": self._bot.cfg, "llm": None}

    async def reply(self, case: dict[str, Any]) -> SystemReply:
        inp = case["input"]
        t0 = time.perf_counter()
        r = await asyncio.to_thread(
            self._bot.respond, inp["user_message"], inp.get("history", []),
            bool((case.get("context") or {}).get("handoff_consent")))
        latency = (time.perf_counter() - t0) * 1000
        sources = []
        if r["action"] == "answer" and r.get("intent_id"):
            sources = [{"doc_id": f"faq:{r['intent_id']}", "title": "", "url": r.get("source_url") or ""}]
        return SystemReply(action=r["action"], text=r["text"], reason=r.get("reason"), sources=sources,
                           latency_ms=round(latency, 2),
                           extra={"domain": r.get("domain"), "intent_id": r.get("intent_id"),
                                  "p_intent": r.get("p_intent"), "ticket_created": r.get("ticket_created")})


# ---------------------------------------------------------------- RAG
def seed_session(case: dict[str, Any]) -> SessionState:
    """Состояние чата перед ходом кейса: история из `input.history`, счётчик уточнений из
    `clarification_count`. Если последняя реплика — вопрос бота, ход считается ответом на
    уточнение: домен определяется вместе с предыдущими сообщениями, поиск — по всей цепочке."""
    hist = case["input"].get("history", [])
    s = SessionState(session_id=case["case_id"])
    s.history = [Turn(role=h["role"], text=h["text"]) for h in hist]
    s.clarifications = int(case.get("clarification_count") or 0)
    if hist and hist[-1]["role"] == "assistant":
        s.pending = PendingClarification.QUESTION
        s.chain = [h["text"] for h in hist if h["role"] == "user"]
    return s


class RagSystem:
    name = "rag"

    def __init__(self, settings: Settings | None = None, **bot_kwargs) -> None:
        from polybot.bot import Bot

        self.settings = settings or Settings()
        self.bot = Bot.from_settings(self.settings, **bot_kwargs)

    def info(self) -> dict[str, Any]:
        from polybot.nodes.rag import load_prompt

        s, svc = self.settings, self.bot.services
        _, prompt_version = load_prompt(s.prompt_file)
        llm = s.llm
        return {
            "system": "rag",
            "llm": {"base_url": llm.base_url, "model": getattr(svc.llm, "model_name", llm.model_uri),
                    "temperature": llm.temperature, "seed": llm.seed, "max_tokens": llm.max_tokens,
                    "reasoning_effort": llm.reasoning_effort,
                    "response_format": llm.response_format,
                    "price_rub_per_1k": [llm.price_in_per_1k, llm.price_out_per_1k]},
            "prompt_version": prompt_version,
            "classifier_version": getattr(svc.domain, "version", None),
            "index_version": getattr(svc.retriever, "version", None),
            "retrieval": s.retrieval.model_dump(mode="json",
                                                include={"dense_model", "mode", "bm25_weight", "title_weight",
                                                         "omnidesk_k", "top_k", "prefetch_k", "min_score",
                                                         "min_chunk_score"}),
            "limits": s.limits.model_dump(),
            "graph_sha256": hashlib.sha256(self.bot.graph.to_mermaid().encode()).hexdigest()[:12],
        }

    async def reply(self, case: dict[str, Any]) -> SystemReply:
        sid = case["case_id"]
        self.bot.store.save(seed_session(case))
        own = origin_omnidesk(case)
        resp = await self.bot.respond(sid, case["input"]["user_message"],
                                      exclude_doc_ids=(f"omnidesk:{own}",) if own else ())
        u = resp.llm
        return SystemReply(
            action=ACTION[resp.kind], text=resp.text, reason=resp.reason,
            sources=[{"doc_id": s.doc_id, "title": s.title, "url": s.url} for s in resp.sources],
            latency_ms=resp.latency_ms,
            llm={"model": u.model, "calls": u.calls, "prompt_tokens": u.prompt_tokens,
                 "completion_tokens": u.completion_tokens, "latency_ms": round(u.latency_ms, 1),
                 "cost_rub": u.cost_rub},
            path=resp.path, trace_id=resp.trace_id,
            steps=[{"node": st.node, "decision": st.decision, "duration_ms": st.duration_ms,
                    "output": st.output, "error": st.error} for st in resp.steps],
        )
