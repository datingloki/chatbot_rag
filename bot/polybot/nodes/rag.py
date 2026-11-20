"""Узлы RAG: поиск, проверка опоры, генерация (единственный вызов LLM), проверка ответа."""
from __future__ import annotations

import hashlib
import json
import re
from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from polybot import texts
from polybot.retrieval.query import build_query
from polybot.errors import ToolError
from polybot.nodes.base import BaseNode
from polybot.pipeline.context import Context
from polybot.schemas import PendingClarification, ReplyKind, Source


# ---------------------------------------------------------------- retrieve
class RetrieveDecision(StrEnum):
    OK = "ok"
    ERROR = "error"


class Retrieve(BaseNode):
    """Поисковый запрос — исходный вопрос вместе с уточнениями: по одному ответу
    «да, в заказе» искать нечего."""

    name = "retrieve"
    decisions = RetrieveDecision

    async def run(self, ctx: Context) -> RetrieveDecision:
        query = build_query(ctx.problem_messages)
        top_k = ctx.settings.retrieval.top_k
        try:
            ctx.chunks = await ctx.services.retriever.search(query, top_k, ctx.exclude_doc_ids)
        except ToolError as exc:
            ctx.fail(f"retrieval_{exc.kind}")
            ctx.note(query=query, error=str(exc))
            return RetrieveDecision.ERROR
        ctx.note(query=query, top_k=top_k, index_version=getattr(ctx.services.retriever, "version", None),
                 excluded=list(ctx.exclude_doc_ids),
                 chunks=[{"id": c.chunk_id, "score": round(c.score, 4), "rank": c.meta.get("rank"),
                          "source": c.meta.get("source")} for c in ctx.chunks])
        return RetrieveDecision.OK


# ---------------------------------------------------------------- evidence_gate
class EvidenceDecision(StrEnum):
    FOUND = "found"
    NOT_FOUND = "not_found"


class EvidenceGate(BaseNode):
    """Есть ли опора: косинус лучшего чанка ≥ `min_score`. В промпт идут чанки с косинусом
    ≥ `min_chunk_score` (порог ниже: чанк, найденный BM25, может иметь средний косинус, но
    содержать ответ). Пороги подбираются на dev: `python -m harness.retrieval_eval`."""

    name = "evidence_gate"
    decisions = EvidenceDecision

    async def run(self, ctx: Context) -> EvidenceDecision:
        cfg = ctx.settings.retrieval
        best = max((c.score for c in ctx.chunks), default=0.0)
        found = best >= cfg.min_score
        ctx.evidence = [c for c in ctx.chunks if c.score >= cfg.min_chunk_score] if found else []
        ctx.note(best_score=round(best, 4), min_score=cfg.min_score,
                 min_chunk_score=cfg.min_chunk_score, kept=[c.chunk_id for c in ctx.evidence])
        return EvidenceDecision.FOUND if ctx.evidence else EvidenceDecision.NOT_FOUND


# ---------------------------------------------------------------- generate
REPLY_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": [
            "answer", "clarify", "handoff", "out_of_domain", "professional_advice",
        ]},
        "text": {"type": "string"},
        "cited_chunk_ids": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["type", "text", "cited_chunk_ids"],
    "additionalProperties": False,
}


@lru_cache(maxsize=8)
def load_prompt(path: Path) -> tuple[str, str]:
    """(текст промпта, версия = имя файла + первые 12 символов sha256)."""
    text = path.read_text(encoding="utf-8")
    sha = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return text, f"{path.stem}@{sha}"


SOURCE_LABELS = {"faq": "FAQ 1cpoly.ru", "manual": "Руководство пользователя",
                 "omnidesk": "Архив ответов службы поддержки"}


def source_label(c) -> str:
    kind = SOURCE_LABELS.get(c.meta.get("source"), "")
    where = c.url or c.meta.get("section", "")
    return " — ".join(x for x in (kind, where) if x) or c.title


def build_messages(ctx: Context, system_prompt: str) -> list[dict[str, str]]:
    lim = ctx.settings.limits
    turns = [t for t in ctx.session.history][-lim.history_turns:]
    history = "\n".join(
        f"{'Пользователь' if t.role == 'user' else 'Бот'}: {t.text[:lim.history_chars_per_turn]}"
        for t in turns
    ) or "(нет)"
    fragments = "\n\n".join(
        f"[{c.chunk_id}] {c.title}\nИсточник: {source_label(c)}\n{c.text}" for c in ctx.evidence
    )
    status = (f"Уточняющих вопросов по этой проблеме уже задано: {ctx.session.clarifications} "
              f"из {lim.max_clarifications}.")
    user = (
        f"Предыдущие сообщения диалога:\n{history}\n\n"
        f"Фрагменты базы знаний:\n{fragments}\n\n"
        f"{status}\n\n"
        f"Вопрос пользователя:\n" + "\n".join(ctx.problem_messages)
    )
    return [{"role": "system", "content": system_prompt}, {"role": "user", "content": user}]


class GenerateDecision(StrEnum):
    OK = "ok"
    ERROR = "error"


class Generate(BaseNode):
    name = "generate"
    decisions = GenerateDecision

    async def run(self, ctx: Context) -> GenerateDecision:
        system_prompt, version = load_prompt(ctx.settings.prompt_file)
        ctx.prompt_version = version
        messages = build_messages(ctx, system_prompt)
        try:
            result = await ctx.services.llm.complete(messages, json_schema=REPLY_SCHEMA)
        except ToolError as exc:
            ctx.fail(f"llm_{exc.kind}")
            ctx.note(prompt_version=version, error=str(exc))
            return GenerateDecision.ERROR
        ctx.llm_raw = result.text
        u = ctx.llm_usage
        u.model, u.calls = result.model, u.calls + 1
        u.prompt_tokens += result.prompt_tokens
        u.completion_tokens += result.completion_tokens
        u.latency_ms += result.latency_ms
        cfg = ctx.settings.llm
        cost = (result.prompt_tokens * cfg.price_in_per_1k
                + result.completion_tokens * cfg.price_out_per_1k) / 1000
        u.cost_rub = round(u.cost_rub + cost, 6)
        ctx.note(prompt_version=version, model=result.model, prompt_tokens=result.prompt_tokens,
                 completion_tokens=result.completion_tokens, latency_ms=result.latency_ms,
                 cost_rub=round(cost, 6),
                 raw=result.text)
        return GenerateDecision.OK


# ---------------------------------------------------------------- validate
# C07: никаких прямых правок БД и SQL.
SQL_RE = re.compile(
    r"\b(select|update|delete|insert|alter|drop|truncate)\b[^.\n]{0,60}\b(from|set|into|table|where)\b"
    r"|sql[- ]?запрос|через\s+sql|консол[ьи]\s+(субд|sql)",
    re.I,
)
# C06: не заявляем о созданной заявке без подтверждённой интеграции.
FALSE_HANDOFF_RE = re.compile(
    r"(заявк\w*|обращени\w*)\s+(уже\s+)?(создан\w*|оформлен\w*|передан\w*|зарегистрирован\w*)", re.I
)
# C10: без эмодзи.
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF]")
MAX_REPLY_CHARS = 3500   # запас до лимита сообщения Telegram (4096)

REPLY_TYPES = set(REPLY_SCHEMA["properties"]["type"]["enum"])
REFUSAL_TEMPLATES = {
    "out_of_domain": texts.REFUSE,
    "professional_advice": texts.REFUSE_PROFESSIONAL,
}


FENCE_RE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.S | re.I)


def strip_fences(raw: str) -> str:
    """Без json_schema (response_format=json_object/none) модели иногда оборачивают JSON
    в ```json … ```. Снимаем обёртку; остальной текст не трогаем."""
    m = FENCE_RE.match(raw)
    return m.group(1) if m else raw


class ValidateDecision(StrEnum):
    OK = "ok"
    FAIL = "fail"


class Validate(BaseNode):
    """Проверка ответа LLM кодом. Не прошёл — готовый fallback без повторной генерации (2.2)."""

    name = "validate"
    decisions = ValidateDecision

    async def run(self, ctx: Context) -> ValidateDecision:
        problem = self._problem(ctx)
        if problem is not None:
            ctx.fail(f"validation_{problem}")
            ctx.note(failed=problem)
            return ValidateDecision.FAIL
        return ValidateDecision.OK

    def _problem(self, ctx: Context) -> str | None:
        try:
            draft = json.loads(strip_fences(ctx.llm_raw or ""))
        except json.JSONDecodeError:
            return "bad_json"
        if not isinstance(draft, dict) or draft.get("type") not in REPLY_TYPES:
            return "bad_schema"
        kind = draft["type"]

        # Второй рубеж проверки домена (C05): классификатор пропустил запрос, LLM его отклонила.
        # Текст отказа — готовый шаблон: текст модели не показываем, чтобы посторонняя задача
        # не была выполнена даже частично.
        if kind in REFUSAL_TEMPLATES:
            ctx.reply(ReplyKind.REFUSE, REFUSAL_TEMPLATES[kind], reason=f"llm_{kind}")
            ctx.note(type=kind, reply_kind=ctx.reply_kind,
                     classifier_label=(ctx.domain or {}).get("label"))
            return None

        text, cited = draft.get("text"), draft.get("cited_chunk_ids")
        if not isinstance(text, str) or not text.strip() or not isinstance(cited, list):
            return "bad_schema"
        text = text.strip()
        if len(text) > MAX_REPLY_CHARS:
            return "too_long"
        if SQL_RE.search(text):
            return "unsafe_db_advice"
        if FALSE_HANDOFF_RE.search(text):
            return "false_handoff_claim"
        if EMOJI_RE.search(text):
            return "tone_emoji"

        if kind == "answer":
            known = {c.chunk_id: c for c in ctx.evidence}
            if not cited:
                return "uncited_answer"
            if any(cid not in known for cid in cited):
                return "unknown_citation"
            seen: dict[str, Source] = {}
            for cid in cited:
                c = known[cid]
                doc_id = c.meta.get("doc_id") or c.chunk_id.split("#", 1)[0]
                seen.setdefault(doc_id, Source(title=c.title, url=c.url, doc_id=doc_id))
            ctx.sources = list(seen.values())
            ctx.reply(ReplyKind.ANSWER, text, reason="grounded_answer")
        elif kind == "clarify":
            if ctx.clarifications_left <= 0:
                # Шестого уточнения нет: предлагаем консультанта (C04).
                ctx.reply(ReplyKind.HANDOFF, f"{texts.HANDOFF_LIMIT} {texts.HANDOFF_TAIL}",
                          reason="clarify_limit")
            else:
                ctx.reply(ReplyKind.CLARIFY, text, reason="question_unclear")
                ctx.set_pending = PendingClarification.QUESTION
        else:
            ctx.reply(ReplyKind.HANDOFF, f"{text} {texts.HANDOFF_TAIL}", reason="client_specific")
        ctx.note(type=kind, cited=cited, reply_kind=ctx.reply_kind)
        return None
