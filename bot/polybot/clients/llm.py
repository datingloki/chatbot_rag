"""Клиент LLM.

Контракт инструмента `llm`:
- сигнатура: `await complete(messages, json_schema) -> LLMResult`;
- попытка — min(`settings.llm.timeout_s` (12 с), остаток дедлайна хода); повторов
  `settings.llm.max_retries` (1), и только если до дедлайна остаётся ≥ `min_retry_s` (5 с)
  (polybot/deadline.py); повторяются timeout, unavailable и rate_limited;
- при ошибке бросает `ToolError(tool="llm", kind=...)`: timeout, unavailable (сеть, 5xx),
  auth (401/403), rate_limited (429), bad_request (400/404/422), bad_response (пустой ответ);
- лимит вызовов — 1 на сообщение пользователя: его обеспечивает граф (узел `generate` один).

`OpenAICompatibleLLM` работает с любым OpenAI-совместимым сервером: Yandex AI Studio, Ollama, vLLM.
`ScriptedLLM` отдаёт заранее заданные ответы — для тестов и, на КТ4, для реплея трейса.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from polybot import deadline
from polybot.errors import ToolError, ToolErrorKind
from polybot.settings import LLMSettings

RETRYABLE = (ToolErrorKind.TIMEOUT, ToolErrorKind.UNAVAILABLE, ToolErrorKind.RATE_LIMITED)

THINK_RE = re.compile(r"<think>.*?</think>", re.S)


@dataclass
class LLMResult:
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


class LLMClient(Protocol):
    model_name: str

    async def complete(
        self, messages: list[dict[str, str]], json_schema: dict[str, Any] | None = None
    ) -> LLMResult: ...


def status_kind(status: int) -> ToolErrorKind:
    if status in (401, 403):
        return ToolErrorKind.AUTH
    if status == 429:
        return ToolErrorKind.RATE_LIMITED
    if 400 <= status < 500:
        return ToolErrorKind.BAD_REQUEST
    return ToolErrorKind.UNAVAILABLE


def response_format(mode: str, json_schema: dict[str, Any] | None) -> dict[str, Any] | None:
    if json_schema is None or mode == "none":
        return None
    if mode == "json_object":
        return {"type": "json_object"}
    return {"type": "json_schema",
            "json_schema": {"name": "reply", "schema": json_schema, "strict": True}}


class OpenAICompatibleLLM:
    def __init__(self, cfg: LLMSettings):
        from openai import AsyncOpenAI

        self.cfg = cfg
        self.model_name = cfg.model_uri
        headers = {"x-data-logging-enabled": "false"} if cfg.disable_data_logging else None
        self._client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key.get_secret_value(),
            project=cfg.project,
            default_headers=headers,
            timeout=cfg.timeout_s,
            max_retries=0,                  # повторы — в complete(), с учётом дедлайна хода
        )

    def build_params(
        self, messages: list[dict[str, str]], json_schema: dict[str, Any] | None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {
            "model": self.cfg.model_uri,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "max_tokens": self.cfg.max_tokens,
        }
        if self.cfg.seed is not None:
            params["seed"] = self.cfg.seed
        if self.cfg.reasoning_effort:
            params["extra_body"] = {"reasoning_effort": self.cfg.reasoning_effort}
        fmt = response_format(self.cfg.response_format, json_schema)
        if fmt is not None:
            params["response_format"] = fmt
        return params

    async def complete(
        self, messages: list[dict[str, str]], json_schema: dict[str, Any] | None = None
    ) -> LLMResult:
        params = self.build_params(messages, json_schema)
        last: ToolError | None = None
        for attempt in range(self.cfg.max_retries + 1):
            left = deadline.remaining()
            if attempt and left < self.cfg.min_retry_s:
                break                                   # на повтор не хватает времени хода
            if left <= 0:
                raise ToolError("llm", ToolErrorKind.TIMEOUT, "дедлайн хода истёк")
            try:
                return await self._attempt(params, timeout=min(self.cfg.timeout_s, left))
            except ToolError as exc:
                if exc.kind not in RETRYABLE:
                    raise
                last = exc
        assert last is not None
        raise last

    async def _attempt(self, params: dict[str, Any], timeout: float) -> LLMResult:
        import openai

        t0 = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(**params, timeout=timeout)
        except openai.APITimeoutError as exc:
            raise ToolError("llm", ToolErrorKind.TIMEOUT, str(exc)) from exc
        except openai.APIConnectionError as exc:
            raise ToolError("llm", ToolErrorKind.UNAVAILABLE, str(exc)) from exc
        except openai.APIStatusError as exc:
            detail = f"HTTP {exc.status_code}: {str(exc.message)[:300]}"
            raise ToolError("llm", status_kind(exc.status_code), detail) from exc
        latency = (time.perf_counter() - t0) * 1000

        if not resp.choices or not resp.choices[0].message.content:
            finish = resp.choices[0].finish_reason if resp.choices else None
            detail = "пустой ответ"
            if finish == "length":
                detail += f": обрезан по max_tokens={self.cfg.max_tokens} (рассуждения? см. LLM__REASONING_EFFORT)"
            raise ToolError("llm", ToolErrorKind.BAD_RESPONSE, detail)
        # Qwen и другие модели с режимом рассуждений могут вернуть <think>…</think>.
        text = THINK_RE.sub("", resp.choices[0].message.content).strip()
        usage = resp.usage
        return LLMResult(
            text=text,
            model=resp.model or self.cfg.model_uri,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_ms=round(latency, 2),
        )


Script = str | ToolError | Callable[[list[dict[str, str]]], str]


class ScriptedLLM:
    """Отдаёт ответы по очереди. Элемент — строка, `ToolError` (будет брошена) или функция
    от сообщений. Все полученные промпты сохраняются в `calls`."""

    model_name = "scripted"

    def __init__(self, *script: Script):
        self.script = list(script)
        self.calls: list[list[dict[str, str]]] = []

    async def complete(
        self, messages: list[dict[str, str]], json_schema: dict[str, Any] | None = None
    ) -> LLMResult:
        self.calls.append(messages)
        if not self.script:
            raise AssertionError("ScriptedLLM: ответы закончились — LLM вызвали лишний раз")
        item = self.script.pop(0)
        if isinstance(item, ToolError):
            raise item
        text = item(messages) if callable(item) else item
        return LLMResult(text=text, model=self.model_name, prompt_tokens=0,
                         completion_tokens=0, latency_ms=0.0)
