"""Клиент LLM.

Контракт инструмента `llm`:
- сигнатура: `await complete(messages, json_schema) -> LLMResult`;
- таймаут `settings.llm.timeout_s` (20 с), ретраев `settings.llm.max_retries` (1);
- при ошибке бросает `ToolError(tool="llm", kind=timeout|unavailable|bad_response)`;
- лимит вызовов — 1 на сообщение пользователя: его обеспечивает граф (узел `generate` один).

`OpenAICompatibleLLM` работает с любым OpenAI-совместимым сервером: Ollama, vLLM.
`ScriptedLLM` отдаёт заранее заданные ответы — для тестов и, на КТ4, для реплея трейса.
"""
from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from polybot.errors import ToolError, ToolErrorKind
from polybot.settings import LLMSettings

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


class OpenAICompatibleLLM:
    def __init__(self, cfg: LLMSettings):
        from openai import AsyncOpenAI

        self.cfg = cfg
        self.model_name = cfg.model
        self._client = AsyncOpenAI(
            base_url=cfg.base_url,
            api_key=cfg.api_key,
            timeout=cfg.timeout_s,
            max_retries=cfg.max_retries,
        )

    async def complete(
        self, messages: list[dict[str, str]], json_schema: dict[str, Any] | None = None
    ) -> LLMResult:
        import openai

        params: dict[str, Any] = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,
            "max_tokens": self.cfg.max_tokens,
        }
        if json_schema is not None:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "reply", "schema": json_schema, "strict": True},
            }
        t0 = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(**params)
        except openai.APITimeoutError as exc:
            raise ToolError("llm", ToolErrorKind.TIMEOUT, str(exc)) from exc
        except openai.APIConnectionError as exc:
            raise ToolError("llm", ToolErrorKind.UNAVAILABLE, str(exc)) from exc
        except openai.APIStatusError as exc:
            raise ToolError("llm", ToolErrorKind.UNAVAILABLE, f"HTTP {exc.status_code}") from exc
        latency = (time.perf_counter() - t0) * 1000

        if not resp.choices or not resp.choices[0].message.content:
            raise ToolError("llm", ToolErrorKind.BAD_RESPONSE, "пустой ответ")
        # Qwen3 и другие модели с режимом рассуждений могут вернуть <think>…</think>.
        text = THINK_RE.sub("", resp.choices[0].message.content).strip()
        usage = resp.usage
        return LLMResult(
            text=text,
            model=resp.model or self.cfg.model,
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
