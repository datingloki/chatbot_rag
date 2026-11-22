"""Базовый узел графа (по образцу 08week_observability, src/pipeline/base_node.py).

Отличия от 08week:
- узел возвращает решение, а не кладёт его в контекст: так решение нельзя «забыть» сбросить;
- решение проверяется по типу: узел-развилка обязан вернуть значение своего перечисления;
- каждый шаг пишется в `ctx.steps` (вход, выход, решение, время, ошибка). На КТ4 эта запись
  становится span'ом Langfuse без изменения узлов.
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any, ClassVar

from polybot.pipeline.context import Context
from polybot.schemas import StepRecord


class BaseNode(ABC):
    name: ClassVar[str]
    # Перечисление решений для узла-развилки; None — у узла один выход.
    decisions: ClassVar[type[StrEnum] | None] = None

    async def execute(self, ctx: Context) -> StrEnum | None:
        record = StepRecord(node=self.name, started_at=time.time(), input=self.describe_input(ctx))
        ctx.current_step = record
        ctx.steps.append(record)
        t0 = time.perf_counter()
        try:
            decision = await self.run(ctx)
            self._check_decision(decision)
            record.decision = decision.value if decision is not None else None
            return decision
        except Exception as exc:
            record.error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            record.duration_ms = round((time.perf_counter() - t0) * 1000, 2)
            ctx.current_step = None

    def _check_decision(self, decision: Any) -> None:
        if self.decisions is None:
            if decision is not None:
                raise TypeError(f"узел {self.name} без развилки вернул решение {decision!r}")
        elif not isinstance(decision, self.decisions):
            raise TypeError(
                f"узел {self.name} должен вернуть {self.decisions.__name__}, а вернул {decision!r}"
            )

    def describe_input(self, ctx: Context) -> dict[str, Any]:
        """Что записать во вход шага. По умолчанию — ничего, чтобы не дублировать сообщение."""
        return {}

    @abstractmethod
    async def run(self, ctx: Context) -> StrEnum | None: ...
