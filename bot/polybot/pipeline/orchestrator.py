"""Оркестратор: проходит граф от старта до финала.

Критерий остановки (КТ3):
- обычный случай — дошли до финального узла;
- исчерпан бюджет шагов `max_steps` → причина `budget_exhausted`, переход в fallback;
- узел бросил неожиданное исключение → причина `internal_error`, переход в fallback.

Путь fallback → final выполняется вне бюджета, но с собственным ограничением: если и он
не доходит до финала, оркестратор бросает `RuntimeError` — это ошибка графа, а не хода.
"""
from __future__ import annotations

import logging

from polybot.pipeline.context import Context
from polybot.pipeline.graph import Graph

logger = logging.getLogger(__name__)


class Orchestrator:
    def __init__(self, graph: Graph, max_steps: int):
        self.graph = graph
        self.max_steps = max_steps

    async def run(self, ctx: Context) -> None:
        node = self.graph.start
        budget = self.max_steps
        recovering = False
        recovery_budget = len(self.graph.nodes)

        while True:
            if not recovering:
                if budget == 0:
                    logger.warning("trace=%s budget_exhausted path=%s", ctx.trace_id, ctx.path)
                    ctx.fail("budget_exhausted")
                    node, recovering = self.graph.fallback, True
                else:
                    budget -= 1
            if recovering:
                if recovery_budget == 0:
                    raise RuntimeError(f"путь fallback не дошёл до финала: {ctx.path}")
                recovery_budget -= 1

            ctx.path.append(node.name)
            try:
                decision = await node.node.execute(ctx)
            except Exception:
                if recovering:
                    raise
                logger.exception("trace=%s node=%s failed", ctx.trace_id, node.name)
                ctx.fail("internal_error")
                node, recovering = self.graph.fallback, True
                continue

            if node is self.graph.end:
                return
            node = self.graph.next_node(node, decision)
            if node is self.graph.fallback:
                recovering = True
