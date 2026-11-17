"""Граф узлов (по образцу 08week, src/pipeline/config.py) с более строгой проверкой.

При сборке графа проверяется:
- имена узлов уникальны, стартовый, финальный и fallback-узлы существуют;
- все переходы ведут на существующие узлы;
- у узла-развилки описан переход для **каждого** значения его перечисления решений,
  у обычного узла — ровно один следующий узел, у финального — ни одного;
- каждый узел достижим из старта и с каждого узла можно дойти до финала.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from polybot.pipeline.base_node import BaseNode


@dataclass
class Node:
    name: str
    node: BaseNode
    next: str | dict[StrEnum, str] | None


class GraphError(ValueError):
    pass


class Graph:
    def __init__(self, nodes: list[Node], start: str, end: str, fallback: str):
        self.nodes = nodes
        self.by_name: dict[str, Node] = {}
        for n in nodes:
            if n.name in self.by_name:
                raise GraphError(f"узел {n.name!r} объявлен дважды")
            self.by_name[n.name] = n
        for role, name in (("start", start), ("end", end), ("fallback", fallback)):
            if name not in self.by_name:
                raise GraphError(f"{role}-узел {name!r} не объявлен")
        self.start, self.end, self.fallback = (self.by_name[x] for x in (start, end, fallback))
        self._validate_transitions()
        self._validate_paths()

    # ------------------------------------------------------------------ проверки
    def successors(self, n: Node) -> list[str]:
        if n.next is None:
            return []
        if isinstance(n.next, str):
            return [n.next]
        return list(n.next.values())

    def _validate_transitions(self) -> None:
        for n in self.nodes:
            decisions = n.node.decisions
            if n is self.end:
                if n.next is not None:
                    raise GraphError(f"у финального узла {n.name!r} не должно быть перехода")
                continue
            if n.next is None:
                raise GraphError(f"узел {n.name!r} никуда не ведёт")
            if decisions is None and not isinstance(n.next, str):
                raise GraphError(f"у узла {n.name!r} нет развилки, нужен один следующий узел")
            if decisions is not None:
                if not isinstance(n.next, dict):
                    raise GraphError(f"узел-развилка {n.name!r}: нужен переход по каждому решению")
                missing = set(decisions) - set(n.next)
                extra = set(n.next) - set(decisions)
                if missing or extra:
                    raise GraphError(
                        f"узел {n.name!r}: нет переходов для {sorted(missing)}; "
                        f"лишние решения {sorted(map(str, extra))}"
                    )
            for target in self.successors(n):
                if target not in self.by_name:
                    raise GraphError(f"узел {n.name!r} ссылается на несуществующий {target!r}")

    def _validate_paths(self) -> None:
        reachable: set[str] = set()
        stack = [self.start.name]
        while stack:
            name = stack.pop()
            if name in reachable:
                continue
            reachable.add(name)
            stack.extend(self.successors(self.by_name[name]))
        reachable.add(self.fallback.name)   # в fallback оркестратор переходит сам
        stack = [self.fallback.name]
        while stack:
            name = stack.pop()
            for nxt in self.successors(self.by_name[name]):
                if nxt not in reachable:
                    reachable.add(nxt)
                    stack.append(nxt)
        unreachable = set(self.by_name) - reachable
        if unreachable:
            raise GraphError(f"узлы недостижимы из старта: {sorted(unreachable)}")

        leads_to_end = {self.end.name}
        changed = True
        while changed:
            changed = False
            for n in self.nodes:
                if n.name not in leads_to_end and any(
                    s in leads_to_end for s in self.successors(n)
                ):
                    leads_to_end.add(n.name)
                    changed = True
        dead = set(self.by_name) - leads_to_end
        if dead:
            raise GraphError(f"с узлов {sorted(dead)} нельзя дойти до финала {self.end.name!r}")

    # ------------------------------------------------------------------ навигация
    def next_node(self, current: Node, decision: StrEnum | None) -> Node:
        if isinstance(current.next, dict):
            return self.by_name[current.next[decision]]  # type: ignore[index]
        return self.by_name[current.next]  # type: ignore[index]

    # ------------------------------------------------------------------ диаграмма
    def to_mermaid(self) -> str:
        lines = ["graph TD"]
        for n in self.nodes:
            shape = f'{{"{n.name}"}}' if n.node.decisions else f'["{n.name}"]'
            lines.append(f"    {n.name}{shape}")
            if isinstance(n.next, str):
                lines.append(f"    {n.name} --> {n.next}")
            elif isinstance(n.next, dict):
                for decision, target in n.next.items():
                    lines.append(f"    {n.name} -->|{decision.value}| {target}")
        # «end» — зарезервированное слово mermaid, поэтому классы называются иначе.
        lines.append(f"    classDef startNode fill:#d8f0d8\n    class {self.start.name} startNode")
        lines.append(f"    classDef endNode fill:#f6d6dc\n    class {self.end.name} endNode")
        lines.append(f"    classDef fallbackNode fill:#fbe8c8\n    class {self.fallback.name} fallbackNode")
        return "\n".join(lines)
