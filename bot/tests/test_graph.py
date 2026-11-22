from enum import StrEnum

import pytest

from polybot.graph_config import build_graph
from polybot.nodes.base import BaseNode
from polybot.pipeline.graph import Graph, GraphError, Node


class Two(StrEnum):
    A = "a"
    B = "b"


class Fork(BaseNode):
    name = "fork"
    decisions = Two

    async def run(self, ctx):
        return Two.A


class Plain(BaseNode):
    name = "plain"

    async def run(self, ctx):
        return None


def test_bot_graph_is_valid_and_renders():
    g = build_graph()
    mermaid = g.to_mermaid()
    for name in ("input_check", "domain", "clarify_limit", "retrieve", "generate", "validate",
                 "fallback", "final"):
        assert name in g.by_name
        assert name in mermaid
    assert "domain -->|unclear| clarify_limit" in mermaid


def test_readme_graph_matches_code():
    """Схема в bot/README.md — это вывод to_mermaid() (плюс пунктирные пояснения)."""
    from pathlib import Path

    readme = (Path(__file__).resolve().parent.parent / "README.md").read_text(encoding="utf-8")
    block = readme.split("```mermaid", 1)[1].split("```", 1)[0]
    readme_lines = [line for line in block.strip().splitlines() if "-.->" not in line
                    and not line.strip().startswith("orchestrator(")]
    assert readme_lines == build_graph().to_mermaid().splitlines(), (
        "граф изменился: обновите схему в README (python -m polybot.graph_config)")


def test_missing_decision_branch_rejected():
    with pytest.raises(GraphError, match="нет переходов"):
        Graph([Node("fork", Fork(), {Two.A: "end"}), Node("end", Plain(), None)],
              start="fork", end="end", fallback="end")


def test_unknown_target_rejected():
    with pytest.raises(GraphError, match="несуществующий"):
        Graph([Node("fork", Fork(), {Two.A: "end", Two.B: "nowhere"}),
               Node("end", Plain(), None)],
              start="fork", end="end", fallback="end")


def test_node_that_cannot_reach_end_rejected():
    with pytest.raises(GraphError, match="нельзя дойти до финала"):
        Graph([Node("fork", Fork(), {Two.A: "end", Two.B: "loop"}),
               Node("loop", Plain(), "loop"),
               Node("end", Plain(), None)],
              start="fork", end="end", fallback="end")


def test_unreachable_node_rejected():
    with pytest.raises(GraphError, match="недостижимы"):
        Graph([Node("start", Plain(), "end"), Node("orphan", Plain(), "end"),
               Node("end", Plain(), None)],
              start="start", end="end", fallback="end")
