"""Граф бота (plan_rag_bot.md, раздел 2). `python -m polybot.graph_config` печатает mermaid."""
from __future__ import annotations

from polybot.nodes.final import Final
from polybot.nodes.rag import (
    EvidenceDecision,
    EvidenceGate,
    Generate,
    GenerateDecision,
    Retrieve,
    RetrieveDecision,
    Validate,
    ValidateDecision,
)
from polybot.nodes.replies import ClarifyDomain, Fallback, Handoff, NoAnswer, Refuse, Reject
from polybot.nodes.routing import (
    ClarifyLimit,
    Domain,
    DomainDecision,
    InputCheck,
    InputDecision,
    LimitDecision,
)
from polybot.pipeline.graph import Graph, Node


def build_graph() -> Graph:
    return Graph(
        [
            Node("input_check", InputCheck(),
                 {InputDecision.OK: "domain", InputDecision.REJECT: "reject"}),
            Node("domain", Domain(),
                 {DomainDecision.IN: "retrieve", DomainDecision.OUT: "refuse",
                  DomainDecision.UNCLEAR: "clarify_limit"}),
            Node("clarify_limit", ClarifyLimit(),
                 {LimitDecision.ALLOWED: "clarify_domain", LimitDecision.EXHAUSTED: "handoff"}),
            Node("retrieve", Retrieve(),
                 {RetrieveDecision.OK: "evidence_gate", RetrieveDecision.ERROR: "fallback"}),
            Node("evidence_gate", EvidenceGate(),
                 {EvidenceDecision.FOUND: "generate", EvidenceDecision.NOT_FOUND: "no_answer"}),
            Node("generate", Generate(),
                 {GenerateDecision.OK: "validate", GenerateDecision.ERROR: "fallback"}),
            Node("validate", Validate(),
                 {ValidateDecision.OK: "final", ValidateDecision.FAIL: "fallback"}),
            Node("reject", Reject(), "final"),
            Node("refuse", Refuse(), "final"),
            Node("clarify_domain", ClarifyDomain(), "final"),
            Node("no_answer", NoAnswer(), "final"),
            Node("handoff", Handoff(), "final"),
            Node("fallback", Fallback(), "final"),
            Node("final", Final(), None),
        ],
        start="input_check",
        end="final",
        fallback="fallback",
    )


if __name__ == "__main__":
    print(build_graph().to_mermaid())
