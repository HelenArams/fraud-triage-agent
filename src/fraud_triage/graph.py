"""The triage graph.

ingest -> features -> score -+-> auto_clear ----------> END
                             +-> investigate ---------> END
                             +-> (high) escalated ----> END
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, TypedDict

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from fraud_triage.config import Settings
from fraud_triage.features import Features, compute
from fraud_triage.investigate import investigate
from fraud_triage.llm.base import LLMClient
from fraud_triage.loader import parse_rows
from fraud_triage.models import Transaction, TriageDecision, Verdict
from fraud_triage.scoring import RuleScore, score
from fraud_triage.tools import ToolContext


class TriageState(TypedDict, total=False):
    txn: dict[str, Any]
    history: list[dict[str, Any]]
    out_dir: str | None
    features: dict[str, Any]
    rule_score: dict[str, Any]
    decision: dict[str, Any]
    tool_calls: list[str]
    notes: Annotated[list[str], operator.add]
    path: Annotated[list[str], operator.add]
    final_verdict: str


def _txn(state: TriageState) -> Transaction:
    return Transaction.model_validate(state["txn"])


def _history(state: TriageState) -> list[Transaction]:
    return [Transaction.model_validate(h) for h in state.get("history", [])]


def build_graph(
    llm: LLMClient,
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver[Any] | None = None,
) -> CompiledStateGraph[TriageState, None, TriageState, TriageState]:
    settings = settings or Settings()

    def ingest(state: TriageState) -> dict[str, Any]:
        # run graph input through the loader as well, so a caller that skips
        # load_transactions still gets PAN masking and column checks
        [txn] = parse_rows([state["txn"]], source="graph input")
        history = parse_rows(state.get("history", []), source="graph history")
        prior = [h for h in history if h.card_token == txn.card_token and h.ts < txn.ts]
        return {
            "txn": txn.model_dump(mode="json"),
            "history": [h.model_dump(mode="json") for h in prior],
            "path": ["ingest"],
        }

    def features(state: TriageState) -> dict[str, Any]:
        f = compute(_txn(state), _history(state))
        return {"features": f.model_dump(), "path": ["features"]}

    def score_node(state: TriageState) -> dict[str, Any]:
        rs = score(Features.model_validate(state["features"]), settings)
        return {"rule_score": rs.model_dump(), "path": ["score"]}

    def route_after_score(state: TriageState) -> Literal["auto_clear", "investigate", "escalated"]:
        band = state["rule_score"]["band"]
        if band == "low":
            return "auto_clear"
        if band == "high":
            return "escalated"
        return "investigate"

    def auto_clear(state: TriageState) -> dict[str, Any]:
        rs = RuleScore.model_validate(state["rule_score"])
        decision = TriageDecision(
            verdict=Verdict.CLEAR,
            confidence=round(1 - rs.score, 3),
            rationale=f"Rule score {rs.score:.2f} is below the review threshold.",
            evidence=["rule_score"],
        )
        return {
            "decision": decision.model_dump(mode="json"),
            "final_verdict": Verdict.CLEAR.value,
            "path": ["auto_clear"],
        }

    def investigate_node(state: TriageState) -> dict[str, Any]:
        ctx = ToolContext(
            txn=_txn(state),
            history=_history(state),
            features=Features.model_validate(state["features"]),
            rule_score=RuleScore.model_validate(state["rule_score"]),
        )
        inv = investigate(llm, ctx, settings.max_tool_turns)
        return {
            "decision": inv.decision.model_dump(mode="json"),
            "final_verdict": inv.decision.verdict.value,
            "tool_calls": inv.tool_calls,
            "notes": inv.notes,
            "path": ["investigate"],
        }

    def escalated(state: TriageState) -> dict[str, Any]:
        # TODO: hand these to a human instead of just marking them
        return {"final_verdict": Verdict.BLOCK_CARD.value, "path": ["escalated"]}

    g = StateGraph(TriageState)
    g.add_node("ingest", ingest)
    g.add_node("features", features)
    g.add_node("score", score_node)
    g.add_node("auto_clear", auto_clear)
    g.add_node("investigate", investigate_node)
    g.add_node("escalated", escalated)

    g.add_edge(START, "ingest")
    g.add_edge("ingest", "features")
    g.add_edge("features", "score")
    g.add_conditional_edges("score", route_after_score)
    g.add_edge("auto_clear", END)
    g.add_edge("investigate", END)
    g.add_edge("escalated", END)
    return g.compile(checkpointer=checkpointer)
