"""Glue between the CLI and the graph: one thread per transaction."""

from __future__ import annotations

import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from fraud_triage.graph import TriageState
from fraud_triage.models import ReviewAnswer, Transaction


@dataclass(frozen=True)
class Outcome:
    txn_id: str
    thread_id: str
    score: float
    band: str
    path: list[str]
    verdict: str | None
    pending: bool = False


def open_checkpointer(path: str | Path) -> SqliteSaver:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    # langgraph may touch the connection from a worker thread
    conn = sqlite3.connect(str(path), check_same_thread=False)
    return SqliteSaver(conn)


def thread_config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def outcome_from_state(thread_id: str, values: dict[str, Any], pending: bool = False) -> Outcome:
    rs = values.get("rule_score", {})
    return Outcome(
        txn_id=values["txn"]["id"],
        thread_id=thread_id,
        score=rs.get("score", 0.0),
        band=rs.get("band", "?"),
        path=list(values.get("path", [])),
        verdict=values.get("final_verdict"),
        pending=pending,
    )


def triage_all(
    graph: Any, txns: Sequence[Transaction], out_dir: str | None = None
) -> list[Outcome]:
    outcomes = []
    for txn in txns:
        same_card = [
            t.model_dump(mode="json")
            for t in txns
            if t.card_token == txn.card_token and t.ts < txn.ts
        ]
        state: TriageState = {
            "txn": txn.model_dump(mode="json"),
            "history": same_card,
            "out_dir": out_dir,
        }
        cfg = thread_config(txn.id)
        graph.invoke(state, cfg)
        snap = graph.get_state(cfg)
        outcomes.append(outcome_from_state(txn.id, snap.values, pending=bool(snap.next)))
    return outcomes


def resume(graph: Any, thread_id: str, answer: ReviewAnswer) -> Outcome:
    cfg = thread_config(thread_id)
    snap = graph.get_state(cfg)
    if "human_review" not in snap.next:
        raise LookupError(f"thread {thread_id!r} is not waiting for review")
    graph.invoke(Command(resume=answer.model_dump(mode="json")), cfg)
    snap = graph.get_state(cfg)
    return outcome_from_state(thread_id, snap.values, pending=bool(snap.next))
