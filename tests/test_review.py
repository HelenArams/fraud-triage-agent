"""Interrupt in one process, resume in another - which is how the CLI uses it."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from fraud_triage.config import Settings
from fraud_triage.graph import build_graph
from fraud_triage.llm.offline import OfflineAnalyst
from fraud_triage.models import ReviewAnswer, Verdict
from fraud_triage.runner import open_checkpointer, resume, triage_all

from factories import ato_case, make_txn


def fresh_graph(db: Path) -> Any:
    return build_graph(OfflineAnalyst(), Settings(), open_checkpointer(db))


def test_interrupt_then_resume_from_new_graph(tmp_path: Path) -> None:
    db = tmp_path / "cp.sqlite"
    txn, hist = ato_case()
    outcomes = triage_all(fresh_graph(db), [*hist, txn])
    o = outcomes[-1]
    assert o.pending and o.verdict is None

    # "new process": a brand new graph over the same sqlite file
    done = resume(
        fresh_graph(db), o.thread_id, ReviewAnswer(approve=True, note="called the cardholder")
    )
    assert not done.pending
    assert done.verdict == "BLOCK_CARD"
    assert done.path[-1] == "human_review"

    values = fresh_graph(db).get_state({"configurable": {"thread_id": o.thread_id}}).values
    assert values["review"]["note"] == "called the cardholder"


def test_reject_clears_and_override_wins(tmp_path: Path) -> None:
    db = tmp_path / "cp.sqlite"
    txn, hist = ato_case()
    triage_all(fresh_graph(db), [*hist, txn])
    done = resume(fresh_graph(db), txn.id, ReviewAnswer(approve=False, note="customer travelling"))
    assert done.verdict == "CLEAR"

    assert (
        ReviewAnswer(approve=False, verdict=Verdict.REVIEW).final_verdict(Verdict.BLOCK_CARD)
        == "REVIEW"
    )


def test_resume_on_finished_thread_fails(tmp_path: Path) -> None:
    db = tmp_path / "cp.sqlite"
    triage_all(fresh_graph(db), [make_txn(id="t_ok")])
    with pytest.raises(LookupError):
        resume(fresh_graph(db), "t_ok", ReviewAnswer(approve=True))
