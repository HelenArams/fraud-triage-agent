from __future__ import annotations

import json
from pathlib import Path

from fraud_triage.config import Settings
from fraud_triage.graph import build_graph
from fraud_triage.llm.offline import OfflineAnalyst
from fraud_triage.models import ReviewAnswer
from fraud_triage.report import to_markdown
from fraud_triage.runner import open_checkpointer, resume, thread_config, triage_all

from factories import ato_case, make_txn


def test_auto_clear_report(tmp_path: Path) -> None:
    graph = build_graph(OfflineAnalyst(), Settings(), open_checkpointer(tmp_path / "cp.sqlite"))
    triage_all(graph, [make_txn(id="t_ok")], str(tmp_path / "out"))
    md = (tmp_path / "out" / "t_ok.md").read_text()
    assert md.startswith("# t_ok: CLEAR")
    assert "- none" in md
    data = json.loads((tmp_path / "out" / "t_ok.json").read_text())
    assert data["final_verdict"] == "CLEAR"
    assert data["card"] == "411111******1111"


def test_reviewed_report_has_decision_and_review(tmp_path: Path) -> None:
    db, out = tmp_path / "cp.sqlite", tmp_path / "out"
    txn, hist = ato_case()

    def graph() -> object:
        return build_graph(OfflineAnalyst(), Settings(), open_checkpointer(db))

    triage_all(graph(), [*hist, txn], str(out))
    assert not (out / "t_ato.md").exists()  # still waiting on a human

    resume(graph(), "t_ato", ReviewAnswer(approve=True, note="cardholder denies", analyst="helen"))
    md = (out / "t_ato.md").read_text()
    assert md.startswith("# t_ato: BLOCK_CARD")
    assert "`new_device`" in md
    assert "## Decision" in md
    assert "Tools used: feature_breakdown, similar_cases, submit_decision" in md
    assert "- helen: approved" in md
    assert "- note: cardholder denies" in md


def test_markdown_lists_guardrail_notes(tmp_path: Path) -> None:
    graph = build_graph(OfflineAnalyst(), Settings(), open_checkpointer(tmp_path / "cp.sqlite"))
    triage_all(graph, [make_txn(id="t_ok")])
    state = dict(graph.get_state(thread_config("t_ok")).values)
    state["guardrail_notes"] = ["override CLEAR -> REVIEW: something"]
    assert "## Guardrails\n\n- override CLEAR -> REVIEW: something" in to_markdown(state)
