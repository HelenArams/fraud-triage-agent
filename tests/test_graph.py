from __future__ import annotations

from typing import Any

from fraud_triage.config import Settings
from fraud_triage.graph import build_graph
from fraud_triage.llm.fake import FakeClient, call, turn
from fraud_triage.llm.offline import OfflineAnalyst
from fraud_triage.runner import open_checkpointer, thread_config

from factories import ato_case, card_testing_case, make_txn, state_for


def run(llm: Any, state: dict[str, Any], tmp_path: Any, thread: str = "t") -> dict[str, Any]:
    graph = build_graph(llm, Settings(), open_checkpointer(tmp_path / "cp.sqlite"))
    graph.invoke(state, thread_config(thread))
    return dict(graph.get_state(thread_config(thread)).values)


def test_low_risk_auto_clears_without_llm(tmp_path: Any) -> None:
    fake = FakeClient([])
    out = run(fake, state_for(make_txn()), tmp_path)
    assert out["path"] == ["ingest", "features", "score", "auto_clear", "report"]
    assert out["final_verdict"] == "CLEAR"
    assert fake.calls == []


def test_mid_risk_goes_to_investigate(tmp_path: Any) -> None:
    txn, hist = ato_case()
    fake = FakeClient(
        [
            turn(call("feature_breakdown"), call("card_history", limit=3)),
            turn(
                call(
                    "submit_decision",
                    verdict="REVIEW",
                    confidence=0.7,
                    rationale="New device from abroad and a big jump in amount (feature:new_device).",
                    evidence=["feature:new_device", "rule_score"],
                )
            ),
        ]
    )
    out = run(fake, state_for(txn, hist), tmp_path)
    assert out["path"][3:] == ["investigate", "guardrails", "finalize", "report"]
    assert out["final_verdict"] == "REVIEW"
    assert out["tool_calls"] == ["feature_breakdown", "card_history", "submit_decision"]
    # second request carries both tool results back in one user message
    last = fake.calls[1].messages[-1]
    assert [b["type"] for b in last["content"]] == ["tool_result", "tool_result"]


def test_ingest_drops_other_cards_and_future_history(tmp_path: Any) -> None:
    txn, hist = ato_case()
    noise = [
        make_txn(id="other", card_token="tok_zz"),
        make_txn(id="later", ts=txn.ts.replace(year=2027)),
    ]
    out = run(OfflineAnalyst(), state_for(txn, hist + noise), tmp_path)
    assert {h["id"] for h in out["history"]} == {h.id for h in hist}


def test_high_risk_skips_the_llm_and_waits_for_a_human(tmp_path: Any) -> None:
    txn, hist = card_testing_case()
    fake = FakeClient([])
    graph = build_graph(fake, Settings(), open_checkpointer(tmp_path / "cp.sqlite"))
    graph.invoke(state_for(txn, hist), thread_config("t"))
    snap = graph.get_state(thread_config("t"))
    assert snap.next == ("human_review",)
    assert snap.values["rule_score"]["band"] == "high"
    assert snap.values["decision"]["verdict"] == "BLOCK_CARD"
    assert "feature:card_testing" in snap.values["decision"]["evidence"]
    [intr] = snap.interrupts
    assert intr.value["txn_id"] == "t_ct"
    assert "final_verdict" not in snap.values
    assert fake.calls == []


def test_block_from_investigate_goes_to_human(tmp_path: Any) -> None:
    txn, hist = ato_case()
    graph = build_graph(OfflineAnalyst(), Settings(), open_checkpointer(tmp_path / "cp.sqlite"))
    graph.invoke(state_for(txn, hist), thread_config("t"))
    snap = graph.get_state(thread_config("t"))
    assert snap.values["decision"]["verdict"] == "BLOCK_CARD"
    assert snap.next == ("human_review",)


def test_investigate_recovers_from_bad_tool_and_bad_decision(tmp_path: Any) -> None:
    txn, hist = ato_case()
    fake = FakeClient(
        [
            turn(call("lookup_ssn")),
            turn(call("submit_decision", verdict="MAYBE", confidence=2, rationale="", evidence=[])),
            turn(text="I think it's fine."),
            turn(
                call(
                    "submit_decision",
                    verdict="REVIEW",
                    confidence=0.5,
                    rationale="Rule score is mid (rule_score).",
                    evidence=["rule_score"],
                )
            ),
        ]
    )
    out = run(fake, state_for(txn, hist), tmp_path)
    assert out["decision"]["verdict"] == "REVIEW"
    assert any("unknown tool" in n for n in out["notes"])
    assert any("failed validation" in n for n in out["notes"])


def test_investigate_gives_up_after_max_turns(tmp_path: Any) -> None:
    txn, hist = ato_case()
    fake = FakeClient([turn(call("feature_breakdown")) for _ in range(10)])
    graph = build_graph(fake, Settings(max_tool_turns=3), open_checkpointer(tmp_path / "cp.sqlite"))
    graph.invoke(state_for(txn, hist), thread_config("x"))
    out = graph.get_state(thread_config("x")).values
    assert out["decision"]["verdict"] == "REVIEW"
    assert len(fake.calls) == 3
