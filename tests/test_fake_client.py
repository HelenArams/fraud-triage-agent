from __future__ import annotations

from fraud_triage.llm.base import AssistantTurn, tool_result
from fraud_triage.llm.fake import FakeClient, call, turn


def test_turn_exposes_tool_calls_and_text() -> None:
    t = turn(call("feature_breakdown"), call("similar_cases", k=2), text="checking")
    assert t.text == "checking"
    assert [c.name for c in t.tool_calls] == ["feature_breakdown", "similar_cases"]
    assert t.tool_calls[1].input == {"k": 2}
    assert t.stop_reason == "tool_use"


def test_fake_client_replays_script_and_records() -> None:
    def second(messages: object) -> AssistantTurn:
        return turn(text="done")

    fake = FakeClient([turn(call("feature_breakdown")), second])
    first = fake.create(system="s", tools=[], messages=[{"role": "user", "content": "hi"}])
    msgs = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": first.content},
        {"role": "user", "content": [tool_result(first.tool_calls[0].id, "{}")]},
    ]
    assert fake.create(system="s", tools=[], messages=msgs).text == "done"
    assert len(fake.calls) == 2
    assert fake.calls[1].messages[-1]["content"][0]["type"] == "tool_result"
