"""Scripted client for tests. Each step is either a ready AssistantTurn or a
function of the conversation so far."""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from fraud_triage.llm.base import AssistantTurn, Message, ToolSpec

Step = AssistantTurn | Callable[[Sequence[Message]], AssistantTurn]

_ids = itertools.count(1)


def call(name: str, **input: Any) -> dict[str, Any]:
    return {"type": "tool_use", "id": f"toolu_fake_{next(_ids)}", "name": name, "input": input}


def turn(*calls: dict[str, Any], text: str = "") -> AssistantTurn:
    blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
    blocks.extend(calls)
    return AssistantTurn(content=blocks, stop_reason="tool_use" if calls else "end_turn")


@dataclass
class RecordedCall:
    system: str
    tools: list[str]
    messages: list[Message]


@dataclass
class FakeClient:
    script: list[Step]
    calls: list[RecordedCall] = field(default_factory=list)

    def create(
        self, *, system: str, tools: Sequence[ToolSpec], messages: Sequence[Message]
    ) -> AssistantTurn:
        self.calls.append(
            RecordedCall(system, [t.name for t in tools], copy.deepcopy(list(messages)))
        )
        if not self.script:
            raise AssertionError("FakeClient script exhausted")
        step = self.script.pop(0)
        return step(messages) if callable(step) else step
