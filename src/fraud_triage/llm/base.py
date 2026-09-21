"""Provider-neutral shapes for the investigate loop.

Messages and content blocks use the Messages API wire format (plain dicts), so
the Anthropic and Bedrock clients pass them straight through and the fake and
offline clients can read them without any SDK objects.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

Message = dict[str, Any]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    input: dict[str, Any]


@dataclass
class AssistantTurn:
    content: list[dict[str, Any]] = field(default_factory=list)
    stop_reason: str = "end_turn"

    @property
    def tool_calls(self) -> list[ToolCall]:
        return [
            ToolCall(id=b["id"], name=b["name"], input=dict(b.get("input") or {}))
            for b in self.content
            if b.get("type") == "tool_use"
        ]

    @property
    def text(self) -> str:
        return "".join(b.get("text", "") for b in self.content if b.get("type") == "text")


class LLMClient(Protocol):
    def create(
        self, *, system: str, tools: Sequence[ToolSpec], messages: Sequence[Message]
    ) -> AssistantTurn: ...


def tool_result(call_id: str, content: str, *, is_error: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_result", "tool_use_id": call_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block
