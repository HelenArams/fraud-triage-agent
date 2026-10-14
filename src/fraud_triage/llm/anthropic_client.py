"""Claude via the Messages API, either first-party or on Amazon Bedrock.

Both clients expose the same `messages.create`, so the only difference is how
the SDK client is constructed.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Any, cast

import anthropic
from anthropic.types import MessageParam, TextBlockParam, ToolParam

from fraud_triage.llm.base import AssistantTurn, Message, ToolSpec

log = logging.getLogger(__name__)


class LLMUnavailable(RuntimeError):
    """The provider failed after the SDK's own retries."""


def _to_block(block: Any) -> dict[str, Any]:
    # rebuild the param shape by hand instead of model_dump(): response blocks
    # carry extra read-only fields we don't want to echo back
    if block.type == "text":
        return {"type": "text", "text": block.text}
    if block.type == "tool_use":
        return {"type": "tool_use", "id": block.id, "name": block.name, "input": block.input}
    return cast(dict[str, Any], block.model_dump(mode="json", exclude_none=True))


class AnthropicClient:
    def __init__(self, model: str, *, max_tokens: int = 4096, sdk: Any = None) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._sdk = sdk if sdk is not None else anthropic.Anthropic()

    def create(
        self, *, system: str, tools: Sequence[ToolSpec], messages: Sequence[Message]
    ) -> AssistantTurn:
        # Render order is tools -> system -> messages, so a single breakpoint at
        # the end of the system prompt caches the tool definitions as well. Both
        # are static across every transaction in a run.
        system_blocks: list[TextBlockParam] = [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ]
        tool_params: list[ToolParam] = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in tools
        ]
        try:
            resp = self._sdk.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=system_blocks,
                tools=tool_params,
                messages=cast(list[MessageParam], list(messages)),
            )
        except (anthropic.APIStatusError, anthropic.APIConnectionError) as e:
            raise LLMUnavailable(f"{type(e).__name__}: {e}") from e

        usage = resp.usage
        log.debug(
            "llm usage: in=%s out=%s cache_read=%s cache_write=%s",
            usage.input_tokens,
            usage.output_tokens,
            usage.cache_read_input_tokens,
            usage.cache_creation_input_tokens,
        )
        return AssistantTurn(
            content=[_to_block(b) for b in resp.content],
            stop_reason=resp.stop_reason or "end_turn",
        )


class BedrockClient(AnthropicClient):
    def __init__(
        self, model: str, *, region: str | None = None, max_tokens: int = 4096, sdk: Any = None
    ) -> None:
        # region falls back to AWS_REGION / the boto3 profile; the SDK raises if
        # it can't find one, which is what we want
        super().__init__(
            model,
            max_tokens=max_tokens,
            sdk=sdk if sdk is not None else anthropic.AnthropicBedrock(aws_region=region),
        )
