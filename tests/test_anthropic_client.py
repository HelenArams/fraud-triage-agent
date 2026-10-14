"""The real clients, pointed at a stand-in SDK object. No network."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import anthropic
import httpx2
import pytest
from anthropic.types import Message

from fraud_triage.config import Settings
from fraud_triage.llm.anthropic_client import AnthropicClient, BedrockClient, LLMUnavailable
from fraud_triage.llm.factory import get_llm
from fraud_triage.llm.offline import OfflineAnalyst
from fraud_triage.tools import TOOL_SPECS


def message(*content: dict[str, Any], stop_reason: str = "tool_use") -> Message:
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-5",
            "content": list(content),
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 1200,
                "output_tokens": 80,
                "cache_read_input_tokens": 1100,
                "cache_creation_input_tokens": 0,
            },
        }
    )


class StubSDK:
    def __init__(self, response: Message | Exception) -> None:
        self.kwargs: dict[str, Any] = {}
        self._response = response
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> Message:
        self.kwargs = kwargs
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


def test_request_shape_and_cache_breakpoint() -> None:
    sdk = StubSDK(
        message(
            {"type": "text", "text": "Checking the breakdown first."},
            {"type": "tool_use", "id": "toolu_1", "name": "feature_breakdown", "input": {}},
        )
    )
    client = AnthropicClient("claude-sonnet-4-5", sdk=sdk)
    turn = client.create(
        system="sys", tools=TOOL_SPECS, messages=[{"role": "user", "content": "hi"}]
    )

    assert sdk.kwargs["model"] == "claude-sonnet-4-5"
    assert sdk.kwargs["system"] == [
        {"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}
    ]
    assert [t["name"] for t in sdk.kwargs["tools"]] == [t.name for t in TOOL_SPECS]
    assert "temperature" not in sdk.kwargs

    assert turn.text == "Checking the breakdown first."
    [c] = turn.tool_calls
    assert (c.id, c.name, c.input) == ("toolu_1", "feature_breakdown", {})
    # blocks go back into history as plain params
    assert turn.content[1] == {
        "type": "tool_use",
        "id": "toolu_1",
        "name": "feature_breakdown",
        "input": {},
    }


def test_api_errors_become_llm_unavailable() -> None:
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    sdk = StubSDK(anthropic.APIConnectionError(request=req))
    with pytest.raises(LLMUnavailable):
        AnthropicClient("m", sdk=sdk).create(system="s", tools=[], messages=[])


def test_bedrock_client_uses_same_path() -> None:
    sdk = StubSDK(message({"type": "text", "text": "ok"}, stop_reason="end_turn"))
    client = BedrockClient("us.anthropic.claude-sonnet-4-5-20250929-v1:0", sdk=sdk)
    assert client.create(system="s", tools=[], messages=[]).text == "ok"
    assert sdk.kwargs["model"].startswith("us.anthropic.")


def test_factory(monkeypatch: pytest.MonkeyPatch) -> None:
    assert isinstance(get_llm(Settings(provider="offline")), OfflineAnalyst)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-not-real")
    c = get_llm(Settings(provider="anthropic", model="claude-sonnet-4-5"))
    assert isinstance(c, AnthropicClient)
    assert c.model == "claude-sonnet-4-5"
    b = get_llm(Settings(provider="bedrock", aws_region="eu-central-1"))
    assert isinstance(b, BedrockClient)


def test_provider_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FRAUD_PROVIDER", "bedrock")
    monkeypatch.setenv("FRAUD_BEDROCK_MODEL", "eu.anthropic.claude-sonnet-4-5-20250929-v1:0")
    s = Settings()
    assert s.provider == "bedrock"
    assert s.bedrock_model.startswith("eu.")
