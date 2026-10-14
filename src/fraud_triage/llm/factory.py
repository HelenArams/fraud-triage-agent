from __future__ import annotations

from fraud_triage.config import Settings
from fraud_triage.llm.base import LLMClient
from fraud_triage.llm.offline import OfflineAnalyst


def get_llm(settings: Settings) -> LLMClient:
    match settings.provider:
        case "offline":
            return OfflineAnalyst()
        case "anthropic":
            from fraud_triage.llm.anthropic_client import AnthropicClient

            return AnthropicClient(settings.model, max_tokens=settings.max_tokens)
        case "bedrock":
            from fraud_triage.llm.anthropic_client import BedrockClient

            return BedrockClient(
                settings.bedrock_model, region=settings.aws_region, max_tokens=settings.max_tokens
            )
