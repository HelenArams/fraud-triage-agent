from __future__ import annotations

from fraud_triage.config import Settings
from fraud_triage.llm.base import LLMClient
from fraud_triage.llm.offline import OfflineAnalyst


def get_llm(settings: Settings) -> LLMClient:
    match settings.provider:
        case "offline":
            return OfflineAnalyst()
        case _:
            raise NotImplementedError(f"provider {settings.provider!r} not wired up yet")
