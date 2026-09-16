from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

Provider = Literal["anthropic", "bedrock", "offline"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="FRAUD_", env_file=".env", extra="ignore")

    provider: Provider = "offline"
    model: str = "claude-sonnet-4-5"
    bedrock_model: str = "us.anthropic.claude-sonnet-4-5-20250929-v1:0"
    aws_region: str | None = None
    max_tokens: int = 4096
    max_tool_turns: int = 6

    # score bands: < low -> auto clear, >= high -> straight to a human
    low_threshold: float = Field(default=0.30, ge=0, le=1)
    high_threshold: float = Field(default=0.80, ge=0, le=1)
    # the analyst may never CLEAR at or above this score
    clear_ceiling: float = Field(default=0.50, ge=0, le=1)
