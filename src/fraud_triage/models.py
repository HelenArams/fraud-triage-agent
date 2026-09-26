from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

Channel = Literal["ecom", "pos", "atm"]


class Transaction(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str
    ts: datetime
    card_token: str
    masked_pan: str
    amount_minor: int = Field(ge=0)
    currency: str = Field(min_length=3, max_length=3)
    mcc: str = Field(pattern=r"^\d{4}$")
    merchant_id: str
    # free text from the acquirer descriptor - never trusted, see guardrails
    merchant_name: str | None = None
    merchant_country: str = Field(min_length=2, max_length=2)
    card_country: str = Field(min_length=2, max_length=2)
    channel: Channel
    three_ds_result: str | None = None
    device_id: str | None = None
    ip_country: str | None = None

    @field_validator("merchant_name", "three_ds_result", "device_id", "ip_country", mode="before")
    @classmethod
    def _blank_to_none(cls, v: object) -> object:
        if isinstance(v, str) and not v.strip():
            return None
        return v

    @field_validator("currency", "merchant_country", "card_country", "ip_country")
    @classmethod
    def _upper(cls, v: str | None) -> str | None:
        return v.upper() if v else v


class Verdict(StrEnum):
    CLEAR = "CLEAR"
    REVIEW = "REVIEW"
    BLOCK_CARD = "BLOCK_CARD"


class TriageDecision(BaseModel):
    verdict: Verdict
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = Field(min_length=1)
    # refs like "feature:new_device", "case:C-104", "txn:t_0012"
    evidence: list[str] = Field(default_factory=list)


class ReviewAnswer(BaseModel):
    """What the analyst sends back when resuming a thread parked in human_review."""

    approve: bool
    note: str = ""
    # override the proposed verdict; default is proposed-if-approved, CLEAR if rejected
    verdict: Verdict | None = None
    analyst: str = "analyst"

    def final_verdict(self, proposed: Verdict) -> Verdict:
        if self.verdict is not None:
            return self.verdict
        return proposed if self.approve else Verdict.CLEAR
