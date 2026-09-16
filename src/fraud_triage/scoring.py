"""Turn features into a 0-1 rule score with human-readable reasons.

Signals are combined noisy-OR style: score = 1 - prod(1 - w_i). Each signal can
only push the score up, nothing goes above 1, and two weak signals together
count for more than either alone without needing hand-tuned interaction terms.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel

from fraud_triage.config import Settings
from fraud_triage.features import Features

Band = Literal["low", "mid", "high"]


class Reason(BaseModel):
    code: str
    weight: float
    detail: str


class RuleScore(BaseModel):
    score: float
    band: Band
    reasons: list[Reason]


def reasons_for(f: Features) -> list[Reason]:
    r: list[Reason] = []
    if f.card_testing:
        r.append(
            Reason(
                code="card_testing", weight=0.70, detail="many small auths across merchants in 1h"
            )
        )
    if f.velocity_1h >= 5:
        r.append(
            Reason(
                code="velocity_1h", weight=0.15, detail=f"{f.velocity_1h} auths in the last hour"
            )
        )
    if f.velocity_24h >= 15:
        r.append(Reason(code="velocity_24h", weight=0.10, detail=f"{f.velocity_24h} auths in 24h"))
    if f.amount_z >= 3:
        r.append(
            Reason(code="amount_spike", weight=0.25, detail=f"amount z-score {f.amount_z:.1f}")
        )
    elif f.amount_z >= 2:
        r.append(
            Reason(code="amount_spike", weight=0.12, detail=f"amount z-score {f.amount_z:.1f}")
        )
    if f.card_vs_ip_country:
        r.append(
            Reason(
                code="ip_country_mismatch",
                weight=0.15,
                detail="IP country differs from card country",
            )
        )
    if f.card_vs_merchant_country:
        # travellers do this all the time, keep it small
        r.append(
            Reason(code="merchant_country_mismatch", weight=0.05, detail="cross-border merchant")
        )
    if f.new_device:
        r.append(
            Reason(code="new_device", weight=0.15, detail="device not seen on this card before")
        )
    if f.risky_mcc:
        r.append(Reason(code="risky_mcc", weight=0.20, detail=f"high-risk MCC ({f.risky_mcc})"))
    if f.three_ds_missing:
        r.append(
            Reason(code="three_ds_missing", weight=0.15, detail="ecom without 3DS authentication")
        )
    return r


def band_for(score: float, settings: Settings) -> Band:
    if score < settings.low_threshold:
        return "low"
    if score >= settings.high_threshold:
        return "high"
    return "mid"


def score(f: Features, settings: Settings | None = None) -> RuleScore:
    settings = settings or Settings()
    reasons = reasons_for(f)
    s = 1.0 - math.prod(1.0 - r.weight for r in reasons)
    s = round(s, 3)
    return RuleScore(score=s, band=band_for(s, settings), reasons=reasons)
