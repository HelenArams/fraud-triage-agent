"""Deterministic features over a transaction and its card history.

`history` is every earlier transaction we know about for the same card. Nothing
here is clever on purpose: these are the signals an analyst would check first,
and they need to be explainable line by line.
"""

from __future__ import annotations

import statistics
from collections.abc import Sequence
from datetime import timedelta

from pydantic import BaseModel

from fraud_triage.models import Transaction

# gambling, quasi-cash / crypto, money transfer, stored value load
RISKY_MCC: dict[str, str] = {
    "7995": "gambling",
    "7800": "gambling (lottery)",
    "7801": "gambling (online)",
    "7802": "gambling (horse/dog racing)",
    "6051": "quasi-cash / crypto",
    "4829": "money transfer",
    "6540": "stored value load",
}

# 3DS results we treat as authenticated. "A" (attempted) still shifts liability.
THREE_DS_OK = frozenset({"Y", "A"})

CARD_TESTING_MAX_MINOR = 200
CARD_TESTING_MIN_AUTHS = 4
CARD_TESTING_MIN_MERCHANTS = 3
MIN_HISTORY_FOR_ZSCORE = 3


class Features(BaseModel):
    velocity_1h: int
    velocity_24h: int
    amount_z: float
    history_size: int
    card_vs_merchant_country: bool
    card_vs_ip_country: bool
    new_device: bool
    risky_mcc: str | None
    three_ds_missing: bool
    card_testing: bool


def _window(txn: Transaction, history: Sequence[Transaction], span: timedelta) -> list[Transaction]:
    start = txn.ts - span
    return [h for h in history if start <= h.ts < txn.ts] + [txn]


def amount_zscore(txn: Transaction, history: Sequence[Transaction]) -> float:
    amounts = [h.amount_minor for h in history]
    if len(amounts) < MIN_HISTORY_FOR_ZSCORE:
        return 0.0
    mean = statistics.fmean(amounts)
    sd = statistics.pstdev(amounts)
    if sd == 0:
        # flat history: any real jump is notable, but don't divide by zero
        return 0.0 if txn.amount_minor <= mean else 10.0
    return round((txn.amount_minor - mean) / sd, 2)


def is_card_testing(txn: Transaction, history: Sequence[Transaction]) -> bool:
    recent = [
        t
        for t in _window(txn, history, timedelta(hours=1))
        if t.amount_minor <= CARD_TESTING_MAX_MINOR
    ]
    if len(recent) < CARD_TESTING_MIN_AUTHS:
        return False
    return len({t.merchant_id for t in recent}) >= CARD_TESTING_MIN_MERCHANTS


def compute(txn: Transaction, history: Sequence[Transaction]) -> Features:
    prior = [h for h in history if h.card_token == txn.card_token and h.ts < txn.ts]
    seen_devices = {h.device_id for h in prior if h.device_id}
    return Features(
        velocity_1h=len(_window(txn, prior, timedelta(hours=1))),
        velocity_24h=len(_window(txn, prior, timedelta(hours=24))),
        amount_z=amount_zscore(txn, prior),
        history_size=len(prior),
        card_vs_merchant_country=txn.merchant_country != txn.card_country,
        card_vs_ip_country=bool(txn.ip_country) and txn.ip_country != txn.card_country,
        # no history -> we can't call a device "new"
        new_device=bool(txn.device_id) and bool(prior) and txn.device_id not in seen_devices,
        risky_mcc=RISKY_MCC.get(txn.mcc),
        three_ds_missing=txn.channel == "ecom" and (txn.three_ds_result or "") not in THREE_DS_OK,
        card_testing=is_card_testing(txn, prior),
    )
