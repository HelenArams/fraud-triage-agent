from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from fraud_triage.models import Transaction

T0 = datetime(2026, 9, 1, 10, 0, tzinfo=UTC)


def make_txn(**overrides: Any) -> Transaction:
    base: dict[str, Any] = {
        "id": "t1",
        "ts": T0,
        "card_token": "tok_a",
        "masked_pan": "411111******1111",
        "amount_minor": 2500,
        "currency": "EUR",
        "mcc": "5411",
        "merchant_id": "m_grocer",
        "merchant_name": "Corner Grocer",
        "merchant_country": "DE",
        "card_country": "DE",
        "channel": "pos",
        "three_ds_result": None,
        "device_id": None,
        "ip_country": None,
    }
    base.update(overrides)
    return Transaction.model_validate(base)


def minutes(n: float) -> timedelta:
    return timedelta(minutes=n)


@pytest.fixture
def txn() -> Transaction:
    return make_txn()
