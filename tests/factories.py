"""Small builders shared by the tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

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


def state_for(txn: Transaction, history: list[Transaction] | None = None) -> dict[str, Any]:
    return {
        "txn": txn.model_dump(mode="json"),
        "history": [h.model_dump(mode="json") for h in history or []],
    }


def ato_case() -> tuple[Transaction, list[Transaction]]:
    """New device from abroad, no 3DS, big amount: lands in the mid band."""
    known = {"channel": "ecom", "three_ds_result": "Y", "device_id": "dev-1", "ip_country": "DE"}
    hist = [
        make_txn(id=f"h{i}", ts=T0 - timedelta(days=i + 1), amount_minor=a, **known)
        for i, a in enumerate((2000, 2500, 3000, 2200))
    ]
    txn = make_txn(
        id="t_ato",
        channel="ecom",
        three_ds_result="N",
        device_id="dev-x",
        ip_country="RO",
        amount_minor=99000,
        merchant_id="m_curry_online",
    )
    return txn, hist


def card_testing_case() -> tuple[Transaction, list[Transaction]]:
    """Six tiny auths across merchants in a few minutes: high band."""
    kw = {"channel": "ecom", "three_ds_result": "N", "ip_country": "BR", "amount_minor": 100}
    hist = [
        make_txn(id=f"h{i}", ts=T0 - minutes(3 * (i + 1)), merchant_id=f"m{i}", **kw)
        for i in range(5)
    ]
    return make_txn(id="t_ct", merchant_id="m9", **kw), hist
