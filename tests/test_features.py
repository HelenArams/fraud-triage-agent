from __future__ import annotations

from fraud_triage.features import compute

from factories import T0, make_txn, minutes


def test_velocity_counts_current_txn_and_windows() -> None:
    hist = [
        make_txn(id="h1", ts=T0 - minutes(30)),
        make_txn(id="h2", ts=T0 - minutes(90)),
        make_txn(id="h3", ts=T0 - minutes(60 * 30)),
        make_txn(id="other", ts=T0 - minutes(5), card_token="tok_b"),
    ]
    f = compute(make_txn(), hist)
    assert f.velocity_1h == 2
    assert f.velocity_24h == 3
    assert f.history_size == 3


def test_future_txns_are_not_history() -> None:
    f = compute(make_txn(), [make_txn(id="later", ts=T0 + minutes(5))])
    assert f.history_size == 0


def test_amount_z_needs_some_history() -> None:
    hist = [make_txn(id="h1", ts=T0 - minutes(60), amount_minor=1000)]
    assert compute(make_txn(amount_minor=90000), hist).amount_z == 0.0


def test_amount_z_spike() -> None:
    hist = [
        make_txn(id=f"h{i}", ts=T0 - minutes(60 * (i + 1)), amount_minor=a)
        for i, a in enumerate([2000, 2500, 3000, 2200, 2800])
    ]
    f = compute(make_txn(amount_minor=50000), hist)
    assert f.amount_z > 3
    assert compute(make_txn(amount_minor=2500), hist).amount_z < 1


def test_amount_z_flat_history() -> None:
    hist = [make_txn(id=f"h{i}", ts=T0 - minutes(60 * (i + 1))) for i in range(4)]
    assert compute(make_txn(amount_minor=2500), hist).amount_z == 0.0
    assert compute(make_txn(amount_minor=9000), hist).amount_z == 10.0


def test_country_mismatch() -> None:
    f = compute(make_txn(merchant_country="FR", ip_country="DE"), [])
    assert f.card_vs_merchant_country
    assert not f.card_vs_ip_country
    f = compute(make_txn(ip_country="NG", channel="ecom", three_ds_result="Y"), [])
    assert f.card_vs_ip_country


def test_new_device_only_with_history() -> None:
    hist = [make_txn(id="h1", ts=T0 - minutes(60), device_id="dev-1")]
    assert compute(make_txn(device_id="dev-2"), hist).new_device
    assert not compute(make_txn(device_id="dev-1"), hist).new_device
    assert not compute(make_txn(device_id="dev-2"), []).new_device


def test_risky_mcc() -> None:
    assert compute(make_txn(mcc="7995"), []).risky_mcc == "gambling"
    assert compute(make_txn(mcc="6051"), []).risky_mcc == "quasi-cash / crypto"
    assert compute(make_txn(mcc="5411"), []).risky_mcc is None


def test_three_ds_only_matters_on_ecom() -> None:
    assert compute(make_txn(channel="ecom", three_ds_result="N"), []).three_ds_missing
    assert compute(make_txn(channel="ecom", three_ds_result=None), []).three_ds_missing
    assert not compute(make_txn(channel="ecom", three_ds_result="A"), []).three_ds_missing
    assert not compute(make_txn(channel="pos", three_ds_result=None), []).three_ds_missing


def test_card_testing_small_auths_across_merchants() -> None:
    hist = [
        make_txn(id=f"h{i}", ts=T0 - minutes(5 * (i + 1)), amount_minor=100, merchant_id=f"m{i}")
        for i in range(4)
    ]
    assert compute(make_txn(amount_minor=100, merchant_id="m9"), hist).card_testing


def test_card_testing_needs_distinct_merchants() -> None:
    hist = [
        make_txn(id=f"h{i}", ts=T0 - minutes(5 * (i + 1)), amount_minor=100, merchant_id="m_same")
        for i in range(5)
    ]
    assert not compute(make_txn(amount_minor=100, merchant_id="m_same"), hist).card_testing
