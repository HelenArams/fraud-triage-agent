from __future__ import annotations

import json
from pathlib import Path

import pytest

from fraud_triage.loader import LoaderError, load_transactions
from fraud_triage.pan import luhn_ok, scrub

HEADER = (
    "id,ts,card_token,masked_pan,amount_minor,currency,mcc,merchant_id,merchant_name,"
    "merchant_country,card_country,channel,three_ds_result,device_id,ip_country"
)


def write_csv(tmp_path: Path, *rows: str, header: str = HEADER) -> Path:
    p = tmp_path / "t.csv"
    p.write_text("\n".join([header, *rows]) + "\n")
    return p


def test_luhn() -> None:
    assert luhn_ok("4111111111111111")
    assert luhn_ok("5555555555554444")
    assert not luhn_ok("4111111111111112")
    assert not luhn_ok("abc")


def test_scrub_masks_spaced_and_plain_pans() -> None:
    text, n = scrub("card 4111 1111 1111 1111 and 5555555555554444, ref 1234567890123")
    assert n == 2
    assert "411111******1111" in text
    assert "555555******4444" in text
    # 13 digits but fails Luhn -> left alone
    assert "1234567890123" in text


def test_scrub_ignores_timestamps_and_amounts() -> None:
    text, n = scrub("2026-09-01T10:00:00Z amount 120000")
    assert n == 0
    assert text == "2026-09-01T10:00:00Z amount 120000"


def test_loader_masks_full_pan(tmp_path: Path) -> None:
    p = write_csv(
        tmp_path,
        "t1,2026-09-01T10:00:00Z,tok_a,4111111111111111,2500,eur,5411,m1,Shop,DE,DE,pos,,,",
    )
    [t] = load_transactions(p)
    assert t.masked_pan == "411111******1111"
    assert t.currency == "EUR"
    assert t.device_id is None


def test_loader_masks_pan_hidden_in_descriptor(tmp_path: Path) -> None:
    p = write_csv(
        tmp_path,
        "t1,2026-09-01T10:00:00Z,tok_a,411111******1111,2500,EUR,5411,m1,"
        "REFUND 5555555555554444,DE,DE,pos,,,",
    )
    [t] = load_transactions(p)
    assert t.merchant_name == "REFUND 555555******4444"


def test_loader_rejects_unknown_columns(tmp_path: Path) -> None:
    p = write_csv(
        tmp_path,
        "t1,2026-09-01T10:00:00Z,tok_a,411111******1111,2500,EUR,5411,m1,Shop,DE,DE,pos,,,,123",
        header=HEADER + ",cvv",
    )
    with pytest.raises(LoaderError, match="cvv"):
        load_transactions(p)


def test_loader_reports_bad_row(tmp_path: Path) -> None:
    p = write_csv(
        tmp_path,
        "t1,2026-09-01T10:00:00Z,tok_a,411111******1111,-5,EUR,5411,m1,Shop,DE,DE,pos,,,",
    )
    with pytest.raises(LoaderError, match="row 1"):
        load_transactions(p)


def test_loader_json_sorted_by_ts(tmp_path: Path) -> None:
    rows = [
        {
            "id": f"t{i}",
            "ts": ts,
            "card_token": "tok_a",
            "masked_pan": "411111******1111",
            "amount_minor": 100,
            "currency": "EUR",
            "mcc": "5411",
            "merchant_id": "m1",
            "merchant_country": "DE",
            "card_country": "DE",
            "channel": "pos",
        }
        for i, ts in [(1, "2026-09-01T12:00:00Z"), (2, "2026-09-01T09:00:00Z")]
    ]
    p = tmp_path / "t.json"
    p.write_text(json.dumps(rows))
    assert [t.id for t in load_transactions(p)] == ["t2", "t1"]
