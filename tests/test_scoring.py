from __future__ import annotations

import pytest

from fraud_triage.config import Settings
from fraud_triage.features import Features
from fraud_triage.scoring import band_for, score


def feats(**kw: object) -> Features:
    base: dict[str, object] = {
        "velocity_1h": 1,
        "velocity_24h": 1,
        "amount_z": 0.0,
        "history_size": 5,
        "card_vs_merchant_country": False,
        "card_vs_ip_country": False,
        "new_device": False,
        "risky_mcc": None,
        "three_ds_missing": False,
        "card_testing": False,
    }
    base.update(kw)
    return Features.model_validate(base)


def test_clean_txn_scores_zero() -> None:
    rs = score(feats())
    assert rs.score == 0.0
    assert rs.band == "low"
    assert rs.reasons == []


def test_single_signal_is_its_weight() -> None:
    assert score(feats(risky_mcc="gambling")).score == pytest.approx(0.2)


def test_noisy_or_combination() -> None:
    rs = score(feats(new_device=True, card_vs_ip_country=True, three_ds_missing=True, amount_z=4.0))
    # 1 - .85 * .85 * .85 * .75
    assert rs.score == pytest.approx(0.539, abs=1e-3)
    assert rs.band == "mid"
    assert {r.code for r in rs.reasons} == {
        "new_device",
        "ip_country_mismatch",
        "three_ds_missing",
        "amount_spike",
    }


def test_card_testing_burst_is_high() -> None:
    rs = score(
        feats(card_testing=True, velocity_1h=6, three_ds_missing=True, card_vs_ip_country=True)
    )
    assert rs.band == "high"


def test_score_never_exceeds_one() -> None:
    rs = score(
        feats(
            card_testing=True,
            velocity_1h=20,
            velocity_24h=40,
            amount_z=9,
            card_vs_ip_country=True,
            card_vs_merchant_country=True,
            new_device=True,
            risky_mcc="gambling",
            three_ds_missing=True,
        )
    )
    assert 0.9 < rs.score <= 1.0


@pytest.mark.parametrize(
    ("s", "band"),
    [(0.0, "low"), (0.299, "low"), (0.3, "mid"), (0.79, "mid"), (0.8, "high"), (1.0, "high")],
)
def test_band_thresholds(s: float, band: str) -> None:
    assert band_for(s, Settings()) == band


def test_thresholds_come_from_settings() -> None:
    tight = Settings(low_threshold=0.1, high_threshold=0.5)
    assert score(feats(risky_mcc="gambling"), tight).band == "mid"
