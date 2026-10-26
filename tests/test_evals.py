from __future__ import annotations

from pathlib import Path

from fraud_triage.config import Settings
from fraud_triage.evals import load_cases, main, run_case
from fraud_triage.llm.fake import FakeClient, call, turn

CASES = Path(__file__).resolve().parents[1] / "evals" / "cases.yaml"


def test_cases_file_parses_with_defaults() -> None:
    cases = load_cases(CASES)
    assert {c.name for c in cases} >= {"card_testing", "injection_in_merchant_name", "clean"}
    assert all(t["currency"] == "EUR" for c in cases for t in c.txns)


def test_offline_eval_passes() -> None:
    assert main([str(CASES)]) == 0


def test_eval_reports_mismatch() -> None:
    [ato] = [c for c in load_cases(CASES) if c.name == "account_takeover"]
    lazy = FakeClient(
        [
            turn(
                call(
                    "submit_decision",
                    verdict="REVIEW",
                    confidence=0.4,
                    rationale="Not sure (rule_score).",
                    evidence=["rule_score"],
                )
            )
        ]
    )
    r = run_case(ato, lazy, Settings())
    assert not r.ok
    assert "verdict REVIEW != BLOCK_CARD" in r.failures
