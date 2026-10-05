from __future__ import annotations

from fraud_triage.config import Settings
from fraud_triage.guardrails import check
from fraud_triage.models import TriageDecision, Verdict
from fraud_triage.scoring import RuleScore


def rs(score: float) -> RuleScore:
    return RuleScore(score=score, band="mid", reasons=[])


def decision(verdict: str = "CLEAR", **kw: object) -> TriageDecision:
    base: dict[str, object] = {
        "verdict": verdict,
        "confidence": 0.9,
        "rationale": "Looks like normal spend (rule_score).",
        "evidence": ["rule_score"],
    }
    base.update(kw)
    return TriageDecision.model_validate(base)


def test_clear_above_ceiling_is_overridden() -> None:
    out = check(decision("CLEAR"), rs(0.62), Settings())
    assert out.decision.verdict == Verdict.REVIEW
    assert out.decision.confidence == 0.5
    assert out.overridden


def test_clear_below_ceiling_passes() -> None:
    out = check(decision("CLEAR"), rs(0.35), Settings())
    assert out.decision.verdict == Verdict.CLEAR
    assert out.notes == []


def test_pan_is_scrubbed_from_rationale() -> None:
    d = decision("REVIEW", rationale="Card 4111 1111 1111 1111 used abroad (rule_score).")
    out = check(d, rs(0.4), Settings())
    assert "4111 1111" not in out.decision.rationale
    assert "411111******1111" in out.decision.rationale
    assert any("scrubbed" in n for n in out.notes)
