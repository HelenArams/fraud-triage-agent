"""Checks between the model's decision and anything that acts on it."""

from __future__ import annotations

from dataclasses import dataclass, field

from fraud_triage.config import Settings
from fraud_triage.models import TriageDecision, Verdict
from fraud_triage.pan import scrub
from fraud_triage.scoring import RuleScore


@dataclass
class Checked:
    decision: TriageDecision
    notes: list[str] = field(default_factory=list)

    @property
    def overridden(self) -> bool:
        return any(n.startswith("override") for n in self.notes)


def check(decision: TriageDecision, rule_score: RuleScore, settings: Settings) -> Checked:
    notes: list[str] = []
    verdict = decision.verdict
    confidence = decision.confidence

    rationale, n = scrub(decision.rationale)
    evidence: list[str] = []
    for ref in decision.evidence:
        clean, m = scrub(ref)
        n += m
        evidence.append(clean)
    if n:
        notes.append(f"scrubbed {n} card number(s) from model output")

    if verdict == Verdict.CLEAR and rule_score.score >= settings.clear_ceiling:
        notes.append(
            f"override CLEAR -> REVIEW: rule score {rule_score.score:.2f} is above the "
            f"auto-clear ceiling {settings.clear_ceiling:.2f}"
        )
        verdict = Verdict.REVIEW

    if verdict != decision.verdict:
        confidence = min(confidence, 0.5)
    return Checked(
        TriageDecision(
            verdict=verdict, confidence=confidence, rationale=rationale, evidence=evidence
        ),
        notes,
    )
