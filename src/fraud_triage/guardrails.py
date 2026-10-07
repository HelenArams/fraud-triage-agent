"""Checks between the model's decision and anything that acts on it."""

from __future__ import annotations

import re
from collections.abc import Collection
from dataclasses import dataclass, field

from fraud_triage.config import Settings
from fraud_triage.models import TriageDecision, Verdict
from fraud_triage.pan import scrub
from fraud_triage.scoring import RuleScore

# Cheap tripwire, not a classifier. The real defence is that routing never
# depends on descriptor text; this just makes the attempt visible.
_INJECTION = re.compile(
    r"ignore (all |any )?(previous|prior|above)|system (note|prompt|message)|"
    r"\b(you are|as an ai|assistant|reviewer)\b|return verdict|verdict\s*[:=]",
    re.IGNORECASE,
)


def looks_like_injection(text: str | None) -> bool:
    return bool(text and _INJECTION.search(text))


@dataclass
class Checked:
    decision: TriageDecision
    notes: list[str] = field(default_factory=list)

    @property
    def overridden(self) -> bool:
        return any(n.startswith("override") for n in self.notes)


def check(
    decision: TriageDecision,
    rule_score: RuleScore,
    settings: Settings,
    known_refs: Collection[str] | None = None,
    merchant_text: str | None = None,
) -> Checked:
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

    if known_refs is not None:
        unknown = [e for e in evidence if e not in known_refs]
        if unknown:
            notes.append(f"dropped unknown evidence refs: {', '.join(unknown)}")
            evidence = [e for e in evidence if e in known_refs]
    if not evidence and verdict != Verdict.REVIEW:
        notes.append(f"override {verdict} -> REVIEW: rationale has no usable evidence references")
        verdict = Verdict.REVIEW

    if looks_like_injection(merchant_text):
        notes.append(
            "merchant descriptor contains instruction-like text (possible prompt injection)"
        )
        if verdict == Verdict.CLEAR:
            notes.append(
                "override CLEAR -> REVIEW: not auto-clearing a txn with a suspicious descriptor"
            )
            verdict = Verdict.REVIEW

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
