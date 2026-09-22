"""Tools the investigating analyst (Claude or the offline stand-in) can call.

Everything is read-only and scoped to the transaction under review: there is no
way to ask for another card's history, and merchant descriptor text never comes
back through a tool (only our own merchant reference data does).
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Any

from pydantic import BaseModel

from fraud_triage.features import Features
from fraud_triage.llm.base import ToolSpec
from fraud_triage.models import Transaction, TriageDecision, Verdict
from fraud_triage.scoring import RuleScore


class Case(BaseModel):
    id: str
    outcome: str
    verdict: Verdict
    signals: list[str]
    summary: str


class ToolError(Exception):
    pass


@cache
def case_library() -> tuple[Case, ...]:
    raw = resources.files("fraud_triage.data").joinpath("cases.json").read_text()
    return tuple(Case.model_validate(c) for c in json.loads(raw))


@cache
def merchant_reference() -> dict[str, dict[str, Any]]:
    raw = resources.files("fraud_triage.data").joinpath("merchants.json").read_text()
    data: dict[str, dict[str, Any]] = json.loads(raw)
    return data


@dataclass(frozen=True)
class ToolContext:
    txn: Transaction
    history: Sequence[Transaction]
    features: Features
    rule_score: RuleScore

    @property
    def signal_codes(self) -> set[str]:
        return {r.code for r in self.rule_score.reasons}

    def known_refs(self) -> set[str]:
        refs = {"rule_score", f"txn:{self.txn.id}", f"merchant:{self.txn.merchant_id}"}
        refs |= {f"txn:{h.id}" for h in self.history}
        refs |= {f"feature:{name}" for name in Features.model_fields}
        refs |= {f"feature:{code}" for code in self.signal_codes}
        refs |= {f"case:{c.id}" for c in case_library()}
        return refs


def _txn_summary(t: Transaction) -> dict[str, Any]:
    # no merchant_name on purpose - descriptors are untrusted input
    return {
        "id": t.id,
        "ts": t.ts.isoformat(),
        "amount": f"{t.amount_minor / 100:.2f} {t.currency}",
        "mcc": t.mcc,
        "merchant_id": t.merchant_id,
        "merchant_country": t.merchant_country,
        "channel": t.channel,
        "three_ds": t.three_ds_result,
        "device_id": t.device_id,
        "ip_country": t.ip_country,
    }


def card_history(ctx: ToolContext, limit: int = 10) -> list[dict[str, Any]]:
    limit = max(1, min(int(limit), 50))
    return [_txn_summary(t) for t in sorted(ctx.history, key=lambda t: t.ts, reverse=True)[:limit]]


def merchant_profile(ctx: ToolContext, merchant_id: str) -> dict[str, Any]:
    seen = sum(1 for h in ctx.history if h.merchant_id == merchant_id)
    ref = merchant_reference().get(merchant_id)
    if ref is None:
        return {"merchant_id": merchant_id, "known": False, "seen_on_this_card": seen}
    return {"merchant_id": merchant_id, "known": True, "seen_on_this_card": seen, **ref}


def similar_cases(ctx: ToolContext, k: int = 3) -> list[dict[str, Any]]:
    mine = ctx.signal_codes
    if not mine:
        return []
    scored = []
    for case in case_library():
        theirs = set(case.signals)
        jaccard = len(mine & theirs) / len(mine | theirs)
        if jaccard > 0:
            scored.append((jaccard, case))
    scored.sort(key=lambda x: (-x[0], x[1].id))
    return [
        {
            "case_id": c.id,
            "similarity": round(j, 2),
            "outcome": c.outcome,
            "verdict": c.verdict,
            "summary": c.summary,
        }
        for j, c in scored[: max(1, min(int(k), 5))]
    ]


def feature_breakdown(ctx: ToolContext) -> dict[str, Any]:
    return {"features": ctx.features.model_dump(), "rule_score": ctx.rule_score.model_dump()}


TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="feature_breakdown",
        description="Computed features for the transaction under review and the rule score with the reasons that fired.",
        input_schema={"type": "object", "properties": {}, "additionalProperties": False},
    ),
    ToolSpec(
        name="card_history",
        description="Earlier transactions on the same card, newest first. Only this card is available.",
        input_schema={
            "type": "object",
            "properties": {"limit": {"type": "integer", "minimum": 1, "maximum": 50}},
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="merchant_profile",
        description="Reference data for a merchant id (country, MCC, onboarding date, chargeback rate in bps).",
        input_schema={
            "type": "object",
            "properties": {"merchant_id": {"type": "string"}},
            "required": ["merchant_id"],
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="similar_cases",
        description="Past analyst cases whose fired signals overlap with this transaction, with their outcome.",
        input_schema={
            "type": "object",
            "properties": {"k": {"type": "integer", "minimum": 1, "maximum": 5}},
            "additionalProperties": False,
        },
    ),
    ToolSpec(
        name="submit_decision",
        description=(
            "Submit the final triage decision. Call exactly once when done. Every claim in the "
            "rationale must be backed by an evidence reference such as feature:new_device, "
            "case:C-103, txn:t_0012, merchant:m_x or rule_score."
        ),
        input_schema=TriageDecision.model_json_schema(),
    ),
]


def run_tool(ctx: ToolContext, name: str, args: dict[str, Any]) -> str:
    try:
        match name:
            case "feature_breakdown":
                out: Any = feature_breakdown(ctx)
            case "card_history":
                out = card_history(ctx, **args)
            case "merchant_profile":
                out = merchant_profile(ctx, **args)
            case "similar_cases":
                out = similar_cases(ctx, **args)
            case _:
                raise ToolError(f"unknown tool {name!r}")
    except TypeError as e:
        raise ToolError(f"bad arguments for {name}: {e}") from e
    return json.dumps(out, default=str)
