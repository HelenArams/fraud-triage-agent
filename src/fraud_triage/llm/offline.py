"""A rule-based stand-in for the LLM analyst so the whole graph runs without
API keys. It speaks the same tool protocol: first asks for the feature
breakdown and similar cases, then submits a decision built from them."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from fraud_triage.llm.base import AssistantTurn, Message, ToolSpec

BLOCK_SCORE = 0.5


class OfflineAnalyst:
    def __init__(self) -> None:
        self._n = 0

    def _id(self) -> str:
        self._n += 1
        return f"toolu_offline_{self._n}"

    def create(
        self, *, system: str, tools: Sequence[ToolSpec], messages: Sequence[Message]
    ) -> AssistantTurn:
        results = _tool_results(messages)
        if "feature_breakdown" not in results:
            return AssistantTurn(
                content=[
                    {
                        "type": "tool_use",
                        "id": self._id(),
                        "name": "feature_breakdown",
                        "input": {},
                    },
                    {
                        "type": "tool_use",
                        "id": self._id(),
                        "name": "similar_cases",
                        "input": {"k": 3},
                    },
                ],
                stop_reason="tool_use",
            )
        decision = decide(results["feature_breakdown"], results.get("similar_cases", []))
        return AssistantTurn(
            content=[
                {"type": "tool_use", "id": self._id(), "name": "submit_decision", "input": decision}
            ],
            stop_reason="tool_use",
        )


def _tool_results(messages: Sequence[Message]) -> dict[str, Any]:
    names: dict[str, str] = {}
    out: dict[str, Any] = {}
    for m in messages:
        if not isinstance(m.get("content"), list):
            continue
        for block in m["content"]:
            if block.get("type") == "tool_use":
                names[block["id"]] = block["name"]
            elif block.get("type") == "tool_result" and not block.get("is_error"):
                name = names.get(block["tool_use_id"])
                if name:
                    out[name] = json.loads(block["content"])
    return out


def decide(breakdown: dict[str, Any], cases: list[dict[str, Any]]) -> dict[str, Any]:
    rs = breakdown["rule_score"]
    score: float = rs["score"]
    codes = [r["code"] for r in rs["reasons"]]
    takeover = {"new_device", "ip_country_mismatch"} <= set(codes)

    if "card_testing" in codes or (takeover and score >= BLOCK_SCORE):
        verdict = "BLOCK_CARD"
    elif score >= 0.3:
        verdict = "REVIEW"
    else:
        verdict = "CLEAR"

    fraud_cases = [c for c in cases if c["outcome"] == "fraud"]
    parts = [f"Rule score {score:.2f}"]
    if codes:
        parts.append("driven by " + ", ".join(codes))
    if cases:
        top = cases[0]
        parts.append(f"closest past case {top['case_id']} was {top['outcome']}")
    if verdict == "REVIEW" and fraud_cases:
        parts.append(f"{len(fraud_cases)} of {len(cases)} similar cases were fraud")

    evidence = (
        ["rule_score"] + [f"feature:{c}" for c in codes] + [f"case:{c['case_id']}" for c in cases]
    )
    return {
        "verdict": verdict,
        "confidence": round(min(0.95, 0.55 + score * 0.4), 2),
        "rationale": "; ".join(parts) + ".",
        "evidence": evidence,
    }
