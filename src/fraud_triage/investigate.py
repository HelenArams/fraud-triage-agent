"""The tool-use loop behind the `investigate` node."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError

from fraud_triage.llm.base import LLMClient, Message, tool_result
from fraud_triage.models import TriageDecision, Verdict
from fraud_triage.prompts import SYSTEM_PROMPT, user_prompt
from fraud_triage.tools import TOOL_SPECS, ToolContext, ToolError, run_tool

log = logging.getLogger(__name__)


@dataclass
class Investigation:
    decision: TriageDecision
    tool_calls: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def investigate(llm: LLMClient, ctx: ToolContext, max_turns: int = 6) -> Investigation:
    messages: list[Message] = [{"role": "user", "content": user_prompt(ctx.txn, ctx.rule_score)}]
    calls: list[str] = []
    notes: list[str] = []

    for _ in range(max_turns):
        turn = llm.create(system=SYSTEM_PROMPT, tools=TOOL_SPECS, messages=messages)
        messages.append({"role": "assistant", "content": turn.content})

        if not turn.tool_calls:
            messages.append(
                {"role": "user", "content": "Please finish by calling submit_decision."}
            )
            continue

        results: list[dict[str, Any]] = []
        for call in turn.tool_calls:
            calls.append(call.name)
            if call.name == "submit_decision":
                try:
                    decision = TriageDecision.model_validate(call.input)
                except ValidationError as e:
                    notes.append("submit_decision failed validation, asked to retry")
                    results.append(tool_result(call.id, f"invalid decision: {e}", is_error=True))
                    continue
                return Investigation(decision, calls, notes)
            try:
                results.append(tool_result(call.id, run_tool(ctx, call.name, call.input)))
            except ToolError as e:
                notes.append(f"tool error: {e}")
                results.append(tool_result(call.id, str(e), is_error=True))
        messages.append({"role": "user", "content": results})

    log.warning("%s: no decision after %d turns", ctx.txn.id, max_turns)
    notes.append(f"no decision after {max_turns} turns")
    fallback = TriageDecision(
        verdict=Verdict.REVIEW,
        confidence=0.0,
        rationale="The analyst did not reach a decision; sending to the queue.",
        evidence=["rule_score"],
    )
    return Investigation(fallback, calls, notes)
