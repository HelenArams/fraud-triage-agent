"""Markdown + JSON report per transaction."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def to_json(state: Mapping[str, Any]) -> dict[str, Any]:
    txn = state["txn"]
    return {
        "txn_id": txn["id"],
        "card": txn["masked_pan"],
        "amount_minor": txn["amount_minor"],
        "currency": txn["currency"],
        "merchant_id": txn["merchant_id"],
        "rule_score": state.get("rule_score"),
        "path": state.get("path", []),
        "decision": state.get("decision"),
        "guardrail_notes": state.get("guardrail_notes", []),
        "tool_calls": state.get("tool_calls", []),
        "notes": state.get("notes", []),
        "review": state.get("review"),
        "final_verdict": state.get("final_verdict"),
    }


def to_markdown(state: Mapping[str, Any]) -> str:
    r = to_json(state)
    txn = state["txn"]
    rs = r["rule_score"] or {}
    d = r["decision"] or {}
    lines = [
        f"# {r['txn_id']}: {r['final_verdict']}",
        "",
        f"- card: `{r['card']}`",
        f"- amount: {r['amount_minor'] / 100:.2f} {r['currency']} at `{r['merchant_id']}` "
        f"({txn['merchant_country']}, mcc {txn['mcc']}, {txn['channel']})",
        f"- rule score: **{rs.get('score', 0):.2f}** ({rs.get('band', '?')})",
        f"- path: {' > '.join(r['path'])}",
        "",
        "## Signals",
        "",
    ]
    reasons = rs.get("reasons") or []
    lines += [f"- `{x['code']}` ({x['weight']:.2f}): {x['detail']}" for x in reasons] or ["- none"]

    if d:
        lines += [
            "",
            "## Decision",
            "",
            f"**{d['verdict']}**, confidence {d['confidence']:.2f}",
            "",
            d["rationale"],
            "",
            "Evidence: " + (", ".join(f"`{e}`" for e in d["evidence"]) or "none"),
        ]
    if r["tool_calls"]:
        lines += ["", "Tools used: " + ", ".join(r["tool_calls"])]
    if r["guardrail_notes"]:
        lines += ["", "## Guardrails", ""] + [f"- {n}" for n in r["guardrail_notes"]]
    if r["review"]:
        rv = r["review"]
        lines += [
            "",
            "## Human review",
            "",
            f"- {rv['analyst']}: {'approved' if rv['approve'] else 'rejected'}"
            + (f" (verdict set to {rv['verdict']})" if rv.get("verdict") else ""),
        ]
        if rv.get("note"):
            lines.append(f"- note: {rv['note']}")
    return "\n".join(lines) + "\n"


def write(state: Mapping[str, Any], out_dir: str | Path) -> list[str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    txn_id = state["txn"]["id"]
    md = out / f"{txn_id}.md"
    js = out / f"{txn_id}.json"
    md.write_text(to_markdown(state))
    js.write_text(json.dumps(to_json(state), indent=2) + "\n")
    return [str(md), str(js)]
