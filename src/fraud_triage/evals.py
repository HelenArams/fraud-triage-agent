"""Run the labelled scenarios in evals/cases.yaml through the graph.

    python -m fraud_triage.evals evals/cases.yaml           # offline analyst
    python -m fraud_triage.evals evals/cases.yaml --live    # FRAUD_PROVIDER

Offline mode mostly checks the plumbing (features, routing, guardrails). With
--live the same cases tell you whether the model's verdicts agree with the
labels; expect the occasional REVIEW where the label says BLOCK_CARD.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

from fraud_triage.config import Settings
from fraud_triage.graph import build_graph
from fraud_triage.llm.base import LLMClient
from fraud_triage.llm.factory import get_llm
from fraud_triage.llm.offline import OfflineAnalyst
from fraud_triage.loader import parse_rows
from fraud_triage.runner import thread_config


class Expect(BaseModel):
    route: str
    verdict: str
    band: str | None = None
    note_contains: str | None = None


class EvalCase(BaseModel):
    name: str
    note: str = ""
    txns: list[dict[str, Any]] = Field(min_length=1)
    expect: Expect


@dataclass
class EvalResult:
    case: EvalCase
    band: str
    route: str
    verdict: str
    failures: list[str]

    @property
    def ok(self) -> bool:
        return not self.failures


def load_cases(path: str | Path) -> list[EvalCase]:
    raw = yaml.safe_load(Path(path).read_text())
    defaults = raw.get("defaults", {})
    cases = []
    for c in raw["cases"]:
        c["txns"] = [{**defaults, **t} for t in c["txns"]]
        cases.append(EvalCase.model_validate(c))
    return cases


def run_case(case: EvalCase, llm: LLMClient, settings: Settings) -> EvalResult:
    txns = parse_rows(case.txns, source=case.name)
    target = next(t for t in txns if t.id == case.txns[-1]["id"])
    graph = build_graph(llm, settings, InMemorySaver())
    cfg = thread_config(case.name)
    graph.invoke(
        {
            "txn": target.model_dump(mode="json"),
            "history": [t.model_dump(mode="json") for t in txns if t.id != target.id],
        },
        cfg,
    )
    snap = graph.get_state(cfg)
    values = snap.values
    if snap.next:
        route = "human_review"
        verdict = values["decision"]["verdict"]
    else:
        route = next(p for p in reversed(values["path"]) if p != "report")
        verdict = values["final_verdict"]
    band = values["rule_score"]["band"]
    notes = [*values.get("guardrail_notes", []), *values.get("notes", [])]

    e = case.expect
    failures = []
    if e.band and band != e.band:
        failures.append(f"band {band} != {e.band}")
    if route != e.route:
        failures.append(f"route {route} != {e.route}")
    if verdict != e.verdict:
        failures.append(f"verdict {verdict} != {e.verdict}")
    if e.note_contains and not any(e.note_contains in n for n in notes):
        failures.append(f"no note containing {e.note_contains!r}")
    return EvalResult(case, band, route, verdict, failures)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="fraud_triage.evals")
    ap.add_argument("cases", nargs="?", default="evals/cases.yaml")
    ap.add_argument("--live", action="store_true", help="use FRAUD_PROVIDER instead of offline")
    args = ap.parse_args(argv)

    settings = Settings()
    llm: LLMClient = get_llm(settings) if args.live else OfflineAnalyst()
    console = Console()
    if args.live and settings.provider == "offline":
        console.print("[yellow]--live with FRAUD_PROVIDER=offline, nothing live about it[/yellow]")

    results = [run_case(c, llm, settings) for c in load_cases(args.cases)]
    table = Table("case", "band", "route", "verdict", "result")
    for r in results:
        table.add_row(
            r.case.name,
            r.band,
            r.route,
            r.verdict,
            "[green]ok[/green]" if r.ok else "[red]" + "; ".join(r.failures) + "[/red]",
        )
    console.print(table)
    passed = sum(r.ok for r in results)
    mode = settings.provider if args.live else "offline"
    console.print(f"{passed}/{len(results)} passed ({mode})")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
