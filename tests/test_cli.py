from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from fraud_triage.cli import app

SAMPLE = Path(__file__).resolve().parents[1] / "samples" / "transactions.csv"
runner = CliRunner()


def test_explain() -> None:
    res = runner.invoke(app, ["explain", "t_0024", "--input", str(SAMPLE)])
    assert res.exit_code == 0, res.output
    assert "new_device" in res.output
    assert "rule score 0.539 (mid)" in res.output


def test_explain_unknown_txn() -> None:
    res = runner.invoke(app, ["explain", "t_nope", "--input", str(SAMPLE)])
    assert res.exit_code == 1


def test_run_then_review(tmp_path: Path) -> None:
    out = tmp_path / "out"
    res = runner.invoke(app, ["run", str(SAMPLE), "--out", str(out)])
    assert res.exit_code == 0, res.output
    assert "waiting for review" in res.output
    assert "fraud-triage review t_0024" in res.output
    assert json.loads((out / "t_0001.json").read_text())["final_verdict"] == "CLEAR"
    # the sample has one row exported with a full PAN; it must not reach the reports
    for f in out.glob("*.json"):
        assert "4111111111111111" not in f.read_text()

    res = runner.invoke(
        app, ["review", "t_0024", "--reject", "--note", "customer on holiday", "--out", str(out)]
    )
    assert res.exit_code == 0, res.output
    assert "final verdict CLEAR" in res.output
    assert "customer on holiday" in (out / "t_0024.md").read_text()

    res = runner.invoke(app, ["review", "t_0024", "--approve", "--out", str(out)])
    assert res.exit_code == 1


def test_run_rejects_bad_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("id,ts,cvv\n1,2026-09-01T00:00:00Z,123\n")
    res = runner.invoke(app, ["run", str(bad), "--out", str(tmp_path / "o")])
    assert res.exit_code == 2
    assert "cvv" in res.output
