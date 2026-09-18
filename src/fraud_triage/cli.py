from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from fraud_triage.config import Settings
from fraud_triage.features import compute
from fraud_triage.loader import LoaderError, load_transactions
from fraud_triage.scoring import score

app = typer.Typer(
    no_args_is_help=True, add_completion=False, help="Triage suspicious card transactions."
)
console = Console()


@app.callback()
def main() -> None:
    """Triage suspicious card transactions."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


InputOpt = Annotated[Path, typer.Option("--input", "-i", help="CSV or JSON with transactions")]


@app.command()
def explain(
    txn_id: str,
    input_path: InputOpt = Path("samples/transactions.csv"),
) -> None:
    """Show the feature and rule-score breakdown for one transaction."""
    try:
        txns = load_transactions(input_path)
    except LoaderError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2) from e
    txn = next((t for t in txns if t.id == txn_id), None)
    if txn is None:
        console.print(f"[red]no transaction {txn_id!r} in {input_path}[/red]")
        raise typer.Exit(1)

    feats = compute(txn, txns)
    rs = score(feats, Settings())
    console.print(
        f"[bold]{txn.id}[/bold]  {txn.amount_minor / 100:.2f} {txn.currency}  "
        f"mcc {txn.mcc}  {txn.channel}  card {txn.masked_pan}"
    )
    table = Table("feature", "value")
    for name, value in feats.model_dump().items():
        table.add_row(name, str(value))
    console.print(table)
    if rs.reasons:
        reasons = Table("reason", "weight", "detail")
        for r in rs.reasons:
            reasons.add_row(r.code, f"{r.weight:.2f}", r.detail)
        console.print(reasons)
    console.print(f"rule score [bold]{rs.score:.3f}[/bold] ({rs.band})")


if __name__ == "__main__":
    app()
