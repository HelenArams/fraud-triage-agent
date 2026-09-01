"""Load transactions from CSV or JSON.

PANs are masked before anything else touches the row, and unknown columns are
rejected instead of silently dropped - a stray `pan` or `cvv` column in an
export is exactly the kind of thing you want to fail loudly on.
"""

from __future__ import annotations

import csv
import json
import logging
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from fraud_triage.models import Transaction
from fraud_triage.pan import scrub

log = logging.getLogger(__name__)

COLUMNS = frozenset(Transaction.model_fields)


class LoaderError(ValueError):
    pass


def _clean_row(row: Mapping[str, Any], where: str) -> dict[str, Any]:
    unknown = set(row) - COLUMNS
    if unknown:
        raise LoaderError(f"{where}: unknown column(s): {', '.join(sorted(unknown))}")
    out: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, str):
            value, n = scrub(value)
            if n:
                log.warning("%s: masked full PAN in column %r", where, key)
        out[key] = value
    return out


def parse_rows(rows: Iterable[Mapping[str, Any]], source: str = "<rows>") -> list[Transaction]:
    txns: list[Transaction] = []
    for i, row in enumerate(rows, start=1):
        where = f"{source} row {i}"
        cleaned = _clean_row(row, where)
        try:
            txns.append(Transaction.model_validate(cleaned))
        except ValidationError as e:
            raise LoaderError(f"{where}: {e}") from e
    return sorted(txns, key=lambda t: t.ts)


def load_transactions(path: str | Path) -> list[Transaction]:
    p = Path(path)
    if p.suffix.lower() == ".json":
        data = json.loads(p.read_text())
        if not isinstance(data, list):
            raise LoaderError(f"{p}: expected a JSON array of transactions")
        return parse_rows(data, p.name)
    with p.open(newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None:
            raise LoaderError(f"{p}: empty file")
        return parse_rows(reader, p.name)
