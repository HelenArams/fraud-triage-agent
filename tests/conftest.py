from __future__ import annotations

import pytest

from fraud_triage.models import Transaction

from factories import make_txn


@pytest.fixture
def txn() -> Transaction:
    return make_txn()
