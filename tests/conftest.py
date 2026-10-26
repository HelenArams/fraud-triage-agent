from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path_factory: pytest.TempPathFactory) -> None:
    # a FRAUD_PROVIDER=anthropic in someone's shell (or a .env) must not leak into tests
    for key in list(os.environ):
        if key.startswith("FRAUD_"):
            monkeypatch.delenv(key)
    monkeypatch.chdir(tmp_path_factory.mktemp("cwd"))
