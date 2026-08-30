PY ?= .venv/bin/python

.PHONY: venv install lint fmt test eval demo clean

venv:
	python3.12 -m venv .venv

install:
	$(PY) -m pip install -e ".[dev]"

lint:
	$(PY) -m ruff check src tests
	$(PY) -m ruff format --check src tests
	$(PY) -m mypy --strict src

fmt:
	$(PY) -m ruff format src tests
	$(PY) -m ruff check --fix src tests

test:
	$(PY) -m pytest

demo:
	$(PY) -m fraud_triage.cli run samples/transactions.csv --out out/

clean:
	rm -rf out .pytest_cache .mypy_cache .ruff_cache
