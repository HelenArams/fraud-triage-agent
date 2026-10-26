PY ?= .venv/bin/python

.PHONY: venv install lint fmt test eval eval-live demo clean

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

eval:
	$(PY) -m fraud_triage.evals evals/cases.yaml

eval-live:
	$(PY) -m fraud_triage.evals evals/cases.yaml --live

demo:
	$(PY) -m fraud_triage.cli run samples/transactions.csv --out out/

clean:
	rm -rf out .pytest_cache .mypy_cache .ruff_cache
