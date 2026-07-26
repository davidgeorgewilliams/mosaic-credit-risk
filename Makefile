.PHONY: setup train serve app demo test lint typecheck ci

VENV := .venv
PY := $(VENV)/bin/python

setup:
	python3 -m venv $(VENV)
	$(PY) -m pip install -q --upgrade pip
	$(PY) -m pip install -q -e ".[dev]"

train:
	$(PY) -m credit_risk.models.train

serve:
	$(PY) -m uvicorn credit_risk.serving.api:app --reload --port 8000

app:
	$(VENV)/bin/streamlit run src/credit_risk/app/streamlit_app.py

demo:
	$(PY) scripts/demo.py

test:
	$(PY) -m pytest -q

lint:
	$(VENV)/bin/ruff check .

typecheck:
	$(VENV)/bin/mypy src

ci: lint typecheck test
