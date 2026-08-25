VENV_PYTHON := .venv/bin/python
ifneq (,$(wildcard $(VENV_PYTHON)))
PYTHON ?= $(VENV_PYTHON)
else
PYTHON ?= python3
endif

.PHONY: install test demo api doctor clean

install:
	python3 -m venv .venv
	$(VENV_PYTHON) -m pip install -e '.[dev]'

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

demo:
	PYTHONPATH=src $(PYTHON) -m sih26158.cli demo --data-root data/projects

api:
	PYTHONPATH=src $(PYTHON) -m uvicorn sih26158.app:app --host 127.0.0.1 --port 8000 --reload

doctor:
	PYTHONPATH=src $(PYTHON) -m sih26158.cli doctor

clean:
	rm -rf data/projects .pytest_cache .ruff_cache
