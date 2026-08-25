VENV_PYTHON := .venv/bin/python
ifneq (,$(wildcard $(VENV_PYTHON)))
PYTHON ?= $(VENV_PYTHON)
else
PYTHON ?= python3
endif

.PHONY: install test demo api doctor clean ui-install ui ui-test ui-build ui-e2e ui-check

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

ui-install:
	cd frontend && npm install

ui:
	cd frontend && npm run dev

ui-test:
	cd frontend && npm test

ui-build:
	cd frontend && npm run build

ui-e2e:
	cd frontend && npm run test:e2e

ui-check: ui-test ui-build ui-e2e

clean:
	rm -rf data/projects .pytest_cache .ruff_cache
