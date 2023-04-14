.PHONY: test install dev venv clean
.ONESHELL:

VENV=.venv
PYTHON=./$(VENV)/bin/python3.11

test:
	$(PYTHON) -m unittest discover

install: venv
	$(PYTHON) -m pip install .

dev: venv
	$(PYTHON) -m pip install -e .[dev]
	pre-commit install

venv:
	test -d $(VENV) || python3.11 -m venv $(VENV)
	source $(VENV)/bin/activate
	$(PYTHON) -m pip install -U pip setuptools wheel

clean:
	rm -r $(VENV)
	find -iname "*.pyc" -delete
